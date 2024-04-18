import pandas as pd
import numpy as np
from PIL import Image
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def prepare_radar_data(radar_data_path, year, noise_threshold, hail_threshold, save_path, months=None, days=None, temporal_res='6min'):
    '''
    Method to load radar data from csv files.

    @param radar_data_path str: Directory where the radar data is stored.
    @param year int: Year to analyse the data from.
    @param noise_threshold float: Threshold underneath which is considered noise (in dBZ).
    @param hail_threshold float: Threshold above which is considered hail (in dBZ).
    @param save_path str: Path to save the radar data to.
    @param months list: List of months to analyse.
    @param days list: List of days to analyse.
    @param temporal_res str: Temporal resolution to resample the data to.

    @return df DataFrame: Reflectivity over time for all stations.
    '''

    # Get the pixels corresponding to each station
    station_loc = pd.read_excel(radar_data_path + '/extract_radarpixel/raingauge_coordinate.xlsx', sheet_name='Sheet1')
    subset = station_loc[['STN_ID', 'pixel_y','pixel_x']]
    location_list = [tuple(x) for x in subset.to_numpy()]

    # Filter out negative pixels (outside radar region)
    location_list_filtered = []
    # Loop over stations and pixel coordinates
    for (id, pixel_y, pixel_x) in location_list:
        # Check if both pixels inside region
        if pixel_y >= 0 and pixel_x >= 0:
            # Store stations inside region
            location_list_filtered.append((id, pixel_y, pixel_x))
    # Update location list
    location_list = location_list_filtered

    # Set the root path of the year under investigation
    radar_png_path = radar_data_path + '/radar_png/' + str(year)

    # Init columns
    columns = [x[0] for x in location_list]
    # Init csv file
    radar_df = pd.DataFrame(columns=columns)
    radar_df.insert(loc=0, column='Datetime', value=[])
    radar_df.set_index('Datetime', inplace=True)
    radar_df.to_csv(save_path)

    # If months not specified, derive from directory
    if months is None:
        months = os.listdir(radar_png_path)
    months.sort()

    # Check if days specified
    if days is None:
        days_specified = False
    else:
        days_specified = True

    # Loop over months
    for month in months:
        # Init DataFrame
        radar_df = pd.DataFrame(columns=columns)
        # Init datetime column
        DateTime = []

        # Set path for this month
        radar_png_month_path = radar_png_path + '/' + month

        # If days not specified, derive from directory
        if not days_specified:
            days = os.listdir(radar_png_month_path)
        days.sort()

        # Loop over days
        for day in days:
            print('Currently at: ', day, '/', month, '/', year)
            # Set path for this day
            radar_png_day_path = radar_png_month_path + '/' + day
            # Get list of files and count
            filelist = os.listdir(radar_png_day_path)

            # Loop over all radar files
            for file in filelist:
                # Try to open the png file, otherwise print that it gave issues
                try:
                    # Get datetime from file name   
                    Year = file[0:4]
                    Month = file[4:6]
                    Day = file[6:8]
                    Hour = file[8:10]
                    Minute = file[10:12]

                    # Load radar data from file
                    data_png = Image.open(radar_png_day_path + '/' + file)
                    data_png_numpy = np.array(data_png)
                    df_data = pd.DataFrame(data_png_numpy)

                    # Init row
                    extract_data = np.arange(len(location_list), dtype=float)

                    # Loop over locations of stations
                    for location in location_list:
                        # Get the data corresponding to the pixel locations
                        pixel_x = int(location[2])
                        pixel_y = int(location[1])
                        value = df_data.iloc[pixel_y, pixel_x]
                        extract_data[location_list.index(location)] = value

                    # Store in dataframe
                    radar_df.loc[len(radar_df)] = extract_data

                    # Format and store datetime
                    Date = Year + "-" + Month + "-" + Day + ' ' + Hour + ':' + Minute
                    DateTime.append(datetime.strptime(Date, "%Y-%m-%d %H:%M"))

                except:
                    print("Unable to open: " + radar_png_day_path + '/' + file)

        # Set datetime as index column
        radar_df.insert(loc=0, column='Datetime', value=DateTime)
        radar_df.set_index('Datetime', inplace=True)
        radar_df = radar_df.sort_index(axis=1)

        # Filter noise and hail
        radar_df[radar_df < noise_threshold] = 0
        radar_df[radar_df > hail_threshold] = hail_threshold

        # Convert dBZ to Z
        radar_df = 10**(radar_df/10)
        radar_df = radar_df.replace(1,0)

        # Average over hours
        radar_df = radar_df.resample('6min').mean()

        # Remove duplicates
        radar_df = radar_df.loc[:,~radar_df.columns.duplicated()].copy()

        # Write to csv
        radar_df.to_csv(save_path, mode='a', index=True, header=False)
        
        # Clear DataFrame and Datetime
        radar_df = None
        DateTime = []