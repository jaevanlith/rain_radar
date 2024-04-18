import numpy as np
import pandas as pd
import math
from statistics import mean
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Event:
    '''
    Rainfall event class
    '''
    
    def __init__(self, start_time, end_time, stations, reflect_min, reflect_avg, reflect_max, rain_intens_min, rain_intens_avg, rain_intens_max):
        
        # Set time properties
        self.start_time = start_time
        self.end_time = end_time
        self.duration = int((end_time - start_time).total_seconds() // 3600)

        # Set station properties
        self.stations = stations
        self.num_stations = len(stations)
        
        # Set reflectivity properties
        self.reflect_min = reflect_min
        self.reflect_avg = reflect_avg
        self.reflect_max = reflect_max

        # Set rain properties
        self.rain_intens_min = rain_intens_min
        self.rain_intens_avg = rain_intens_avg
        self.rain_intens_max = rain_intens_max
        self.rain_cum_avg = rain_intens_avg * self.duration

        # Set type
        if self.rain_intens_avg <= 5.0:
            self.type = "light"
        elif self.rain_intens_avg <= 25.0:
            self.type = "moderate"
        elif self.rain_intens_avg <= 50.0:
            self.type = "heavy"
        else:
            self.type = "extreme"

    def to_string(self):
        '''
        Method to print the attributes of the event.
        '''
        return (
            'Start time: ' + str(self.start_time) + 
            '\nEnd time: ' + str(self.end_time) + 
            '\nDuration: ' + str(self.duration) +
            '\nStations: ' + str(self.stations) +
            '\nNum stations: ' + str(self.num_stations) +
            '\nReflectivity (min, avg, max) in Z: (' + str(self.reflect_min) + ', ' + str(self.reflect_avg) + ', ' + str(self.reflect_max) + ')' +
            '\nRain intensity (min, avg, max) in mm/h: (' + str(self.rain_intens_min) + ', ' + str(self.rain_intens_avg) + ', ' + str(self.rain_intens_max) + ')' +
            '\nRain cummulative avg: ' +  str(self.rain_cum_avg) +
            '\nType: ' + self.type
        )


def select_events_single_station(station, vals, datetime, radar_df, max_no_rain, min_rain_threshold=0.1, temporal_res='6min'):
    '''
    Method to select events per station.

    @param station str: Name of station
    @param vals list[float]: Rain data of given station for one year
    @param datetime list[date]: List of dates and times per hour for the entire year
    @param radar_df DataFrame: Radar data for one year of all stations
    @param max_no_rain float: Maximum number of hours without rain within one event
    @param k float: Rainfall threshold

    @return events list[Event]: List of events at this station for the given year
    @return Z list[float]: List of reflectivity values per hour within events at this station
    @return R list[float]: List of rainfall values per hour within events at this station
    '''
    events = []
    Z = []
    R = []

    i = 0
    while i < len(vals):
        # Check if value is above threshold
        if vals.iloc[i] >= min_rain_threshold:
            # Init statistics for event
            candidate_event = []
            consecutive_hours_no_rain = 0

            # Loop over remaining values
            for j in range(i, len(vals)):
                # Retrieve value
                val = vals.iloc[j]
                # Add to event
                candidate_event.append(val)

                # Check if value is above threshold
                if val >= min_rain_threshold:
                    # Reset statistic
                    consecutive_hours_no_rain = 0
                else:
                    # Update statistic
                    consecutive_hours_no_rain += 1
                    # Check if max hours without rain is exceeded
                    if consecutive_hours_no_rain > max_no_rain:
                        # Set event time
                        start_time = datetime[i]
                        end_time = datetime[j - max_no_rain]

                        # Set event reflectivity
                        reflect_vals = list(radar_df.loc[start_time:end_time][station].values)[:-1]
                        if len(reflect_vals) == 0:
                            reflect_min = float('nan')
                            reflect_avg = float('nan')
                            reflect_max = float('nan')
                        else:
                            reflect_min = min(reflect_vals)
                            reflect_avg = mean(reflect_vals)
                            reflect_max = max(reflect_vals)

                        # Set event rain
                        rain_vals = candidate_event[:-consecutive_hours_no_rain]
                        if len(rain_vals) == 0:
                            rain_intens_min = float('nan')
                            rain_intens_avg = float('nan')
                            rain_intens_max = float('nan')
                        else:
                            rain_intens_min = min(rain_vals)
                            rain_intens_avg = mean(rain_vals)
                            rain_intens_max = max(rain_vals)

                        # Check if no nan values in event, otherwise event is discarded
                        if not math.isnan(rain_intens_avg):
                            # Create new event
                            new_event = Event(start_time, end_time, [station], reflect_min, reflect_avg, reflect_max, rain_intens_min, rain_intens_avg, rain_intens_max)

                            # Add to events list
                            events.append(new_event)

                            # Store rain intensity values
                            R += rain_vals

                            # Check if 6min sampling vs 60min sampling is still correct
                            if temporal_res == '6min':
                                if len(reflect_vals) != 10*len(rain_vals):
                                    raise Exception("Radar dataframe should be sampled per 6min and rain gauge dataframe per 60min. \
                                                    Please check if this is the case!\n \
                                                    The problem occured at station: ",  station, ", from: ", start_time, ", until: ", end_time)
                            elif temporal_res == '60min' or temporal_res == '1H':
                                if len(reflect_vals) != len(rain_vals):
                                    raise Exception("Radar dataframe should be sampled per 60min and rain gauge dataframe per 60min. \
                                                    Please check if this is the case!\n \
                                                    The problem occured at station: ",  station, ", from: ", start_time, ", until: ", end_time)
                            
                            # Store reflectivity values
                            Z += reflect_vals

                        # Continue events selection after end of new event
                        i = j + 1
                        break
            
        # Go to next timestep
        i += 1

    return events, Z, R


def merge_overlapping_events(events):
    '''
    Method to merge single-station events that overlap in time.

    @param events list[Event]: Events detected per station

    @return result list[Event]: Events including multiple stations
    '''
    # Sort the events on start time
    events.sort(key=lambda e: e.start_time)

    # Plot single events sorted
    plot_single_events(events)

    # Init list to store the merged events and store first event
    result = [events[0]]

    # Loop over events (and skip the first)
    for e in events[1:]:
        # Check if current event and last merged event overlap
        if e.start_time <= result[-1].end_time:
            # Merge events and overwrite last one in result
            result[-1] = merge_two_events(result[-1], e)
        else:
            # If event is later than last merged, add event to the result
            result.append(e)

    return result


def merge_two_events(e1, e2):
    '''
    Method to merge two events.

    @param e1 Event: Event to be merged.
    @param e2 Event: Event to be merged.

    @return merged_event Event: Merged event.
    '''
    # Pick earliest start time
    start_time = min(e1.start_time, e2.start_time)
    # Pick latest end time
    end_time = max(e1.end_time, e2.end_time)
    # Concat lists of stations
    stations = e1.stations + e2.stations

    # Recompute reflectivity
    reflect_min = min(e1.reflect_min, e2.reflect_min)
    reflect_avg = (e1.reflect_avg * e1.duration + e2.reflect_avg * e2.duration) / (e1.duration + e2.duration)
    reflect_max = max(e1.reflect_max, e2.reflect_max)

    # Recompute rain intensity
    rain_intens_min = min(e1.rain_intens_min, e2.rain_intens_min)
    rain_intens_avg = (e1.rain_intens_avg * e1.duration + e2.rain_intens_avg * e2.duration) / (e1.duration + e2.duration)
    rain_intens_max = max(e1.rain_intens_max, e2.rain_intens_max)

    # Create new event
    merged_event = Event(start_time, end_time, stations, reflect_min, reflect_avg, reflect_max, rain_intens_min, rain_intens_avg, rain_intens_max)

    return merged_event


def plot_peak_over_threshold(rain_df, threshold=0.5):
    '''
    Method to visualize peak over threshold method.

    @param rain_df DataFrame: Rain gauge data.
    '''
    # Select a station
    station = rain_df.columns[1]
    vals = rain_df[station]

    # Get time column
    datetime = rain_df.index

    for i in range(len(vals)):
        ax = plt.subplot()
        ax.bar(datetime[i], vals[i], color='C0')
    
    ax.xaxis_date()
    ax.axhline(y=threshold, color='r', linestyle='dashed', label='Threshold') 

    ax.set_xlabel('Time')
    ax.set_ylabel('Reflectivity (dBZ)')
    ax.set_title('Rain gauge data from station: ' + station)
    plt.legend()
    plt.show()


def plot_single_events(events):
    '''
    Method to visualize all single events.

    @param events list[Event]: List of events to plot.
    '''
    # Create new plot
    fig = plt.figure()
    ax = fig.add_subplot()

    # Loop over events
    for i in range(100):
        # Retrieve event
        e = events[i]

        # Convert to matplotlib date representation
        start = mdates.date2num(e.start_time)
        end = mdates.date2num(e.end_time)
        width = end - start

        # Plot rectangle
        rect = Rectangle((start, len(events)-i), width, 0.8)
        ax.add_patch(rect)

    # Assign date locator / formatter to the x-axis to get proper labels
    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.DateFormatter('%m%d %H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Set configs
    ax.set_aspect('auto')
    ax.use_sticky_edges = False
    ax.autoscale(enable=True, tight=False)
    plt.xticks(rotation=60)
    ax.set_xlabel('Time')
    ax.set_ylabel('Event')
    ax.set_yticklabels([])
    ax.set_title('First 100 events sorted on start time')

    # Show
    plt.show()


def select_all_events(rain_df, radar_df, max_no_rain, min_rain_threshold=0.1):
    '''
    Method that selects rain events from the rain gauge data.

    @param rain_df DataFrame: Rain data for one year of all stations
    @param radar_df DataFrame: Radar data for one year of all stations
    @param max_no_rain int: Maximum number of hours without rain within one event
    @param k int: Rainfall threshold

    @return events list[Event]: List of events for the given year
    @return Z array[float]: Vector of reflectivity values per hour per station within all events
    @return R array[float]: Vector of rainfall values per hour per station within all events
    '''
    # Init event list
    events = []
    Z = []
    R = []

    # Get time column
    datetime = rain_df.index

    # Loop over stations and correspoding values
    for (station, vals) in rain_df.items():
        # Select events for single station
        single_events, single_Z, single_R = select_events_single_station(station, vals, datetime, radar_df, max_no_rain, min_rain_threshold)
        events += single_events
        Z += single_Z
        R += single_R

    # Merge single-station events that overlap in time
    if len(events) > 1:
        events = merge_overlapping_events(events)

    # Convert lists to numpy arrays for convience in calibration
    Z = np.array(Z)
    R = np.array(R)

    # Reshape reflectivity to vectors of 10, so a vector of values per hour
    new_shape = (len(Z) // 10, 10)
    Z = Z.reshape(new_shape)

    # Throw exception if dimensions of Z and R do not correspond
    if len(Z) != len(R):
        raise Exception("Lengths of reflectivity vector Z and rainfall vector R not equal: " + str(len(Z)) + " != " + str(len(R)))
    
    # Filter out pairs where reflectivity is 0
    non_zero_mask = np.all(Z != 0, axis=1)
    R = R[non_zero_mask]
    Z = Z[non_zero_mask]

    # Filter out pairs where reflectivity is nan
    non_nan_mask = np.all(~np.isnan(Z), axis=1)
    R = R[non_nan_mask]
    Z = Z[non_nan_mask]

    # Filter out pairs where rain intensity is 0
    Z = Z[R != 0]
    R = R[R != 0]

    # Filter out pairs where rain intensity is nan
    Z = Z[~np.isnan(R)]
    R = R[~np.isnan(R)]

    return events, Z, R


def write_events_to_excel(events, save_path):
    '''
    Method save events in excel file.

    @param events list[Event]: List of events for the given year
    @param save_path str: Path where file should be saved (including .xlsx extension)
    '''

    # Set column names
    columns = ['start_time', 'end_time', 'duration', \
                'stations', 'num_stations', \
                'reflect_min_dBZ', 'reflect_avg_dBZ', 'reflect_max_dBZ', \
                'rain_initens_min', 'rain_intens_avg', 'rain_intens_max', 'rain_cum_avg', \
                'type']
    
    # Init empty DataFrame
    events_df = pd.DataFrame(columns=columns)
    # Init func for Z -> dBZ conversion
    to_dBZ = lambda x : 0 if x == 0 else max(10*math.log10(x), 0)

    # Loop over events
    for e in events:
        # Convert attributes from event into row
        event_row = [e.start_time, e.end_time, e.duration, \
                    e.stations, e.num_stations, \
                    to_dBZ(e.reflect_min), to_dBZ(e.reflect_avg), to_dBZ(e.reflect_max), \
                    e.rain_intens_min, e.rain_intens_avg, e.rain_intens_max, e.rain_cum_avg, \
                    e.type]
        
        # Store in dataframe
        events_df.loc[len(events_df)] = event_row

    # Write DataFrame to excel
    events_df.to_excel(save_path)