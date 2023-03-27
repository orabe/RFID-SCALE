import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from time import strftime
import matplotlib.ticker as ticker
import os

def read_data(files_path):
    """
    Load data from a CSV file.

    Args:
        files_path (str): The path to the CSV file containing the weight and RFID data.

    Returns:
        pandas.DataFrame: A DataFrame containing the weight and RFID data.
    """
    # this code is was removed for privacy reasons
    pass
   



def convert_DJ_to_unix_timestampe(ts):
    """
    Convert a timestamp from Duplin Julian format to Unix format.

    Args:
        ts (str): A timestamp in Duplin Julian format.

    Returns:
        datetime: The converted timestamp in Unix format.

    Raises:
        ValueError: If the input timestamp is not in a valid format.
    """
    # this code is was removed for privacy reasons
    pass



def clean_data(df):
    """
    Sorts the input dataframe by datetime, converts datetime from Dublin Julian time to UTC, removes the first two rows, and
    keeps only weight and RFID detections.

    Args:
        df (DataFrame): A pandas DataFrame containing the data to be cleaned.

    Returns:
        DataFrame: A cleaned pandas DataFrame.
    """
    
    # this code is was removed for privacy reasons
    pass



def plot_weight_timeWindow(df, RFID_y, title, ylabel, plot_name, start_time_window=np.nan, end_time_window=np.nan, y_value='eventDuration'):

    """
    Plot weight measurements over time for a specified time window, with RFID detections indicated.

    Args:
        df (dataframe): contains weight and RFID data, sorted by DateTime
        RFID_y (int): y-value for displaying RFID detection markers
        title (str): title of the plot
        ylabel (str): y-axis label for the plot
        plot_name (str): name of the file to save the plot as
        start_time_window (float): (optional) starting time for the time window in minutes
        end_time_window (float): (optional) ending time for the time window in minutes
        y_value (str): (optional) name of the column to use for y-values; default is 'eventDuration'

    Returns:
        None
    """
    
    # this code is was removed for privacy reasons
    pass


def plot_zero_weights(df, event_window_length, figsize, fontsize):
    """
    Plot the estimated "Zero" weight (w0) of the animals.

    Args:
        df (pd.DataFrame): The data to plot.
        event_window_length (int): The length of the time window in minutes.
        figsize (Tuple[int, int]): The figure size in inches.
        fontsize (int): The font size of the plot.

    Returns:
        None
    """

    f = plt.figure(figsize=figsize)

    plt.plot(df)
    plt.plot(np.full((df.shape[0]), np.median(df)), label='median')
    plt.title(f'Estimated "Zero" weight (w0).\nEach value denotes the minimum "zero" weight of {int(event_window_length / 500)}-minutes detections', fontsize=fontsize)
    plt.ylabel('W0', fontsize=fontsize)
    plt.xlabel('--> Time (4 days)', fontsize=fontsize)
    plt.xticks([])
    plt.legend()

    plt.savefig(os.path.join('results', 'jpg' ,f'w0_{int(event_window_length / 500)}-minutes.png'))
    plt.draw()
      

def calc_zero_weight(df, event_window_length):   
    """Calculate the minimum raw weight value in each segment of events.

    Args:
        event_window_length (int): The number of consecutive event values (detections) to be grouped together to calculate the minimum weight value. 500 events correspond to approximately 1 minute.
        df (pandas.DataFrame): A DataFrame holding the raw data.

    Returns:
        tuple: A tuple containing two 1D arrays:
            - arr_eventDuration: An array containing the raw weight values for each segment.
            - arr_eventDuration_min: An array containing the minimum raw weight value for each segment.
    """

    # this code is was removed for privacy reasons
    pass
    
    

def filterout_zero_weight(df, arr_eventDuration, arr_eventDuration_min):    
    """Filter out events with zero weight.

    Args:
        df (pandas dataframe): A dataframe holding raw data.
        arr_eventDuration (ndarray): An array containing the event durations in each event window.
        arr_eventDuration_min (ndarray): An array containing the minimum event duration in each event window.

    Returns:
        pandas dataframe: A dataframe with the events that have non-zero weight, where the event durations have been adjusted to be zero-centered. The dataframe includes both weight and RFID events and is sorted by timestamp.
    """
    # this code is was removed for privacy reasons
    pass



def keep_weights_above_threshold(df_no_zero_weights, min_animal_weight):
    """Keep only animal weights that are above a minimum threshold.

    Args:
        df_no_zero_weights (pandas.DataFrame): Dataframe with no zero weights.
        min_animal_weight (float): The minimum weight that an animal must have to be included in the output DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame containing only the animal weights that are greater than or equal to min_animal_weight.
    """
    df_valid_weights = df_no_zero_weights.loc[df_no_zero_weights['eventDuration'] >= min_animal_weight].reset_index(drop=True) # as long as all values in RFID['eventDuration']>10, then this should not cause any problem.
    return df_valid_weights



def estimate_weight(weights, kKERNEL_WIDTH, kMIN_WEIGHT, kMAX_WEIGHT, kHIST_BINSIZE, histNumBins):
    """This function takes in a list of weight readings from an RFID tag and returns an estimated weight for the series of readings based on the method described in the paper by Noorshams et al. (2017). The function uses a Gaussian kernel generated using Pascal's triangle, and computes the histogram of the input data using numpy. The histogram is made cumulative and the derivative is computed to find the part with the steepest slope. The derivative is then smoothed using the Gaussian kernel, and the location of the maximum value is found using numpy's argmax function. The center of the bin containing the maximum value is returned as the estimated weight.

    Args:
        weights (list): A list of weights.
        kKERNEL_WIDTH (int): The width of the Gaussian kernel used to smooth the derivative.
        kMIN_WEIGHT (float): The minimum weight value for the histogram.
        kMAX_WEIGHT (float): The maximum weight value for the histogram.
        kHIST_BINSIZE (float): The bin size for the histogram.
        histNumBins (int): The number of bins for the histogram.

    Returns:
        result (float): The estimated weight for the series of readings.
    """
    
    # makes a Gaussian kernel using Pascal's triangle
    def gKernel (nK):
        kernel = np.zeros([nK], dtype = np.float32)
        kernel [0] = (1/2)**(nK -1)
        for ii in range (1,nK, 1):
            for ij in range (nK-1, 0, -1):
                kernel [ij] += kernel[ij-1]
        return kernel

    kernel = gKernel (kKERNEL_WIDTH) 

    # copy array into numpy array for histogram and gradient
    numArray = np.array(weights, dtype=np.float32)
    
    # make histogram of data for this mouse - hist is the data, WtVals is the bin boundaries
    hist, WtVals=np.histogram(numArray,range =(kMIN_WEIGHT, kMAX_WEIGHT), bins = histNumBins)
    
    # make histogram cumulative. Part of histogram with most data will have steepest slope
    cumulative = np.cumsum(hist)
    
    # make derivative of histogram to find part with steepest slope
    diffArray = np.gradient(cumulative, 2)
    
    # smooth the derivative lots
    diffArray=np.convolve(diffArray, kernel, mode = 'same')
    
    # find the location of the maximum value in the histogram
    position = np.argmax(diffArray)
    
    # The center of the bin containing the max value is returned as the result
    result = (WtVals[position] + WtVals[position + 1])/2
    
    return result
    
    
def estimate_weight_noorshams_method(df_no_zero_weights, min_animal_weight, RFID_time_window, analysis_interval):
    """Estimates the weight of animals based on load cell data and RFID readings.

    Args:
    - df_no_zero_weights: pandas DataFrame containing load cell data without zero weights
    - min_animal_weight: minimum animal weight to consider in grams
    - RFID_time_window: time window (in seconds) used for matching RFID readings with load cell data
    - analysis_interval: length of the time interval for weight estimation, in hours (either 12 or 24)

    Returns:
    None
    
    The function uses Noorsham's method to estimate the weight of animals based on load cell data and RFID readings.
    The load cell data is first filtered to exclude zero weights and values below the minimum weight threshold.
    The RFID readings are then used to split the load cell data into time intervals of fixed length (12 or 24 hours).
    For each time interval and animal ID, the function estimates the weight using a kernel density estimation method
    and plots the load cell data over time, along with the estimated weight.
    The results are saved as images in the 'results' folder.
    """

    print("- Starting final analysis to estimate weights using Noorshams' method. This may take a few seconds ...")
    kKERNEL_WIDTH = 17 # width of the smoothing kernel used for the derivative of the histogram
    kMIN_WEIGHT = min_animal_weight # (= 10), weights below this value are excluded
    kMAX_WEIGHT = 75 # (= 50), weights above this value are excluded
    kHIST_BINSIZE = 0.1 # width of the bins in the cumulative histogram, in grams
    histNumBins = int ((kMAX_WEIGHT- kMIN_WEIGHT)/kHIST_BINSIZE)

    # for values >10g
    # RFID_df = df_valid_weights[df_valid_weights['unitLabel'] == 'RFID1']#.reset_index(drop=True)

    # for values >0g
    RFID_df = df_no_zero_weights[df_no_zero_weights['unitLabel'] == 'RFID1']#.reset_index(drop=True)

    IDs = RFID_df['IdRFID'].unique()

    # used to split the data to 24/12 hours.
    first_interval = RFID_df['DateTime'].iloc[0]
    second_interval = first_interval + dt.timedelta(hours=analysis_interval)

    days = (df_no_zero_weights['DateTime'].apply(lambda x: x.strftime('%Y-%m-%d'))).unique()
    days = [dt.datetime.strptime(day, '%Y-%m-%d') for day in days]

    # append a day at the end of the days list
    days.append(days[-1] + dt.timedelta(days=1))

    # iterate over analysis_interval (either 24h or 12h) and export the results in separte images. Each image has len(ID) plots.
    for i in range(int((len(days)-2 ) * (24/analysis_interval))):    
        f, axes = plt.subplots(len(IDs), figsize=(15,30))
        epochs = np.zeros(len(IDs), int)
        id_epochs_dict = dict(zip(IDs, epochs))
        id_weights_perInterval_dict = dict(zip(IDs, ([] for _ in IDs)))
        
        RFID_df_interval = RFID_df[(RFID_df['DateTime'] >= first_interval) & (RFID_df['DateTime'] < second_interval)]

        # for values >10g
        # df_valid = df_valid_weights[(df_valid_weights['DateTime'] >= first_interval) & (df_valid_weights['DateTime'] < second_interval)]

        # for values >0g
        df_valid = df_no_zero_weights[(df_no_zero_weights['DateTime'] >= first_interval) & (df_no_zero_weights['DateTime'] < second_interval)]
            
        for RFID_DateTime, RFID_IdRFID in zip(RFID_df_interval['DateTime'], RFID_df_interval['IdRFID']):
            # epochs interval
            window_before = RFID_DateTime - dt.timedelta(seconds=RFID_time_window)
            window_after = RFID_DateTime + dt.timedelta(seconds=RFID_time_window)

            sliced_df = df_valid[(df_valid['DateTime'] >= window_before) & (df_valid['DateTime'] < window_after)]
            sliced_df = sliced_df.sort_values(by=['DateTime'], ignore_index=True)

            # calc estimated weight for each window of weights
            sliced_weights_without_RFID = sliced_df[sliced_df['unitLabel'] != 'RFID1']['eventDuration']
            
            if len(sliced_weights_without_RFID) != 0:
                
                id_weights_perInterval_dict[RFID_IdRFID].extend(sliced_weights_without_RFID.values)
                
                delta_ts = np.array(sliced_df[sliced_df['unitLabel'] != 'RFID1']['DateTime']) - sliced_df[sliced_df['unitLabel'] != 'RFID1']['DateTime'].iloc[0]
                delta_ts = pd.Series(delta_ts).dt.total_seconds()

                id_index = np.where(IDs == RFID_IdRFID)[0][0]
                id_epochs_dict[RFID_IdRFID] += 1
                
                axes[id_index].plot(delta_ts, sliced_weights_without_RFID,linewidth=0.5)
                axes[id_index].set_title(f'Animal ID: {RFID_IdRFID}, {id_epochs_dict[RFID_IdRFID]} epochs')
                
        
        for key, value in id_weights_perInterval_dict.items():
            est_weight = estimate_weight(value, kKERNEL_WIDTH, kMIN_WEIGHT, kMAX_WEIGHT, kHIST_BINSIZE, histNumBins)
            id_index = np.where(IDs == key)[0][0]
            axes[id_index].axhline(est_weight, linewidth=2, c = 'red', label= f'Estimated weight: {np.round(est_weight, 1)}')
            axes[id_index].legend()
            
            
        f.suptitle(f'from: {first_interval.strftime("%Y-%m-%d -- %H:00:00")}   to   {second_interval.strftime("%Y-%m-%d -- %H:00:00")} \n\n\n',     fontweight="bold")
        f.supxlabel('Time(s)')
        f.supylabel('Load cell value')
        
        f.tight_layout()
        plt.savefig(os.path.join('results', 'jpg', f'{analysis_interval}h', f'{first_interval}---{second_interval}.png'))
        
                # df_gradient_method = pd.concat([sliced_df, df_gradient_method])
        
        # 24/12 hours interval
        first_interval = second_interval
        second_interval = first_interval + dt.timedelta(hours=analysis_interval)

    plt.draw()