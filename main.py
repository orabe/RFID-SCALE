from data_functions import *
import argparse

def parse_arguments():
    DESCRIPTION = 'A collection of scripts for analyzing mice weights measured by an RFID-Scale system.'
    parser = argparse.ArgumentParser(prog="RFID-Scale", description=DESCRIPTION)
    parser.add_argument('-f', '--input_file', type=str, metavar='', required=True,
                        help='The absolute path to the input data file in CSV format.')
    parser.add_argument('-l', '--event_window_length', type=int, metavar='', required=False, default=1000,
                        help='The number of consecutive event values (detections) to group together to calculate the zero-weight. Default: 1000.')
    parser.add_argument('-m', '--min_animal_weight', type=float, metavar='', required=False, default=10.0,
                        help='The minimum weight that an animal must have to be included in the output DataFrame. Default: 10.0.')
    parser.add_argument('-t', '--rfid_time_window', type=int, metavar='', required=False, default=2,
                        help='The time window (in seconds) used to match RFID readings with load cell data. Default: 2 seconds.')
    parser.add_argument('-i', '--analysis_interval', type=int, metavar='', required=False, default=24, choices=[12, 24],
                        help='The length of the interval in hours for which the data should be analyzed. Valid choices are 12 or 24. Default: 24.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    data = args.input_file
    event_window_length = args.event_window_length
    min_animal_weight = args.min_animal_weight
    RFID_time_window = args.rfid_time_window
    analysis_interval = args.analysis_interval
    
        # create results directory.
    if not os.path.exists(os.path.join(os.getcwd(), 'results', 'jpg', f'{analysis_interval}h')):
        os.makedirs(os.path.join(os.getcwd(), 'results', 'jpg', f'{analysis_interval}h'))
        os.makedirs(os.path.join(os.getcwd(), 'results', 'csv'))


    df = read_data(data) 
    df = clean_data(df) 
    
    # Calculate the zero weight for groups of detections.
    arr_eventDuration, arr_eventDuration_min = calc_zero_weight(df, event_window_length=1000)   # calculate the minimum weight of each segment
    
    # Plotting raw weights.
    plot_zero_weights(arr_eventDuration_min, event_window_length, figsize=(15,5), fontsize=15) 
    plot_weight_timeWindow(df, title='Raw weight data', plot_name='raw_data',RFID_y=np.median(arr_eventDuration_min), ylabel='Weight (w0 + w_animals)')

    # Filter out the null weight.
    df_no_zero_weights = filterout_zero_weight(df, arr_eventDuration, arr_eventDuration_min)
    plot_weight_timeWindow(df_no_zero_weights, RFID_y=0.5, title='Weights with no zero-weight measurements', plot_name='no_zero_weight', ylabel='Animal weight')

    # Now keep only weights higher than a threshold
    df_valid_weights = keep_weights_above_threshold(df_no_zero_weights, min_animal_weight)
    df_valid_weights.to_csv(os.path.join('results', 'csv', 'df_valid_weights.csv'))
    plot_weight_timeWindow(df_valid_weights, RFID_y=min_animal_weight, title=f'Valid weights (>{int(min_animal_weight)} g)', plot_name='valid_weights', ylabel='Animal weight (g)')
    
    estimate_weight_noorshams_method(df_no_zero_weights, min_animal_weight, RFID_time_window, analysis_interval)

    print(f"\nCongratulations! The analysis has been successfully completed. You can find the results at {os.path.join(os.getcwd(), 'results')}.\n")
    
    
if __name__ == "__main__":
    main()