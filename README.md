# RFID-Scale System User Guide

The RFID-Scale is a collection of scripts for analyzing weight data measured by an RFID-scale system. The scripts are designed to clean and filter the raw weight data, and estimate the weight of individual animals.

# Repository Contents
```sh
RFID-Scale :
  ├───data
  │   └───input_data.csv
  │
  ├───results
  │   ├───csv
  │   └───jpg
  │
  ├───README.md
  ├───data_functions.py
  └───main.py
```

# System Requirements

The RFID-Scale system requires Python 3.x and several Python packages including Pandas, Matplotlib, and NumPy.

# Installation

1. Download or clone the `RfidScale` repository to your local machine using the following command in the shell:
```sh
git clone https://github.com/orabe/RFID-SCALE.git
```
2. Install Python 3.x if it is not already installed.
3. Install required Python packages using `pip` or `conda`:
```sh
pip install pandas matplotlib numpy argparse
```

# Usage

1. Navigate to the directory containing the RFID-Scale scripts in the command line e.g.:
```sh
cd C:\Users\YourUserName\Desktop\RfidScale
```
2. Run the python script `main.py` by typing the following command in your shell:
```sh
python main.py -f <input_file_path> [optional arguments]
```
3. Replace <input_file_path> with the absolute path to your input data file in CSV format.
4. Use optional arguments to customize the analysis:
   * `-l/--event_window_length` sets the number of consecutive event values (detections) to group together to calculate the zero-weight. Default: 1000.
   * `-m/--min_animal_weight` sets the minimum weight that an animal must have to be included in the output DataFrame. Default: 10.0.
   * `-t/--rfid_time_window` specifies the time window, in seconds, used to match RFID readings with load cell data. The default value is 2 seconds, but you can adjust this parameter by providing an integer value after the flag. For example, `-t 5` would set the time window to 5 seconds. This parameter is optional, and if you don't specify it, the default value will be used.
   * `-i/--analysis_interval` sets the length of the interval in hours for which the data should be analyzed. Valid choices are 12 or 24. Default: 24.

The script will output a folder containing the analyzed results in CSV format and several plots.

# Output

The output of the RFID-Scale system includes:

1. A CSV file containing the cleaned and filtered weight data.
2. Several plots including:
   * Raw weight data with zero-weight measurements.
   * Weight data with no zero-weight measurements.
   * Valid weights (above a set threshold).
   * Estimated animal weights using the [Noorshams](https://www.sciencedirect.com/science/article/abs/pii/S0165027017301218) method for 12 or 24 hours.
