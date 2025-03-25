import argparse
import os
import numpy as np
import pandas as pd
import h5py
import logging
import swifter
import sys
import warnings

# Set up logging
logging.basicConfig(filename='gnss_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Log a message when the script starts
logger.info("GNSS processing script has started.")

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Process 15-minute GNSS chunks.")
parser.add_argument("--start_time", type=str, required=True, help="Start time for the chunk to process in YYYY-MM-DD HH:MM:SS format.")
parser.add_argument("--file_path", type=str, required=True, help="Path to the GNSS data file.")
args = parser.parse_args()

# Convert start time to pandas timestamp
start_time = pd.to_datetime(args.start_time)

# Calculate end time directly using the start time
end_time = start_time + pd.Timedelta(minutes=15)

# Log the start and end time of the file:
logger.info(f'Start Time: {start_time} and End Time: {end_time}')

# Log that the data is being looked at
logger.info(f'Starting to open Madrigal data file')

# Open the Madrigal HDF5 file to check column names
with h5py.File(args.file_path, 'r') as f:
    dataset = f['Data/Table Layout']
    column_names = dataset.dtype.names
    logger.info(f"Dataset Columns: {column_names}")

    # Convert entire dataset to DataFrame
    df = pd.DataFrame(dataset[:], columns=column_names)

# Convert 'hour' and 'min' to integers
df['hour'] = df['hour'].astype(int)
df['minute'] = df['min'].astype(int)  # Renamed to 'minute'
df['second'] = df['sec'].astype(int)  # Renamed to 'second'

# Convert the time columns into a pandas timestamp
timestamps = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])

# Apply the mask for the start_time and end_time
mask = (timestamps >= start_time) & (timestamps < end_time)

# Filter data using the mask
df_filtered = df[mask]

# Select only the necessary columns
columns_needed = ['year', 'month', 'day', 'hour', 'minute', 'second', 'gps_site', 
                   'gdlatr', 'gdlonr', 'sat_id', 'elm', 'azm', 'los_tec', 'dlos_tec']

gnss_dataframe_filtered = df_filtered[columns_needed]

# Now, apply the elevation constraint after loading all chunks
logger.info("Applying elevation constraint (elevation >= 15).")
gnss_dataframe_elv_filtered = gnss_dataframe_filtered[gnss_dataframe_filtered['elm'] >= 15]

# Copy the filtered DataFrame (precautionary)
gnss_dataframe_elv_filtered = gnss_dataframe_elv_filtered.copy()

# Reset the index to make sure row numbers line up
gnss_dataframe_elv_filtered.reset_index(drop=True, inplace=True)

# Rename columns to make them more readable (optional step)
gnss_dataframe_elv_filtered = gnss_dataframe_elv_filtered.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day', 'hour': 'Hour', 'minute': 'Minute', 'second': 'Second'})

gnss_dataframe_elv_filtered['time'] = pd.to_datetime(gnss_dataframe_elv_filtered[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])

# Log data info
logger.info(f'Data processing complete. DataFrame created. Shape: {gnss_dataframe_elv_filtered.shape}')
logger.info(f"Checking for NaNs in the initial DataFrame: {gnss_dataframe_elv_filtered.isna().sum()}")

# Perform coordinate conversion (ECEF)
logger.info(f'Moving to coord conversion')

# Add AENeAS path
sys.path.append('/rds/projects/t/themendr-j-brown/aeneas/aeneas')

# Now import the class
from coordinates import Coords

# Define sat range
SATELLITE_RANGE_KM = 20200  # Example satellite range in km

# Function to convert receiver and satellite coordinates to ECEF
def convert_to_ecef(row):
    # Receiver coordinates in ECEF
    receiver_coords = Coords([row["gdlonr"], row["gdlatr"], 0], 'GEODET')
    receiver_ecef = receiver_coords.convert('ECEF').data.squeeze()  # Convert to ECEF (1D array)
    
    # Compute satellite position relative to receiver in ECEF
    satellite_ecef_vector = receiver_coords.reaToEcefDir(SATELLITE_RANGE_KM, row["elm"], row["azm"])
    
    # Convert tuple to a NumPy array and flatten it
    satellite_ecef_vector = np.stack(satellite_ecef_vector, axis=-1).flatten()  
    
    # Compute absolute satellite ECEF position
    satellite_ecef = receiver_ecef + satellite_ecef_vector  # Shift to Earth's center
    
    return np.hstack([receiver_ecef, satellite_ecef])  # Return both as 1D array

# Apply the coordinate conversion function across the DataFrame
gnss_dataframe_elv_filtered[['X_rec', 'Y_rec', 'Z_rec', 'X_sat', 'Y_sat', 'Z_sat']] = gnss_dataframe_elv_filtered.swifter.apply(
    lambda row: pd.Series(convert_to_ecef(row)), axis=1
)

# Log the first few rows of converted coordinates
logger.info("Receiver and satellite ECEF conversion complete.")
logger.info(gnss_dataframe_elv_filtered[['X_rec', 'Y_rec', 'Z_rec', 'X_sat', 'Y_sat', 'Z_sat']].head())

# Check for any NaNs after conversion
logger.info(f"Checking for NaNs after satellite coordinates conversion: {gnss_dataframe_elv_filtered.isna().sum()}")

# Print the first few rows of the final processed DataFrame
logger.info("Displaying the first few rows of the processed data:")
logger.info(gnss_dataframe_elv_filtered.head())

    
# Get unique station names to check they're all still there
station_names = gnss_dataframe_elv_filtered['gps_site'].unique()
logger.info(f"Unique stations: {station_names}")
logger.info(f"Number of unique stations: {len(station_names)}")


# Here we loop through all the individual receiver/satellite links and average over a time interval.
# It should give us a time averaged sTEC value and an average satellite position for that time interval
# Helps cut down the size of data for AENeAS which is good and also consecutive measurements are not really independent
# so averaging helps with this, but increases the representivity error. (more benefit than negative I think)

# Here we loop through all the individual receiver/satellite links and average over a time interval
logger.info("Starting the averaging of GNSS data over 5-minute intervals...")

# Suppress warnings related to setting values on copies of slices from a DataFrame
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# Set the window size for 5 minutes (300 seconds)
window_size = pd.Timedelta(seconds=300)

# Initialize an empty list to store the binned data
binned_data = []

# Initialize the list of fixed time intervals (5-minute intervals)
fixed_intervals = pd.date_range(start=start_time, periods=3, freq=window_size)

# Iterate through each satellite and receiver pair
for station in station_names:
    # Filter the data for the current station
    station_data = gnss_dataframe_elv_filtered[gnss_dataframe_elv_filtered['gps_site'] == station]
    
    # Copy to avoid changing the original DataFrame
    station_data = station_data.copy()

    # Attempt to create the 'time' column from Year, Month, Day, Hour, Minute, and Second
    station_data['time'] = pd.to_datetime(station_data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')

    # Check if any NaN values were created during the conversion of 'time'
    if station_data['time'].isna().any():
        logger.warning(f"NaN values created in 'time' column for station: {station}. This may be due to invalid date/time data.")
    
    # Drop rows with NaN values in 'time' column
    station_data = station_data.dropna(subset=['time'])

    # Sort by time
    station_data = station_data.sort_values(by='time')

    logger.info(f"Processing data for receiver: {station}")

    # Iterate through each satellite (1 to 32)
    for sat_id in range(1, 33):  # Loop through satellite IDs 1 to 32
        # Filter the data for the current satellite
        sat_data = station_data[station_data['sat_id'] == sat_id]
        
        # Ensure there is data for this satellite
        if sat_data.empty:
            continue

        # Sliding window: generate bins for each 5:00 window using fixed intervals
        for start_time in fixed_intervals:
            if pd.isna(start_time):  # Check if start_time is NaT
                continue

            # Window end time
            end_time = start_time + window_size
            if pd.isna(end_time):  # Check if end_time is NaT
                continue

            # Filter data for the current time window
            window_data = sat_data[(sat_data['time'] >= start_time) & (sat_data['time'] < end_time)]

            if len(window_data) > 0:
                # Calculate average TEC
                avg_los_tec = window_data['los_tec'].mean()
                
                # Calculate standard deviation of TEC, handling case for a single data point
                if len(window_data) > 1:
                    std_los_tec = window_data['los_tec'].std()
                else:
                    std_los_tec = window_data['dlos_tec'].iloc[0]  # Use dlos_tec from the single row

                # Average satellite position over the window
                avg_x_sat = window_data['X_sat'].mean()
                avg_y_sat = window_data['Y_sat'].mean()
                avg_z_sat = window_data['Z_sat'].mean()

                # Store the results in the list
                binned_data.append({
                    'Start_Time': start_time,
                    'End_Time': end_time,
                    'gps_site': station,  # Receiver station name
                    'Sat_ID': sat_id,
                    'Avg_los_tec': avg_los_tec,
                    'Std_los_tec': std_los_tec,
                    'Avg_X_sat': avg_x_sat,
                    'Avg_Y_sat': avg_y_sat,
                    'Avg_Z_sat': avg_z_sat,
                    'X_rec': window_data['X_rec'].iloc[0],  # Receiver position from first row of window
                    'Y_rec': window_data['Y_rec'].iloc[0],
                    'Z_rec': window_data['Z_rec'].iloc[0]
                })

                # Log progress
                logger.debug(f"Receiver {station} and Sat_ID {sat_id} data averaged for time step: {start_time} to {end_time}")
            else:
                logger.warning(f"No data found for receiver {station}, Sat_ID {sat_id} during the time window: {start_time} to {end_time}")

# Convert the list of results into a DataFrame
binned_df = pd.DataFrame(binned_data)

# Display the binned data and log the output
logger.info("Binned data conversion complete.")
logger.info(f"Checking for NaNs in the DataFrame after sat coords conversion: {binned_df.isna().sum()}")
logger.info(f"Displaying the last few rows of the binned data:")
logger.info(binned_df.tail())
# Check for any NaN values in each column, log it if found
for col in binned_df.columns:
    if binned_df[col].isna().any():
        logger.info(f"Column '{col}' has NaN values.")
        logger.info(f"Rows with NaN values in '{col}':\n{binned_df[binned_df[col].isna()]}")

# Define the new directory path
output_dir = '20170604_Madrigal_GNSS_AENeAS_formatted_data'

# Ensure the directory exists (create it if it doesn't)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the output filename with formatted date-time based on the end time of the first row in binned_df
output_filename = os.path.join(output_dir, f"data_{end_time.strftime('%Y%m%d_%H%M')}.h5")

# Log the filename and start the file saving process
logger.info(f"Saving data to HDF5 file: {output_filename}")

# Open an HDF5 file in write mode
try:
    with h5py.File(output_filename, 'w') as hdf:
        # Save the variables we want in the correct format

        # Convert columns to the appropriate types and store them in the HDF5 file
        hdf.create_dataset('recID', data=binned_df['gps_site'].values.astype('S'))  # Convert to string
        hdf.create_dataset('rxLoc', data=binned_df[['X_rec', 'Y_rec', 'Z_rec']].values)  # Receiver locations
        hdf.create_dataset('satID', data=binned_df['Sat_ID'].values.astype('int32'))  # Ensure 32-bit integer
        hdf.create_dataset('STEC', data=binned_df['Avg_los_tec'].values.astype('float64'))  # STEC (average LOS TEC)
        hdf.create_dataset('satLoc', data=binned_df[['Avg_X_sat', 'Avg_Y_sat', 'Avg_Z_sat']].values)  # Satellite locations
        hdf.create_dataset('STEC_std', data=binned_df['Std_los_tec'].values)  # STEC error 

        # Add metadata as attributes (using Start_Time as the reference time for the file)
        hdf.attrs['Date'] = binned_df['Start_Time'].iloc[0].strftime('%Y-%m-%d')
        hdf.attrs['Hour'] = binned_df['Start_Time'].iloc[0].hour
        hdf.attrs['TimeWindow'] = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}"


    # Log success message
    logger.info(f"Data successfully saved to {output_filename}")

except Exception as e:
    # Log any exceptions that occur during file saving
    logger.error(f"Error occurred while saving data to {output_filename}: {str(e)}")

# Print the success message to the console (for debugging and terminal confirmation)
logger.info(f"Data saved as {output_filename}")
