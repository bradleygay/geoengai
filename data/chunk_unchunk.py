#For LFS: chunking .nc files
import os
import xarray as xr
import pandas as pd
import math

input_dir = "/Users/bgay/geoengai/data/cmip6"
output_dir = "/Users/bgay/geoengai/data/cmip6chunked"
os.makedirs(output_dir, exist_ok=True)
time_chunk_size = 100  # Adjust based on your needs (e.g., 100 time steps per file)
def chunk_netcdf(file_path, output_dir, time_chunk_size):
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    ds = xr.open_dataset(file_path)
    if "time" not in ds.dims:
        print(f"No time dimension in {file_path}. Skipping...")
        return
    total_time_steps = ds.dims["time"]
    for start in range(0, total_time_steps, time_chunk_size):
        end = min(start + time_chunk_size, total_time_steps)
        chunk = ds.isel(time=slice(start, end))
        chunk_filename = f"{base_name}_chunk_{start}_{end}{ext}"
        chunk.to_netcdf(os.path.join(output_dir, chunk_filename))
        print(f"Saved: {chunk_filename}")
    ds.close()
for file in os.listdir(input_dir):
    if file.endswith(".nc"):
        print(f"Processing {file}...")
        chunk_netcdf(os.path.join(input_dir, file), output_dir, time_chunk_size)
        
####################################################################################################

#For LFS: chunking .pkl files
import os
import pandas as pd
df = pd.read_pickle("/Users/bgay/geoengai/era5_df.pkl")
chunks = [df.iloc[i:i+1000000] for i in range(0, len(df), 1000000)]
for idx, chunk in enumerate(chunks):
    chunk.to_pickle(f”era5_chunk_{idx}.pkl")

df = pd.read_pickle("/Users/bgay/geoengai/cruts408.pkl”)
chunks = [df.iloc[i:i+1000000] for i in range(0, len(df), 1000000)]
for idx, chunk in enumerate(chunks):
    chunk.to_pickle(f”cruts408_chunk_{idx}.pkl")

####################################################################################################
#MORE CHUNKING#
####################################################################################################

#For LFS: chunking .nc files
import os
import xarray as xr
import cftime
import math
import pandas as pd

# Directory containing NetCDF files
input_dir = "/Users/bgay/geoengai/cmip6"
output_dir = "/Users/bgay/geoengai/cmip6chunked"
os.makedirs(output_dir, exist_ok=True)
# File size limit in bytes (e.g., 2GB = 2147483648 bytes)
#file_size_limit = 2 * 1024**3  # 2GB
file_size_limit = 256**3  # 16.8MB

# Function to estimate dataset size in memory
def estimate_size_in_bytes(dataset):
    size = 0
    for var in dataset.variables.values():
        size += var.size * var.dtype.itemsize
    return size

def safely_decode_time(ds):
    if "time" in ds.variables:
        try:
            time_units = ds["time"].attrs.get("units", "days since 2000-01-01")
            calendar = ds["time"].attrs.get("calendar", "standard")
            ds["time"] = xr.conventions.decode_cf_variable(
                "time", ds["time"].values, time_units, calendar, use_cftime=True
            )
        except Exception as e:
            print(f"Failed to decode time for {ds}: {e}")
    return ds

# Function to split NetCDF file into chunks based on file size limit
def chunk_netcdf_by_size(file_path, output_dir, file_size_limit):
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    # Open the NetCDF file with decode_times=False to handle custom calendar
    ds = xr.open_dataset(file_path, decode_times=False)
    # Decode time safely
    ds = safely_decode_time(ds)
    # Estimate total size of the dataset
    total_size = estimate_size_in_bytes(ds)
    print(f"Total size of {filename}: {total_size / (1024**3):.2f} GB")
    # Calculate the number of chunks needed
    num_chunks = math.ceil(total_size / file_size_limit)
    print(f"Splitting into {num_chunks} chunks...")
    if "time" not in ds.dims:
        print(f"No time dimension in {file_path}. Skipping...")
        return
    # Determine time steps per chunk
    total_time_steps = ds.dims["time"]
    time_steps_per_chunk = math.ceil(total_time_steps / num_chunks)
    # Split the dataset along the time dimension
    for i in range(num_chunks):
        start = i * time_steps_per_chunk
        end = min((i + 1) * time_steps_per_chunk, total_time_steps)
        chunk = ds.isel(time=slice(start, end))
        chunk_filename = f"{base_name}_chunk_{i + 1}{ext}"
        chunk.to_netcdf(os.path.join(output_dir, chunk_filename))
        print(f"Saved chunk: {chunk_filename} (time: {start}-{end})")
    ds.close()

# Iterate over all NetCDF files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".nc"):
        print(f"Processing {file}...")
        chunk_netcdf_by_size(os.path.join(input_dir, file), output_dir, file_size_limit)
        
####################################################################################################
        
#For LFS: chunking .pkl files (ERA5)
import os
import pickle
import math
import pandas as pd

# Directory containing .pkl files
input_dir = "/Users/bgay/geoengai/era5"
output_dir = "/Users/bgay/geoengai/era5chunked"
os.makedirs(output_dir, exist_ok=True)

# File size limit in bytes (e.g., 2GB = 2147483648 bytes)
#file_size_limit = 2 * 1024**3  # 2GB
file_size_limit = 256**3  # 16.8MB

# Function to estimate the size of a pickle object
def get_pickle_size(obj):
    return len(pickle.dumps(obj))

# Function to chunk pandas DataFrame
def chunk_dataframe(df, file_size_limit):
    total_size = get_pickle_size(df)
    num_chunks = math.ceil(total_size / file_size_limit)
    rows_per_chunk = math.ceil(len(df) / num_chunks)
    return [
        df.iloc[i : i + rows_per_chunk] for i in range(0, len(df), rows_per_chunk)
    ]

# Function to split and save pickle files
def chunk_pickle_by_size(file_path, output_dir, file_size_limit):
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    # Load the pickle file
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        total_size = get_pickle_size(data)
        print(f"Total size of {filename}: {total_size / (1024**3):.2f} GB")
        # Chunk the DataFrame
        chunks = chunk_dataframe(data, file_size_limit)
        print(f"Splitting {filename} into {len(chunks)} chunks...")
        # Save each chunk
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk_{i+1}{ext}"
            with open(os.path.join(output_dir, chunk_filename), "wb") as chunk_file:
                pickle.dump(chunk, chunk_file)
            print(f"Saved chunk: {chunk_filename}")
    # Add additional cases here for other data types if needed
    else:
        print(f"Unsupported data type for chunking: {type(data)}")
        return

# Iterate over all `.pkl` files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".pkl"):
        print(f"Processing {file}...")
        chunk_pickle_by_size(os.path.join(input_dir, file), output_dir, file_size_limit)

####################################################################################################

#For LFS: chunking .pkl files (CRU TS408)
import os
import pickle
import math
import pandas as pd

# Directory containing .pkl files
input_dir = "/Users/bgay/geoengai/cruts408"
output_dir = "/Users/bgay/geoengai/cruts408chunked"
os.makedirs(output_dir, exist_ok=True)

# File size limit in bytes (e.g., 2GB = 2147483648 bytes)
#file_size_limit = 2 * 1024**3  # 2GB
file_size_limit = 256**3  # 16.8GB

# Function to estimate the size of a pickle object
def get_pickle_size(obj):
    return len(pickle.dumps(obj))

# Function to chunk pandas DataFrame
def chunk_dataframe(df, file_size_limit):
    total_size = get_pickle_size(df)
    num_chunks = math.ceil(total_size / file_size_limit)
    rows_per_chunk = math.ceil(len(df) / num_chunks)
    return [
        df.iloc[i : i + rows_per_chunk] for i in range(0, len(df), rows_per_chunk)
    ]

# Function to split and save pickle files
def chunk_pickle_by_size(file_path, output_dir, file_size_limit):
    filename = os.path.basename(file_path)
    base_name, ext = os.path.splitext(filename)
    # Load the pickle file
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        total_size = get_pickle_size(data)
        print(f"Total size of {filename}: {total_size / (1024**3):.2f} GB")
        # Chunk the DataFrame
        chunks = chunk_dataframe(data, file_size_limit)
        print(f"Splitting {filename} into {len(chunks)} chunks...")
        # Save each chun
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk_{i+1}{ext}"
            with open(os.path.join(output_dir, chunk_filename), "wb") as chunk_file:
                pickle.dump(chunk, chunk_file)
            print(f"Saved chunk: {chunk_filename}")
    # Add additional cases here for other data types if needed
    else:
        print(f"Unsupported data type for chunking: {type(data)}")
        return

# Iterate over all `.pkl` files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".pkl"):
        print(f"Processing {file}...")
        chunk_pickle_by_size(os.path.join(input_dir, file), output_dir, file_size_limit)



####################################################################################################
####################################################################################################
#UNCHUNKING#
####################################################################################################
####################################################################################################

import os
import xarray as xr
from cftime import num2date
# Directory containing NetCDF chunks
chunk_dir = "/Users/bgay/geoengai/data/cmip6chunks/ch4"
output_file = "/Users/bgay/geoengai/data/cmip6_ch4_reassembled_file.nc"
# Get a sorted list of chunk files
chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".nc")])
datasets = []
for f in chunk_files:
    try:
        # Open the dataset without decoding times
        ds = xr.open_dataset(f, decode_times=False)
        # Decode times manually if the 'time' variable exists
        if "time" in ds.variables and "units" in ds["time"].attrs:
            time_units = ds["time"].attrs["units"]
            calendar = ds["time"].attrs.get("calendar", "standard")
            ds["time"] = num2date(ds["time"].values, units=time_units, calendar=calendar)
        datasets.append(ds)
    except Exception as e:
        print(f"Error loading file {f}: {e}")
# Concatenate along the time dimension
combined_dataset = xr.concat(datasets, dim="time")
# Save the reassembled dataset to a single NetCDF file
combined_dataset.to_netcdf(output_file)
print(f"Reassembled file saved at: {output_file}")

####################################################################################################

#Unchunking .pkl files (ERA5)
import os
import pickle
import pandas as pd
# Directory containing pickle chunks
chunk_dir = "/Users/bgay/geoengai/data/era5chunks"
output_file = "/Users/bgay/geoengai/data/era5_reassembled_file.pkl"
# Initialize a list to store chunks
chunks = []
# Get a sorted list of `.pkl` files
chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".pkl")])
# Check if `.pkl` files are found
if not chunk_files:
    print("No `.pkl` files found in the directory.")
else:
    print(f"Found {len(chunk_files)} `.pkl` files.")
# Load and validate chunks
for f in chunk_files:
    try:
        with open(f, "rb") as chunk_file:
            data = pickle.load(chunk_file)
            # Check if the chunk is a DataFrame
            if isinstance(data, pd.DataFrame):
                chunks.append(data)
                print(f"Loaded file {f} with type {type(data)}")
            else:
                print(f"Skipping unsupported type: {type(data)} in file {f}")
    except Exception as e:
        print(f"Error loading file {f}: {e}")
# Concatenate all valid chunks
if chunks:
    combined_data = pd.concat(chunks, ignore_index=True)
    print(f"Combined data contains {len(combined_data)} rows.")
else:
    print("No valid DataFrame objects to concatenate.")
# Save the combined DataFrame
try:
    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)
    print(f"Reassembled DataFrame saved to {output_file}")
except Exception as e:
    print(f"Error saving combined data: {e}")
# Verify the structure of the combined DataFrame
print(combined_data.info())
print(combined_data.head())

####################################################################################################

#Unchunking .pkl files (CRU TS408)
import os
import pickle
import pandas as pd
# Directory containing pickle chunks
chunk_dir = "/Users/bgay/geoengai/data/cruts408chunks"
output_file = "/Users/bgay/geoengai/data/cryts408_reassembled_file.pkl"
# Initialize a list to store chunks
chunks = []
# Get a sorted list of `.pkl` files
chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".pkl")])
# Check if `.pkl` files are found
if not chunk_files:
    print("No `.pkl` files found in the directory.")
else:
    print(f"Found {len(chunk_files)} `.pkl` files.")
# Load and validate chunks
for f in chunk_files:
    try:
        with open(f, "rb") as chunk_file:
            data = pickle.load(chunk_file)
            # Check if the chunk is a DataFrame
            if isinstance(data, pd.DataFrame):
                chunks.append(data)
                print(f"Loaded file {f} with type {type(data)}")
            else:
                print(f"Skipping unsupported type: {type(data)} in file {f}")
    except Exception as e:
        print(f"Error loading file {f}: {e}")
# Concatenate all valid chunks
if chunks:
    combined_data = pd.concat(chunks, ignore_index=True)
    print(f"Combined data contains {len(combined_data)} rows.")
else:
    print("No valid DataFrame objects to concatenate.")
# Save the combined DataFrame
try:
    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)
    print(f"Reassembled DataFrame saved to {output_file}")
except Exception as e:
    print(f"Error saving combined data: {e}")
# Verify the structure of the combined DataFrame
print(combined_data.info())
print(combined_data.head())
