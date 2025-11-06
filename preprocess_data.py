import pandas as pd
import os
import glob
import numpy as np

print("Starting data pre-processing...")

# Define the path to the folder containing the data files
folder_path = r'C:\Users\JohnL\OneDrive - store-dot.com\Documents\Temperature\Temperature Data'
all_files = glob.glob(os.path.join(folder_path, "*.txt"))

if not all_files:
    print(f"Error: No .txt files found in the '{folder_path}' folder. Exiting.")
else:
    print(f"Found {len(all_files)} files to load and process.")
    df_list = []
    for filename in all_files:
        try:
           # Read each file into a temporary dataframe
            temp_df = pd.read_csv(filename, sep='\t', skiprows=3)
            
            # --- NEW: Clean column names by removing '-oC' suffix ---
            # Create a dictionary to map old names to new names, e.g., {'T01-oC': 'T01'}
            rename_map = {col: col.replace('-oC', '') for col in temp_df.columns if '-oC' in col}
            if rename_map:
                temp_df.rename(columns=rename_map, inplace=True)
                print(f"Cleaned column names for {os.path.basename(filename)}")
                
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if df_list:
        # Combine all DataFrames in memory
        df = pd.concat(df_list, ignore_index=True)
        
        # Perform all the cleaning and type conversion
        df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        df.drop_duplicates(subset=['datetime'], inplace=True)
        
        # Define and find available temperature columns
        all_possible_sensors = [
            'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08',
            'T09', 'T10', 'T11', 'T12', 'L-Env', 'R-Env'
        ]
        temperature_columns = [col for col in all_possible_sensors if col in df.columns]

        # --- CORRECTED LOGIC: Replace values <= 0 with NaN ---
        if temperature_columns:
            # Create a boolean DataFrame that is True only for values <= 0
            mask = df[temperature_columns] <= 0
            
            # Count how many such values exist before changing them
            values_to_replace = mask.sum().sum()
            print(f"Found {values_to_replace} data points with values <= 0.")

            # Use the mask to replace only the targeted values with NaN
            df[temperature_columns] = df[temperature_columns].mask(mask, np.nan)
            print("Replaced those values with null (NaN).")
        # --- END CORRECTION ---

        # Set and sort the index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Save the cleaned DataFrame to a fast Feather file
        df.reset_index(inplace=True) 
        output_filename = 'cleaned_data.feather'
        df.to_feather(output_filename)
        
        print("-" * 30)
        print(f"Successfully processed {len(df)} rows.")
        print(f"Cleaned data has been saved to '{output_filename}'")
        print("-" * 30)