import pandas as pd
import zipfile
import os

def preprocess_data(zip_path: str, station: str) -> pd.DataFrame:
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("air_quality_data")
    
    # Find all CSV files in the extracted folder
    csv_files = [os.path.join("air_quality_data", f) for f in os.listdir("air_quality_data") if f.endswith('.csv')]
    
    # Read and concatenate all CSVs
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Add 'station' column if missing
        if 'station' not in df.columns:
            df['station'] = os.path.basename(file).split('_')[2] if len(os.path.basename(file).split('_')) > 2 else station
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    # Filter by station (now always present)
    data = data[data['station'] == station].copy()
    
    # Combine date and time columns into a single datetime column if present
    if {'year', 'month', 'day', 'hour'}.issubset(data.columns):
        data['datetime'] = pd.to_datetime(
            data[['year', 'month', 'day', 'hour']]
        )
        data = data.drop(['year', 'month', 'day', 'hour'], axis=1)
    
    # Handle missing values: fill with column mean for numeric, mode for categorical
    for col in data.columns:
        if data[col].dtype.kind in 'biufc':  # numeric
            data[col] = data[col].fillna(data[col].mean())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Save cleaned data
    data.to_csv('air_quality_cleaned.csv', index=False)
    
    return data

if __name__ == '__main__':
    # Example usage
    cleaned = preprocess_data('beijing+multi+site+air+quality+data.zip', 'Aotizhongxin')
    print(cleaned.head())