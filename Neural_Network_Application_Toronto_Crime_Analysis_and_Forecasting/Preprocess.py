import os
import pandas as pd
import json
from datetime import datetime


def get_latest_file(directory, file_prefix):
    """
    Retrieves the latest file with the specified prefix in the given directory.

    Args:
        directory (str): Directory to search for files.
        file_prefix (str): File prefix to filter files.

    Returns:
        str: Path to the latest file or None if no files are found.
    """
    files = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith('.json')]
    if files:
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
        return os.path.join(directory, latest_file)
    return None


def json_to_dataframe(file_path):
    """
    Converts JSON data to a pandas DataFrame and performs basic cleaning.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract 'features' from JSON and normalize
    if 'features' in data:
        df = pd.json_normalize(data['features'])
    else:
        raise ValueError("Unexpected JSON structure. Check the input file.")

    # Flatten column names
    df.columns = df.columns.str.replace('attributes.', '', regex=False)
    df.columns = df.columns.str.replace('geometry.', '', regex=False)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Convert Unix timestamps to datetime
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'], unit='ms')
    if 'occ_date' in df.columns:
        df['occ_date'] = pd.to_datetime(df['occ_date'], unit='ms')

    return df


def preprocess_data(df):
    """
    Performs preprocessing steps such as filtering, date formatting, and column selection.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Mapping for months and days
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    day_mapping = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }

    # Convert text-based months and days to numeric values
    df['report_month'] = df['report_month'].map(month_mapping)
    df['occ_month'] = df['occ_month'].map(month_mapping)
    df['report_dow'] = df['report_dow'].map(day_mapping)
    df['occ_dow'] = df['occ_dow'].map(day_mapping)

    # Handle missing values
    df = df.fillna({
        'occ_year': df['report_year'],
        'occ_month': df['report_month'],
        'occ_day': df['report_day'],
    })

    # Drop rows with critical missing values
    df = df.dropna(subset=['report_month', 'occ_month', 'report_year', 'occ_year'])

    # Convert numeric columns to integers
    numeric_columns = [
        'report_year', 'report_month', 'report_day', 'report_hour',
        'occ_year', 'occ_month', 'occ_day', 'occ_hour'
    ]
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Filter for data after 2014
    df = df[df['occ_year'] >= 2014]

    # Remove rare classes in 'offence'
    class_counts = df['offence'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['offence'].isin(valid_classes)]

    # Combine into datetime columns
    df['report_datetime'] = pd.to_datetime(
        dict(
            year=df['report_year'],
            month=df['report_month'],
            day=df['report_day'],
            hour=df['report_hour']
        ),
        errors='coerce'
    )
    df['occurred_datetime'] = pd.to_datetime(
        dict(
            year=df['occ_year'],
            month=df['occ_month'],
            day=df['occ_day'],
            hour=df['occ_hour']
        ),
        errors='coerce'
    )

    # Drop rows with invalid datetime conversion
    df = df.dropna(subset=['report_datetime', 'occurred_datetime'])

    return df


if __name__ == "__main__":
    # Specify the directory and file prefix
    directory = os.path.dirname(os.path.abspath(__file__))
    file_prefix = "major_crime_indicators_"

    # Get the latest JSON file
    latest_file_path = get_latest_file(directory, file_prefix)

    if latest_file_path:
        print(f"Processing file: {latest_file_path}")

        # Convert JSON to DataFrame
        try:
            df = json_to_dataframe(latest_file_path)

            # Perform preprocessing
            df = preprocess_data(df)

            # Save the preprocessed data as CSV
            output_file = f"preprocessed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(os.path.join(directory, output_file), index=False)
            print(f"Preprocessed data saved to: {output_file}")
        except Exception as e:
            print(f"Error while processing: {e}")
    else:
        print("No JSON file found. Please run Fetch.py first.")
