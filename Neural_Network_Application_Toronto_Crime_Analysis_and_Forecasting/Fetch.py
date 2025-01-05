import os
import requests
import json
from datetime import datetime

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Fetch data and handle pagination
def fetch_data(filename_prefix="major_crime_indicators"):
    """
    Fetches data from the API with pagination and saves it to a JSON file.

    Args:
        filename_prefix (str): Prefix for the output filename.

    Returns:
        str: Path to the saved JSON file or None if the request fails.
    """
    url = "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Major_Crime_Indicators_Open_Data/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": "4326",
        "f": "json",
        "resultOffset": 0,
        "resultRecordCount": 2000  # Default API limit
    }

    all_data = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None

        data = response.json()
        if "features" not in data or not data["features"]:
            break

        all_data.extend(data["features"])
        params["resultOffset"] += params["resultRecordCount"]

    # Construct the filename with a timestamp
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(script_directory, filename)
    with open(file_path, 'w') as file:
        json.dump({"features": all_data}, file)
    print(f"Data saved to {file_path} with {len(all_data)} records.")
    return file_path

if __name__ == "__main__":
    fetch_data()
