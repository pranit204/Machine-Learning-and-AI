import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths and file prefix
data_directory = "/Users/pranitsanghavi/PycharmProjects/PythonProject/Toronto_Police_Service_Major_Crime_Indicators/Data"
file_prefix = "preprocessed_data_"

def get_latest_csv(data_directory, file_prefix):
    """
    Retrieves the latest CSV file with the specified prefix in the given directory.

    Args:
        data_directory (str): Directory containing CSV files.
        file_prefix (str): File prefix to filter files.

    Returns:
        str: Full path to the latest CSV file.
    """
    files = [f for f in os.listdir(data_directory) if f.startswith(file_prefix) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No preprocessed CSV files found. Please run Preprocess.py first.")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(data_directory, f)))
    return os.path.join(data_directory, latest_file)

# Fetch the latest CSV file
try:
    latest_csv = get_latest_csv(data_directory, file_prefix)
    df = pd.read_csv(latest_csv)

    st.write(f"Using latest preprocessed CSV: {latest_csv}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Ensure datetime columns are properly parsed
df['occurred_datetime'] = pd.to_datetime(df['occurred_datetime'], errors='coerce')

# Viz1: Yearly Trend of Crime
st.header("Yearly Crime Trend Visualization")
try:
    # Filter options
    offenses_viz1 = df['offence'].value_counts().head(15).index.tolist()
    neighborhoods_viz1 = df['neighbourhood_158'].dropna().unique().tolist()

    selected_offense_viz1 = st.selectbox("Select an offense type (Top 15)", ["All"] + offenses_viz1)
    selected_neighborhood_viz1 = st.selectbox("Select a neighborhood", ["All"] + neighborhoods_viz1)

    # Filter data
    filtered_df_viz1 = df.copy()
    if selected_offense_viz1 != "All":
        filtered_df_viz1 = filtered_df_viz1[filtered_df_viz1['offence'] == selected_offense_viz1]
    if selected_neighborhood_viz1 != "All":
        filtered_df_viz1 = filtered_df_viz1[filtered_df_viz1['neighbourhood_158'] == selected_neighborhood_viz1]

    # Group data by Year
    trend_data = filtered_df_viz1.groupby('occ_year').size().reset_index(name='count')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=trend_data, x='occ_year', y='count', marker="o", ax=ax)
    ax.set_title(f"Yearly Trend of '{selected_offense_viz1}' in '{selected_neighborhood_viz1}'")
    ax.set_xlabel("Year")
    ax.set_ylabel("Crime Count")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error generating Viz1: {e}")

# Viz2: Top 10 Neighborhoods
st.header("Top 10 Notorious Neighborhoods")

try:
    # Get the top 15 offenses
    offenses_viz2 = df['offence'].value_counts().head(15).index.tolist()
    selected_offense_viz2 = st.selectbox("Select an offense type (Top 15 offences)", ["All"] + offenses_viz2)

    # Copy dataset for independent filtering
    filtered_df_viz2 = df.copy()

    # Apply offense filter
    if selected_offense_viz2 != "All":
        filtered_df_viz2 = filtered_df_viz2[filtered_df_viz2['offence'] == selected_offense_viz2]

    # Check if data is available after filtering
    if filtered_df_viz2.empty:
        st.warning("No data available for the selected offense type.")
    else:
        # Aggregate data for top 10 neighborhoods
        neighborhood_data_viz2 = (
            filtered_df_viz2.groupby('neighbourhood_158')
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
            .head(10)
        )

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=neighborhood_data_viz2,
            x='count',
            y='neighbourhood_158',
            ax=ax,
            palette="viridis"
        )
        ax.set_title(f"Top 10 Most Notorious Neighborhoods for '{selected_offense_viz2}'")
        ax.set_xlabel("Crime Count")
        ax.set_ylabel("Neighborhood")
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error generating Viz2: {e}")

st.header("Column List")
try:
    df.columns
except Exception as e:
    st.error(f'Column list error: {e}')
