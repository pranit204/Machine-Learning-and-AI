import gzip
import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests

# Utility Functions
def load_model_and_trends(pkl_file_url):
    """
    Load the trained model, scaler, trends, and preprocessed data from a compressed pickle file.
    """
    try:
        # Download the compressed file
        response = requests.get(pkl_file_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Decompress the content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            data = pickle.load(f)

        return data["model"], data["scaler"], data.get("trends", {}), data.get("preprocessed_data", None)

    except Exception as e:
        raise FileNotFoundError(f"Error loading compressed model file: {e}")


# Define path to the compressed pickle file
compressed_pkl_file_path = "./lstm_model.pkl.gz"

# Streamlit App
def main():
    st.title("🚔 Toronto Crime Analysis and Forecasting Dashboard")

    # Load the model, trends, and preprocessed data
    try:
        model, scaler, trends, df = load_model_and_trends(compressed_pkl_file_path)
        if df is None:
            st.error("Preprocessed data missing in the pickle file. Ensure the model script saves preprocessed data.")
            return
    except FileNotFoundError:
        st.error("Compressed model file not found. Please ensure the model is trained, compressed, and saved properly.")
        return
    except KeyError:
        st.error("Trends data missing in the pickle file. Ensure the model script saves trends.")
        return

    # Ensure datetime columns are properly parsed
    df['occurred_datetime'] = pd.to_datetime(df['occurred_datetime'], errors='coerce')

    # Section 1: Yearly Crime Trend Visualization
    st.header("Yearly Crime Trend Visualization")
    offenses_viz1 = df['offence'].value_counts().head(15).index.tolist()
    neighborhoods_viz1 = df['neighbourhood_158'].dropna().unique().tolist()

    selected_offense_viz1 = st.selectbox("Select an Offense Type (Top 15)", ["All"] + offenses_viz1)
    selected_neighborhood_viz1 = st.selectbox("Select a Neighborhood", ["All"] + neighborhoods_viz1)

    filtered_df_viz1 = df.copy()
    if selected_offense_viz1 != "All":
        filtered_df_viz1 = filtered_df_viz1[filtered_df_viz1['offence'] == selected_offense_viz1]
    if selected_neighborhood_viz1 != "All":
        filtered_df_viz1 = filtered_df_viz1[filtered_df_viz1['neighbourhood_158'] == selected_neighborhood_viz1]

    trend_data = filtered_df_viz1.groupby('occ_year').size().reset_index(name='count')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=trend_data, x='occ_year', y='count', marker="o", ax=ax)
    ax.set_title(f"Yearly Trend of '{selected_offense_viz1}' in '{selected_neighborhood_viz1}'")
    ax.set_xlabel("Year")
    ax.set_ylabel("Crime Count")
    st.pyplot(fig)

    # Section 2: Top 10 Notorious Neighborhoods
    st.header("Top 10 Most Notorious Neighborhoods")
    selected_offense_viz2 = st.selectbox("Select an Offense Type (Top 15 Offences)", ["All"] + offenses_viz1)

    filtered_df_viz2 = df.copy()
    if selected_offense_viz2 != "All":
        filtered_df_viz2 = filtered_df_viz2[filtered_df_viz2['offence'] == selected_offense_viz2]

    if not filtered_df_viz2.empty:
        neighborhood_data_viz2 = (
            filtered_df_viz2.groupby('neighbourhood_158')
            .size()
            .reset_index(name='count')
            .sort_values(by='count', ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=neighborhood_data_viz2,
            x='count',
            y='neighbourhood_158',
            ax=ax,
            palette="viridis"
        )
        ax.set_title(f"Top 10 Most Notorious Neighborhoods for '{selected_offense_viz2}' Neighborhood'")
        ax.set_xlabel("Crime Count")
        ax.set_ylabel("Neighborhood")
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected offense type.")

    # Section 3: Predictive Insights
    st.header("📈Predictive Insights (next 30 days) using LSTM Neural Networks")
    try:
        top_increases = trends.get("increases", [])
        top_decreases = trends.get("decreases", [])

        st.subheader("Top 10 Neighborhoods with Predicted Crime Increases")
        if top_increases:
            increase_df = pd.DataFrame(top_increases, columns=["Neighborhood", "Percentage Increase"])
            st.dataframe(increase_df.style.format({"Percentage Increase": "{:.2f}%"}))
        else:
            st.info("No data available for neighborhoods with predicted increases.")

        st.subheader("Top 10 Neighborhoods with Predicted Crime Decreases")
        if top_decreases:
            decrease_df = pd.DataFrame(top_decreases, columns=["Neighborhood", "Percentage Decrease"])
            decrease_df = decrease_df.sort_values(by="Percentage Decrease", ascending=True)
            st.dataframe(decrease_df.style.format({"Percentage Decrease": "{:.2f}%"}))
        else:
            st.info("No data available for neighborhoods with predicted decreases.")
    except Exception as e:
        st.error(f"Error generating predictive insights: {e}")

if __name__ == "__main__":
    main()
