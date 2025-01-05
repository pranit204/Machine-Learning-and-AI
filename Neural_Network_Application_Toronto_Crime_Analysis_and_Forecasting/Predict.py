import pickle
import streamlit as st
import pandas as pd


# Utility Functions
def load_model_and_trends(pkl_file):
    """
    Load the trained model, scaler, and trends from the pickle file.
    """
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["trends"]


# Streamlit App
def main():
    st.title("Crime Forecasting Dashboard")

    # File paths
    pkl_file_path = "/Toronto_Police_Service_Major_Crime_Indicators/ML/lstm_model.pkl"

    # Load model, scaler, and trends
    try:
        model, scaler, trends = load_model_and_trends(pkl_file_path)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved properly.")
        return
    except KeyError:
        st.error("Trends data missing in the pickle file. Ensure the model script saves trends.")
        return

    # Extract trends
    top_increases = trends.get("increases", [])
    top_decreases = trends.get("decreases", [])

    # Display trends
    st.header("Top 10 Neighborhoods with Predicted Crime Increases")
    if top_increases:
        increase_df = pd.DataFrame(top_increases, columns=["Neighborhood", "Percentage Increase"])
        st.dataframe(increase_df.style.format({"Percentage Increase": "{:.2f}%"}))
    else:
        st.info("No data available for neighborhoods with predicted increases.")

    st.header("Top 10 Neighborhoods with Predicted Crime Decreases")
    if top_decreases:
        decrease_df = pd.DataFrame(top_decreases, columns=["Neighborhood", "Percentage Decrease"])
        st.dataframe(decrease_df.style.format({"Percentage Decrease": "{:.2f}%"}))
    else:
        st.info("No data available for neighborhoods with predicted decreases.")

    st.success("Data loaded successfully. Visualization complete.")


if __name__ == "__main__":
    main()
