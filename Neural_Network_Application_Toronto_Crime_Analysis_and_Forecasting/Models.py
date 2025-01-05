import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle


# Utility Functions
def load_preprocessed_data(directory, file_prefix):
    """
    Load the most recent preprocessed CSV file.
    """
    files = [f for f in os.listdir(directory) if f.startswith(file_prefix) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No preprocessed data file found.")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
    return pd.read_csv(os.path.join(directory, latest_file))


def create_targets(df):
    """
    Create a 'crime_count' column dynamically if missing.
    """
    if 'crime_count' not in df.columns:
        print("'crime_count' column not found. Calculating dynamically...")
        df['occurred_datetime'] = pd.to_datetime(
            dict(
                year=df['occ_year'],
                month=df['occ_month'],
                day=df['occ_day'],
                hour=df['occ_hour'].fillna(0).astype(int)
            ),
            errors='coerce'
        )
        crime_counts = df.groupby(['occurred_datetime', 'neighbourhood_158']).size().reset_index(name='crime_count')
        df = pd.merge(df, crime_counts, on=['occurred_datetime', 'neighbourhood_158'], how='left')
    return df


def prepare_data_for_lstm(df, sequence_length=30):
    """
    Prepare LSTM input features and labels for the entire dataset's sequence length.
    """
    if 'crime_count' not in df.columns:
        raise KeyError("The 'crime_count' column is missing from the dataset. Check the preprocessing step.")

    # Scale the 'crime_count' column
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['crime_count']])

    if len(scaled_data) < sequence_length:
        raise ValueError(f"Not enough data for training. Dataset size: {len(scaled_data)} rows.")

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Model Functions
def build_lstm_model(input_shape):
    """
    Build the LSTM model architecture.
    """
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_model(model, X, y, epochs=3, batch_size=16, validation_split=0.2):
    """
    Train the LSTM model with input features and labels.
    """
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    return history


def save_model_and_trends(model, scaler, trends, output_file="lstm_model.pkl"):
    """
    Save the trained model, scaler, and computed trends to a pickle file.
    """
    with open(output_file, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "trends": trends}, f)
    print(f"Model, scaler, and trends saved to {output_file}")


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model and calculate RMSE.
    """
    y_pred = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(y_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print(f"Overall RMSE: {rmse}")
    return rmse


def calculate_forecast_changes(df, model, scaler, sequence_length=30, future_steps=30):
    """
    Calculate percentage changes in crime for all neighborhoods.
    """
    neighborhoods = df["neighbourhood_158"].unique()
    results = []

    for neighborhood in neighborhoods:
        neighborhood_df = df[df["neighbourhood_158"] == neighborhood]

        if len(neighborhood_df) < sequence_length:
            continue

        neighborhood_df = neighborhood_df.sort_values("occurred_datetime")
        historical_values = neighborhood_df["crime_count"].values

        # Prepare input sequence for forecasting
        scaler.fit(historical_values.reshape(-1, 1))
        scaled_data = scaler.transform(historical_values.reshape(-1, 1))

        X = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i])

        if len(X) == 0:
            continue

        X = np.array(X)
        last_sequence = X[-1:]

        # Forecast future crime counts
        forecasts = []
        for _ in range(future_steps):
            prediction = model.predict(last_sequence, verbose=0)
            forecasts.append(prediction[0])
            last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, -1), axis=1)

        forecasts_rescaled = scaler.inverse_transform(np.array(forecasts))
        historical_avg = np.mean(historical_values[-sequence_length:])
        forecast_avg = np.mean(forecasts_rescaled)

        pct_change = ((forecast_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
        results.append((neighborhood, pct_change))

    # Sort neighborhoods by increase and decrease
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    top_increases = results_sorted[:10]
    top_decreases = results_sorted[-10:]

    return {"increases": top_increases, "decreases": top_decreases}


# Main Script
if __name__ == "__main__":
    data_directory = "../Data"
    file_prefix = "preprocessed_data_"

    # Load and preprocess data
    print("Loading data...")
    df = load_preprocessed_data(data_directory, file_prefix)
    df = create_targets(df)

    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    try:
        X, y, scaler = prepare_data_for_lstm(df)
    except ValueError as e:
        print(f"Error during data preparation: {e}")
        exit()

    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the model
    print("Building and training the model...")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    train_lstm_model(model, X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    rmse = evaluate_model(model, X_test, y_test, scaler)

    # Calculate trends
    print("Calculating forecast trends...")
    future_steps = 180
    trends = calculate_forecast_changes(df, model, scaler, future_steps=future_steps)

    # Save the model, scaler, and trends
    print("Saving the model, scaler, and trends...")
    save_model_and_trends(model, scaler, trends)

    print(f"Model training complete. RMSE: {rmse}")
    print("Top 10 neighborhoods with increases and decreases saved.")
