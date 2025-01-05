**Crime Forecasting Dashboard with LSTM Neural Networks**


🚓 Project Overview:


The Crime Forecasting Dashboard is a powerful analytics and prediction platform built to analyze crime trends across neighborhoods, forecast future crime rates using LSTM Neural Networks, and provide actionable insights to law enforcement agencies, urban planners, and policymakers. By leveraging historical crime data, the dashboard empowers users to identify areas of concern and plan interventions to make neighborhoods safer.
📊 Key Features
Exploratory Data Analysis (EDA):
Visualizations of yearly crime trends by offense types and neighborhoods.
Identification of the top 10 neighborhoods with the highest crime rates.
Crime Prediction with LSTM Neural Networks:
Utilizes Long Short-Term Memory (LSTM) models for time series forecasting.
Predicts future crime trends in each neighborhood based on historical data.
Calculates percentage changes in crime for each neighborhood over a given period.
Actionable Insights:
Lists the top 10 neighborhoods with the highest projected crime increases.
Highlights the top 10 neighborhoods with the highest projected crime decreases.
Offers dynamic crime forecasts for the next 180 days.
Interactive Dashboard:
Built with Streamlit for user-friendly visualization and interactivity.
Enables users to explore data, analyze trends, and visualize forecasts with minimal effort.
📂 Project Structure
1. Data Pipeline
Preprocessing: Data from historical crime reports is cleaned, formatted, and enhanced with features like datetime fields, cyclical transformations, and crime counts per neighborhood.
Dynamic Crime Count: Automatically calculates crime counts for neighborhoods if missing.
2. LSTM Model Training
Sequence Preparation: Historical data is converted into sequences for training the LSTM model.
Model Architecture:
Two LSTM layers with ReLU activation.
Dropout layers for regularization.
Dense output layer for single-step predictions.
Evaluation:
Model is evaluated using Root Mean Square Error (RMSE) for accuracy.
Key outputs include the trained model, scaler, and insights on trends (top increases and decreases).
3. Predictive Insights
Top 10 Neighborhoods with Crime Increases:
Neighborhoods projected to have the highest percentage increase in crime rates.
Top 10 Neighborhoods with Crime Decreases:
Neighborhoods projected to see the highest percentage decrease in crime rates.
4. Interactive Dashboard
Built with Streamlit to showcase:
EDA Visualizations:
Yearly crime trends.
Top 10 neighborhoods with high crime counts.
Predictive Insights:
Visualizations of predicted crime trends.
Top 10 neighborhoods with increases and decreases.
Dynamic Forecasting: Forecast crime trends for selected neighborhoods for up to 180 days.
📈 Visualizations
EDA Visualizations
Yearly Crime Trends:
Interactive line charts showing crime trends by year, offense, and neighborhood.
Top 10 Notorious Neighborhoods:
Bar chart ranking neighborhoods by crime count for specific offenses.
Predictive Visualizations
Crime Forecasting with LSTM Neural Nets:
Line charts showcasing historical crime data and future forecasts.
Insights on percentage changes in crime for each neighborhood.
Neighborhood Trends:
Tables listing top 10 neighborhoods with the highest projected increases and decreases.
🛠️ Technologies Used
Python:
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras, streamlit.
Machine Learning:
LSTM Neural Networks for time series forecasting.
MinMaxScaler for normalizing data.
Visualization:
Dynamic plots with Seaborn and Matplotlib.
Interactive dashboards with Streamlit.
📜 Key Results
Model Performance:
Achieved an overall RMSE of 89.27 on test data.
Top 10 Neighborhoods with Predicted Increases:
Example: Neighborhood X (15.2%), Neighborhood Y (13.6%), etc.
Top 10 Neighborhoods with Predicted Decreases:
Example: Neighborhood Z (-14.8%), Neighborhood W (-12.3%), etc.
