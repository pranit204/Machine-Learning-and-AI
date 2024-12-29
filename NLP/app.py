import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Title and Description
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of text as Positive or Negative using a pre-trained ensemble model.")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.write("Use this section to check the status of pre-trained models. For your convenience, the model has already been pre-trained, so there’s no need to re-upload the datasets.")

# Enhanced Sidebar Information
st.sidebar.header("Application Overview")
st.sidebar.write("""
This application predicts sentiment as Positive or Negative using a pre-trained ensemble model.

**Key Features:**

- Model Accuracy: 78%
- Ensemble Approach: Logistic Regression + SVM with soft voting
- Confidence-based predictions
""")

# Debugging Information
st.sidebar.write("---")
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"TF-IDF Vectorizer Found: {'True' if os.path.exists('tfidf_vectorizer.pkl') else 'False'}")
st.sidebar.write(f"Ensemble Model Found: {'True' if os.path.exists('ensemble_model.pkl') else 'False'}")
st.sidebar.write(f"Evaluation Metrics Found: {'True' if os.path.exists('evaluation_metrics.pkl') else 'False'}")

# Signature in the sidebar
st.sidebar.write("---")
st.sidebar.write("**Created by [Pranit Sanghavi](https://github.com/pranit204)**")

# Load Pre-trained Model and Metrics
model_file = "ensemble_model.pkl"
vectorizer_file = "tfidf_vectorizer.pkl"
evaluation_metrics_file = "evaluation_metrics.pkl"

import os

if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    st.sidebar.write("All required files are found.")
    model = joblib.load(model_file)
    tfidf = joblib.load(vectorizer_file)
else:
    st.sidebar.error("Required files are missing. Please check deployment.")

model = joblib.load(model_file)
tfidf = joblib.load(vectorizer_file)
metrics = joblib.load(evaluation_metrics_file)

# Display Evaluation Metrics
st.header("Evaluation Metrics")
st.write(f"**Model Accuracy:** {metrics['accuracy']:.2f}")

# Classification Report
st.write("### Classification Report")
classification_report_df = pd.DataFrame(metrics['classification_report']).transpose()
st.dataframe(classification_report_df.style.format("{:.2f}"))

# Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(fig)

# Confidence Distribution
st.write("### Confidence Distribution")
fig, ax = plt.subplots()
sns.histplot(metrics['confidence_scores'], bins=10, kde=True, color='green')
plt.title('Confidence Score Distribution')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
st.pyplot(fig)

# Sentiment Prediction
st.header("Sentiment Prediction")
user_input = st.text_area("Enter text for sentiment prediction", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Vectorize the input
        input_tfidf = tfidf.transform([user_input])

        # Predict sentiment and confidence
        prediction = model.predict(input_tfidf)[0]
        prediction_proba = model.predict_proba(input_tfidf)
        confidence = max(prediction_proba[0])

        # Display result
        st.write(f"**Predicted Sentiment:** {'Positive' if prediction == 1 else 'Negative'}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.error("Please enter some text for prediction.")

# Footer signature
st.markdown("---")
st.markdown("**Created by [Pranit Sanghavi](https://github.com/pranit204)**")
st.markdown("Reach out via [LinkedIn](https://www.linkedin.com/in/pranit-sanghavi) or [Email](mailto:pranit.careers@gmail.com).")
