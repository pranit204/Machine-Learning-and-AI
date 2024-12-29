***Sentiment Analysis App***

This interactive Streamlit app predicts the sentiment of input text as Positive or Negative using a pre-trained ensemble model. It also provides evaluation metrics and visualizations to showcase model performance.

**Cloud Setup:**

The app is deployed and live on Streamlit Cloud. Access the live demo here: https://pranit204-machi-interactive-nlp-streamlit-applicationapp-liu0ja.streamlit.app/

**Key Features:**

Sentiment Prediction:
Input text is classified as Positive or Negative.
Provides a confidence score for each prediction.

**Pre-Trained Model:**

The model has been pre-trained for user convenience.
No need to upload datasets or retrain the model.

**Evaluation Metrics:**

Model Accuracy: 78%
Includes:
Classification Report
Confusion Matrix
Confidence Score Distribution

**Ensemble Model:**

Combines Logistic Regression and SVM using a soft voting mechanism.

**File Structure**

Interactive NLP Streamlit Application

├── app.py                     # Streamlit app script

├── ensemble_model.pkl         # Pre-trained ensemble model

├── tfidf_vectorizer.pkl       # Pre-trained TF-IDF vectorizer

├── evaluation_metrics.pkl     # Pre-saved evaluation metrics

├── README.md                  # Project documentation

├── requirements.txt           # Python dependencies


**Contact**

Created by [Pranit Sanghavi](https://github.com/pranit204)

Reach out via [LinkedIn](https://www.linkedin.com/in/pranit-sanghavi) or [Email](mailto:pranit.careers@gmail.com).
