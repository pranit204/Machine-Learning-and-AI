import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# Title and Description
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of text as Positive or Negative using an ensemble model. Train a new model or use a pre-trained one for predictions.")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.write("Use this section to upload datasets and train the model.")

# Upload Training Data
train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type="csv")
test_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type="csv")

# Model and Vectorizer
model_file = "ensemble_model.pkl"
vectorizer_file = "tfidf_vectorizer.pkl"
model = None
tfidf = None

# Check if pre-trained model exists
try:
    tfidf = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    st.sidebar.write("Pre-trained model and vectorizer loaded successfully.")
except FileNotFoundError:
    st.sidebar.write("No pre-trained model found. Please train a model first.")

# Train a New Model
if st.sidebar.button("Train Model"):
    if train_file and test_file:
        # Load datasets
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Verify dataset structure
        if 'Sentence' not in train_data.columns or 'Polarity' not in train_data.columns:
            st.error("Training data must contain 'Sentence' and 'Polarity' columns.")
        elif 'Sentence' not in test_data.columns or 'Polarity' not in test_data.columns:
            st.error("Test data must contain 'Sentence' and 'Polarity' columns.")
        else:
            # Preprocessing: Normalize text
            def normalize_text(text):
                return text.lower().strip()

            train_data['Sentence'] = train_data['Sentence'].apply(normalize_text)
            test_data['Sentence'] = test_data['Sentence'].apply(normalize_text)

            X_train = train_data['Sentence']
            y_train = train_data['Polarity']
            X_test = test_data['Sentence']
            y_test = test_data['Polarity']

            # Vectorize text using TF-IDF
            tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            # Logistic Regression with Grid Search
            st.write("Tuning Logistic Regression...")
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            grid_search = GridSearchCV(LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'), param_grid, cv=5)
            grid_search.fit(X_train_tfidf, y_train)
            best_logistic = grid_search.best_estimator_

            # SVM with Probabilities Enabled
            st.write("Configuring SVM...")
            svm_model = SVC(probability=True, random_state=42, kernel='linear', class_weight='balanced')

            # Create an ensemble Voting Classifier with 'soft' voting
            st.write("Training ensemble model...")
            ensemble_model = VotingClassifier(estimators=[
                ('lr', best_logistic),
                ('svm', svm_model)
            ], voting='soft')

            # Train the ensemble model
            ensemble_model.fit(X_train_tfidf, y_train)

            # Evaluate the model
            y_pred = ensemble_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Ensemble Model Accuracy: {accuracy:.2f}")

            # Display Classification Report as a Table
            report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("### Classification Report")
            st.dataframe(report_df.style.format("{:.2f}"))

            # Confusion Matrix
            st.write("### Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig)

            # Class Distribution
            st.write("### Class Distribution in Test Data")
            class_counts = pd.Series(y_test).value_counts()
            fig, ax = plt.subplots()
            class_counts.plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
            st.pyplot(fig)

            # Confidence Distribution
            st.write("### Confidence Distribution")
            confidence_scores = ensemble_model.predict_proba(X_test_tfidf).max(axis=1)
            fig, ax = plt.subplots()
            sns.histplot(confidence_scores, bins=10, kde=True, color='green')
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            st.pyplot(fig)

            # Save the ensemble model and vectorizer
            joblib.dump(tfidf, vectorizer_file)
            joblib.dump(ensemble_model, model_file)
            st.success("Ensemble model and vectorizer saved successfully.")
    else:
        st.error("Please upload both training and test datasets to train the model.")

# Prediction Section
st.header("Sentiment Prediction")
user_input = st.text_area("Enter text for sentiment prediction", "")

if st.button("Predict Sentiment"):
    if tfidf is None or model is None:
        st.error("No pre-trained model found. Please train a model first.")
    elif user_input.strip():
        # Vectorize the input
        input_tfidf = tfidf.transform([user_input])

        # Predict sentiment and confidence
        prediction = model.predict(input_tfidf)[0]
        prediction_proba = model.predict_proba(input_tfidf)
        confidence = max(prediction_proba[0])

        # Display result
        st.write(f"Predicted Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.error("Please enter some text for prediction.")
