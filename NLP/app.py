import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import random

# Title
st.title("Sentiment Analysis with BERT")

# Ensure reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

# Device setup
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Computation running on: {device_type}")

# Load Pretrained BERT Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Binary classification
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()
model.to(device_type)

# Define Custom Dataset
class ReviewDataset(Dataset):
    def __init__(self, tokenized_data, target_labels):
        self.data = tokenized_data
        self.labels = torch.tensor(target_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        inputs = {key: val[index] for key, val in self.data.items()}
        inputs['labels'] = self.labels[index]
        return inputs

# Step 1: Upload Data
uploaded_train_file = st.file_uploader("Upload Training Dataset (CSV)", type="csv")
uploaded_test_file = st.file_uploader("Upload Test Dataset (CSV)", type="csv")

if uploaded_train_file and uploaded_test_file:
    # Load datasets
    train_data = pd.read_csv(uploaded_train_file)
    test_data = pd.read_csv(uploaded_test_file)

    st.write("Training Dataset Preview:")
    st.dataframe(train_data.head())
    st.write("Test Dataset Preview:")
    st.dataframe(test_data.head())

    # Prepare data
    train_texts = train_data["Sentence"].astype(str).tolist()
    train_labels = train_data["Polarity"].tolist()
    test_texts = test_data["Sentence"].astype(str).tolist()
    test_labels = test_data["Polarity"].tolist()

    # Tokenize
    max_seq_length = 50
    train_encoded = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt"
    )
    test_encoded = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt"
    )

    # DataLoader
    train_dataset = ReviewDataset(train_encoded, train_labels)
    test_dataset = ReviewDataset(test_encoded, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Step 2: Train Model
    optimizer = AdamW(model.parameters(), lr=2e-5)

    def train_epoch(model, loader, optimizer):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device_type)
            attention_mask = batch['attention_mask'].to(device_type)
            labels = batch['labels'].to(device_type)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        return total_loss / len(loader)

    if st.button("Train BERT Model"):
        epochs = 3
        for epoch in range(epochs):
            avg_loss = train_epoch(model, train_loader, optimizer)
            st.write(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # Step 3: Evaluate Model
    def evaluate_model(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device_type)
                attention_mask = batch['attention_mask'].to(device_type)
                labels = batch['labels'].to(device_type)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())  # Ensure numpy array
                all_labels.extend(labels.cpu().numpy())  # Ensure numpy array
        return np.array(all_preds), np.array(all_labels)


    preds, labels = None, None
    if st.button("Evaluate Model"):
        preds, labels = evaluate_model(model, test_loader)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        st.write(f"Test Accuracy: {acc:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(labels, preds))

        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        st.write("Confusion Matrix:")
        st.write(cm)

    # Step 4: Visualizations
    if preds is not None:
        # Convert to numpy array (if not already)
        preds = np.array(preds)

        # Positive and negative comments
        positive_comments = " ".join(test_data.iloc[np.atleast_1d(preds == 1).nonzero()[0]]['Sentence'])
        negative_comments = " ".join(test_data.iloc[np.atleast_1d(preds == 0).nonzero()[0]]['Sentence'])

        st.subheader("Word Cloud: Positive Comments")
        wordcloud_positive = WordCloud(background_color='white').generate(positive_comments)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.subheader("Word Cloud: Negative Comments")
        wordcloud_negative = WordCloud(background_color='white').generate(negative_comments)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("Please evaluate the model first to generate visualizations.")

    # Step 5: Predict Sentiment
    user_input = st.text_area("Enter text to classify sentiment:")
    if st.button("Predict Sentiment"):
        model.eval()
        encoding = tokenizer(
            user_input,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].to(device_type)
        attention_mask = encoding['attention_mask'].to(device_type)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
