import streamlit as st
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification

# Configuration
BERT_CHECKPOINT = "bert-base-uncased"
GO_EMOTIONS_CHECKPOINT = "joeddav/distilbert-base-uncased-go-emotions-student"
MAX_LEN = 128
HIDDEN_SIZE = 256
LABEL_MAPPING = {0: "Negative", 1: "Positive"}

# Define the LSTM-BERT model
class LSTMBERT(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_classes):
        super(LSTMBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        output = self.fc(lstm_output[:, -1, :])
        return output

# Define Emotion Analyzer
class EmotionAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GO_EMOTIONS_CHECKPOINT)
        self.model = AutoModelForSequenceClassification.from_pretrained(GO_EMOTIONS_CHECKPOINT)
        self.labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
            "remorse", "sadness", "surprise", "neutral"
        ]

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion_scores = {self.labels[i]: probs[0][i].item() for i in range(len(self.labels))}
        sorted_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))
        return sorted_emotions

# Load models
@st.cache_resource
def load_models():
    sentiment_model = LSTMBERT(BERT_CHECKPOINT, HIDDEN_SIZE, 2)
    sentiment_model.load_state_dict(torch.load("lstmb_bert_model_imdb.pth"))
    sentiment_model.eval()
    emotion_analyzer = EmotionAnalyzer()
    tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)
    return sentiment_model, tokenizer, emotion_analyzer

sentiment_model, sentiment_tokenizer, emotion_analyzer = load_models()

# Streamlit App
st.title("Movie Review Sentiment and Emotion Analysis")

user_input = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        # Sentiment Analysis
        inputs = sentiment_tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        outputs = sentiment_model(inputs['input_ids'], inputs['attention_mask'])
        sentiment_prediction = torch.argmax(outputs, dim=1).item()
        sentiment = LABEL_MAPPING[sentiment_prediction]

        # Emotion Analysis
        emotions = emotion_analyzer.analyze(user_input)

        # Display Sentiment Results
        st.subheader("Sentiment Analysis")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** {torch.nn.functional.softmax(outputs, dim=1).max().item() * 100:.2f}%")

        # Display Emotion Results
        st.subheader("Emotion Detection")
        #st.write("Top 5 emotions detected:")
        for emotion, score in list(emotions.items())[:5]:
            st.write(f"{emotion.capitalize()}: {score * 100:.2f}%")
    else:
        st.error("Please enter a valid review.")
