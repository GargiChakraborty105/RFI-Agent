import pandas as pd
import numpy as np
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import string
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download stopwords for text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ✅ Sample RFI Data
rfi_data = [
    {
        "id": 1,
        "subject": "Material Delay Issue",
        "priority_value": 1,
        "priority_name": True,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=5),
        "updated_at": datetime.now() - timedelta(days=2),
        "due_date": datetime.now() + timedelta(days=3),
        "questions_body": [
            "Why is the material delayed?",
            "What steps can be taken to expedite delivery?"
        ],
        "assignees_name": "John Doe",
        "assignees_id": 101,
        "project_id": 501
    },
    {
        "id": 2,
        "subject": "Electrical Wiring Clarification",
        "priority_value": 2,
        "priority_name": False,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=10),
        "updated_at": datetime.now() - timedelta(days=4),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": [
            "What is the standard wiring specification for Zone A?",
            "Can the wiring be replaced with a more efficient model?"
        ],
        "assignees_name": "Alice Smith",
        "assignees_id": 102,
        "project_id": 502
    }
]

# ✅ Sample User Data
user_data = [
    {"id": 101, "name": "John Doe", "email": "john@example.com", "job_title": "Project Manager", "company": "ABC Constructions"},
    {"id": 102, "name": "Alice Smith", "email": "alice@example.com", "job_title": "Electrical Engineer", "company": "XYZ Electrics"}
]


# ✅ Analysis Class Definition
class Analysis:
    def __init__(self, rfi_data, user_data):
        self.rfi_df = pd.DataFrame(rfi_data)
        self.user_df = pd.DataFrame(user_data)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.keyword_extractor = pipeline("ner", aggregation_strategy="simple")

    # ✅ Text Preprocessing
    def preprocess_text_list(self, text_list):
        """Clean and preprocess a list of questions."""
        cleaned_list = []
        for text in text_list:
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            words = text.split()
            words = [word for word in words if word not in stop_words]
            cleaned_list.append(" ".join(words))
        return cleaned_list

    # ✅ Keyword Extraction
    def extract_keywords(self):
        """Extract keywords using HuggingFace NER."""
        self.rfi_df['cleaned_questions'] = self.rfi_df['questions_body'].apply(self.preprocess_text_list)
        self.rfi_df['keywords'] = self.rfi_df['cleaned_questions'].apply(
            lambda questions: list(set([kw['word'] for q in questions for kw in self.keyword_extractor(q)]))
        )

    # ✅ Sentiment Analysis
    def calculate_sentiment_score(self):
        """Calculates sentiment score using HuggingFace."""
        def sentiment_score(text_list):
            scores = [1 if self.sentiment_analyzer(q)[0]['label'] == 'POSITIVE' else 0 for q in text_list]
            return round(np.mean(scores), 2)

        self.rfi_df['sentiment_score'] = self.rfi_df['questions_body'].apply(sentiment_score)

    # ✅ Urgency Scoring
    def calculate_urgency_score(self):
        """Calculates urgency score based on priority and due date."""
        def urgency_score(row):
            days_remaining = (row['due_date'] - datetime.now()).days
            priority_score = 1 / row['priority_value']
            if row['priority_name']:
                priority_score += 0.5
            due_date_score = 1 if days_remaining <= 3 else (0.6 if days_remaining <= 7 else 0.3)
            return round((priority_score + due_date_score) / 2, 2)

        self.rfi_df['urgency_score'] = self.rfi_df.apply(urgency_score, axis=1)

    # ✅ Train Predictive Model for Resolution Time
    def train_resolution_time_predictor(self):
        """Train a Random Forest model to predict resolution time."""
        # Generate historical resolution time
        self.rfi_df['resolution_time'] = (self.rfi_df['updated_at'] - self.rfi_df['created_at']).dt.days

        # Prepare dataset for training
        X = self.rfi_df[['priority_value', 'urgency_score', 'sentiment_score']]
        y = self.rfi_df['resolution_time']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict resolution time
        self.rfi_df['predicted_resolution_time'] = rf_model.predict(X)
        mse = mean_squared_error(y_test, rf_model.predict(X_test))
        print(f"\nMean Squared Error: {mse:.2f}")

    # ✅ Generate Final Analytics Data
    def generate_analytics_data(self):
        """Generate final analytics data combining RFI and User data."""
        analytics_data = self.rfi_df[['id', 'sentiment_score', 'urgency_score', 'predicted_resolution_time']].to_dict(orient='records')
        print("\nFinal Analytics Data:")
        for record in analytics_data:
            print(record)
        return analytics_data

    # ✅ Run Complete Analysis Pipeline
    def run_analysis(self):
        """Run the entire analysis pipeline."""
        self.extract_keywords()
        self.calculate_sentiment_score()
        self.calculate_urgency_score()
        self.train_resolution_time_predictor()
        return self.generate_analytics_data()


# ✅ Running the Analysis Class
analysis = Analysis(rfi_data, user_data)
analytics_data = analysis.run_analysis()
