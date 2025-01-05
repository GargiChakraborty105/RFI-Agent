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

# Suppress warnings and download stopwords
warnings.filterwarnings('ignore')
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
        "questions_body": ["Why is the material delayed?", "Urgent action required."],
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
        "questions_body": ["Clarify wiring specification.", "Is this issue critical?"],
        "assignees_name": "Alice Smith",
        "assignees_id": 102,
        "project_id": 502
    }
]

# ✅ Sample User Data
user_data = [
    {"user_id": 101, "name": "John Doe", "email": "john@example.com", 
     "job_title": "Project Manager", "company": "ABC Constructions", 
     "current_workload": 3, "historical_performance_score": 0.85},
    
    {"user_id": 102, "name": "Alice Smith", "email": "alice@example.com", 
     "job_title": "Electrical Engineer", "company": "XYZ Electrics", 
     "current_workload": 2, "historical_performance_score": 0.92}
]


# ✅ Analysis Class Definition
class Analysis:
    def __init__(self, rfi_data, user_data):
        self.rfi_df = pd.DataFrame(rfi_data)
        self.user_df = pd.DataFrame(user_data)
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    # ✅ Text Preprocessing
    def preprocess_text_list(self, text_list):
        """Cleans and preprocesses text."""
        cleaned_list = []
        for text in text_list:
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            words = text.split()
            words = [word for word in words if word not in stop_words]
            cleaned_list.append(" ".join(words))
        return cleaned_list

    # ✅ Sentiment Analysis for Urgency Classification
    def calculate_sentiment_score(self, text_list):
        """Classifies urgency based on positive/negative sentiment."""
        sentiment_scores = []
        for text in text_list:
            result = self.sentiment_analyzer(text)[0]
            score = 1 if result['label'] == 'POSITIVE' else -1
            sentiment_scores.append(score)
        avg_score = np.mean(sentiment_scores)
        return round(avg_score, 2)

    # ✅ Keyword Detection for Urgency Adjustment
    def detect_critical_keywords(self, text_list):
        """Detect critical terms like 'urgent' or 'blocked'."""
        keywords = ["urgent", "blocked", "critical", "delay", "immediate"]
        critical_count = sum(1 for text in text_list for word in keywords if word in text.lower())
        return critical_count

    # ✅ Urgency Scoring Using Sentiment and Keywords
    def calculate_urgency_score(self, row):
        """Combines priority, due date, sentiment, and keywords."""
        days_remaining = (row['due_date'] - datetime.now()).days
        priority_score = 1 / row['priority_value']
        if row['priority_name']:
            priority_score += 0.5
        due_date_score = 1 if days_remaining <= 3 else (0.6 if days_remaining <= 7 else 0.3)

        # Sentiment and Keyword Adjustment
        sentiment_score = self.calculate_sentiment_score(row['questions_body'])
        keyword_count = self.detect_critical_keywords(row['questions_body'])
        urgency_adjustment = 0.2 * keyword_count + (0.3 if sentiment_score < 0 else 0)

        # Final Urgency Score Calculation
        final_score = round((priority_score + due_date_score + urgency_adjustment) / 2, 2)
        return final_score

    # ✅ Prepare Data for Model Training
    def prepare_data(self):
        """Apply preprocessing and prepare the data."""
        self.rfi_df['cleaned_questions'] = self.rfi_df['questions_body'].apply(self.preprocess_text_list)
        self.rfi_df['urgency_score'] = self.rfi_df.apply(self.calculate_urgency_score, axis=1)
        self.rfi_df['sentiment_score'] = self.rfi_df['questions_body'].apply(self.calculate_sentiment_score)
        self.rfi_df['resolution_time'] = (self.rfi_df['updated_at'] - self.rfi_df['created_at']).dt.days

    # ✅ Train Predictive Model for Resolution Time
    def train_resolution_time_predictor(self):
        """Train a model to predict resolution time."""
        # Prepare Data
        X = self.rfi_df[['priority_value', 'urgency_score', 'sentiment_score']]
        y = self.rfi_df['resolution_time']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict and Evaluate
        self.rfi_df['predicted_resolution_time'] = rf_model.predict(X)
        mse = mean_squared_error(y_test, rf_model.predict(X_test))
        print(f"\nMean Squared Error: {mse:.2f}")

    # ✅ Generate Analytics Data
    def generate_analytics_data(self):
        """Generate final analytics data with sentiment, urgency, and predicted resolution time."""
        analytics_data = self.rfi_df[['id', 'sentiment_score', 'urgency_score', 'predicted_resolution_time']].to_dict(orient='records')
        print("\nFinal Analytics Data:")
        for record in analytics_data:
            print(record)
        return analytics_data

    # ✅ Run Full Pipeline
    def run_analysis(self):
        """Run the entire analysis pipeline."""
        self.prepare_data()
        self.train_resolution_time_predictor()
        return self.generate_analytics_data()


# ✅ Running the Analysis
analysis = Analysis(rfi_data, user_data)
analytics_data = analysis.run_analysis()