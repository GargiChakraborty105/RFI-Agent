import pandas as pd
import numpy as np
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import string
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample RFI Data with List of Questions
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
            "What steps can be taken to expedite delivery?",
            "Is an alternative material available?"
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

# Sample User Data
user_data = [
    {"id": 101, "name": "John Doe", "email": "john@example.com", "job_title": "Project Manager", "company": "ABC Constructions"},
    {"id": 102, "name": "Alice Smith", "email": "alice@example.com", "job_title": "Electrical Engineer", "company": "XYZ Electrics"}
]

# Convert data to DataFrames
rfi_df = pd.DataFrame(rfi_data)
user_df = pd.DataFrame(user_data)

def preprocess_text_list(text_list):
    """Clean and preprocess a list of text questions."""
    cleaned_list = []
    for text in text_list:
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        cleaned_list.append(" ".join(words))
    return cleaned_list

# Apply text preprocessing
rfi_df['cleaned_text'] = rfi_df['questions_body'].apply(preprocess_text_list)

# Load NER Model for Keyword Extraction
keyword_extractor = pipeline("ner", aggregation_strategy="simple")

def extract_keywords_from_list(text_list):
    """Extracts keywords from a list of questions."""
    keywords = []
    for text in text_list:
        extracted_keywords = keyword_extractor(text)
        keywords.extend([kw['word'] for kw in extracted_keywords])
    return list(set(keywords))  # Removing duplicates

# Apply keyword extraction
rfi_df['keywords'] = rfi_df['cleaned_text'].apply(extract_keywords_from_list)

from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def calculate_sentiment_score(text_list):
    """Calculates sentiment score for a list of questions."""
    sentiment_scores = []
    for text in text_list:
        sentiment = sentiment_analyzer(text)[0]
        sentiment_scores.append(1 if sentiment['label'] == 'POSITIVE' else 0)  # 1 for positive, 0 for negative
    return np.mean(sentiment_scores)  # Average sentiment score for the questions in the RFI

# Apply sentiment score calculation
rfi_df['sentiment_score'] = rfi_df['questions_body'].apply(calculate_sentiment_score)

def calculate_urgency(priority_value, priority_name, due_date):
    """Calculates urgency score based on priority and due date."""
    days_remaining = (due_date - datetime.now()).days
    priority_score = 1 / priority_value
    if priority_name:
        priority_score += 0.5
    due_date_score = 1 if days_remaining <= 3 else (0.6 if days_remaining <= 7 else 0.3)
    return round((priority_score + due_date_score) / 2, 2)

# Apply Urgency Scoring
rfi_df['urgency_score'] = rfi_df.apply(
    lambda row: calculate_urgency(row['priority_value'], row['priority_name'], row['due_date']), axis=1
)

# Generate Resolution Time for Predictive Analysis
rfi_df['resolution_time'] = (rfi_df['updated_at'] - rfi_df['created_at']).dt.days

# Prepare Data for Training
X = rfi_df[['priority_value', 'urgency_score', 'sentiment_score']]
y = rfi_df['resolution_time']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the Model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

def calculate_suitability(user_id, rfi_df):
    """Calculates suitability score for assignees."""
    current_workload = len(rfi_df[rfi_df['assignees_id'] == user_id])
    historical_performance = 1 / (current_workload + 1)
    return round(historical_performance, 2)

# Calculate suitability scores for each user
user_df['suitability_score'] = user_df['id'].apply(lambda x: calculate_suitability(x, rfi_df))

# Merge RFI and User Data on Assignees
final_df = pd.merge(rfi_df, user_df, left_on="assignees_id", right_on="id", suffixes=('_rfi', '_user'))

# Display Final Results
print("\nFinal RFI Data with User Information:")
print(final_df[['subject', 'assignees_name', 'sentiment_score', 'urgency_score', 'suitability_score']])