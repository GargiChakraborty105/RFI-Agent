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

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample RFI Data
rfi_data = [
    {
        "rfi_id": 1, 
        "subject": "Material Delay Issue", 
        "status": "Open", 
        "priority_name": True, 
        "priority_value": 1, 
        "created_at": datetime.now() - timedelta(days=5),
        "updated_at": datetime.now() - timedelta(days=2),
        "due_date": datetime.now() + timedelta(days=3),
        "questions_body": ["Why is the material delayed?", "Is an alternative material available?"],
        "assignee_id": 101
    },
    {
        "rfi_id": 2, 
        "subject": "Electrical Wiring Clarification", 
        "status": "In Progress", 
        "priority_name": False, 
        "priority_value": 2, 
        "created_at": datetime.now() - timedelta(days=10),
        "updated_at": datetime.now() - timedelta(days=4),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": ["Clarify the wiring specification for Zone A.", "Can we use alternative wiring?"],
        "assignee_id": 102
    }
]

# Sample User Data
user_data = [
    {"user_id": 101, "name": "John Doe", "email": "john@example.com", "job_title": "Project Manager",
     "company": "ABC Constructions", "current_workload": 3, "historical_performance_score": 0.85},
    
    {"user_id": 102, "name": "Alice Smith", "email": "alice@example.com", "job_title": "Electrical Engineer",
     "company": "XYZ Electrics", "current_workload": 2, "historical_performance_score": 0.92}
]

# Convert to DataFrames
rfi_df = pd.DataFrame(rfi_data)
user_df = pd.DataFrame(user_data)

def preprocess_text_list(text_list):
    """Cleans and preprocesses text."""
    cleaned_list = []
    for text in text_list:
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        cleaned_list.append(" ".join(words))
    return cleaned_list

def calculate_urgency(priority_value, priority_name, due_date):
    """Calculates urgency score based on priority and due date."""
    days_remaining = (due_date - datetime.now()).days
    priority_score = 1 / priority_value
    if priority_name:
        priority_score += 0.5
    due_date_score = 1 if days_remaining <= 3 else (0.6 if days_remaining <= 7 else 0.3)
    return round((priority_score + due_date_score) / 2, 2)

# Apply preprocessing and urgency scoring
rfi_df['cleaned_questions'] = rfi_df['questions_body'].apply(preprocess_text_list)
rfi_df['urgency_score'] = rfi_df.apply(
    lambda row: calculate_urgency(row['priority_value'], row['priority_name'], row['due_date']), axis=1
)
def get_sentiment_score(user_id):
    """Returns the sentiment score based on historical performance."""
    user_record = user_df[user_df['user_id'] == user_id].iloc[0]
    return user_record['historical_performance_score']

# Apply Sentiment Analysis from User Data
rfi_df['sentiment_score'] = rfi_df['assignee_id'].apply(get_sentiment_score)
# Generate historical resolution time for prediction
rfi_df['resolution_time'] = (rfi_df['updated_at'] - rfi_df['created_at']).dt.days

# Prepare data for training the predictive model
X = rfi_df[['priority_value', 'urgency_score', 'sentiment_score']]
y = rfi_df['resolution_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict Resolution Time
rfi_df['predicted_resolution_time'] = rf_model.predict(X)
# Create the analytics data combining RFI and User data insights
analytics_data = rfi_df[['rfi_id', 'sentiment_score', 'urgency_score', 'predicted_resolution_time']].to_dict(orient='records')

# Display the final analytics data as a list of dictionaries
print("\nFinal Analytics Data:")
for record in analytics_data:
    print(record)
