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

# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime, timedelta
# from textblob import TextBlob

# # Sample User Data
# user_data = [
#     {"user_id": 101, "name": "John Doe", "job_title": "Project Manager", "current_workload": 3, "historical_performance_score": 0.85, {"previous_rfi_data": ["subject": "Software Testing Methodology", "questions_body": ["Are there any delays in the delivery of building materials?", "What alternatives do we have for material sourcing?"]]}},
#     {"user_id": 102, "name": "Alice Smith", "job_title": "Electrical Engineer", "current_workload": 2, "historical_performance_score": 0.92},
#     {"user_id": 103, "name": "Bob Johnson", "job_title": "Mechanical Engineer", "current_workload": 4, "historical_performance_score": 0.80},
#     {"user_id": 104, "name": "Carol Williams", "job_title": "Civil Engineer", "current_workload": 5, "historical_performance_score": 0.75},
#     {"user_id": 105, "name": "David Lee", "job_title": "Quality Assurance", "current_workload": 1, "historical_performance_score": 0.95},
#     {"user_id": 106, "name": "Emma Moore", "job_title": "Software Developer", "current_workload": 6, "historical_performance_score": 0.78},
#     {"user_id": 107, "name": "Frank Harris", "job_title": "Data Scientist", "current_workload": 3, "historical_performance_score": 0.90},
#     {"user_id": 108, "name": "Grace Clark", "job_title": "HR Manager", "current_workload": 4, "historical_performance_score": 0.88},
#     {"user_id": 109, "name": "Henry Lewis", "job_title": "Business Analyst", "current_workload": 2, "historical_performance_score": 0.85},
#     {"user_id": 110, "name": "Ivy Walker", "job_title": "Marketing Specialist", "current_workload": 3, "historical_performance_score": 0.80}
# ]

# # Sample RFI Data
# rfi_data = [
#     {"id": 1, "subject": "Project Scope Clarification", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What is the final scope of the project?", "Are there any changes to the project scope from the initial plan?"], "project_id": 501},
#     {"id": 2, "subject": "Electrical Safety Standards", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=5), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What are the electrical safety standards for this project?", "How should we ensure compliance with the safety standards?"], "project_id": 502},
#     {"id": 3, "subject": "Mechanical Equipment Specifications", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=6), "questions_body": ["What are the specifications for mechanical equipment?", "Do we need any special approvals for the equipment?"], "project_id": 503},
#     {"id": 4, "subject": "Construction Site Logistics", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=8), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["What is the logistics plan for the construction site?", "What are the safety measures for construction equipment?"], "project_id": 504},
#     {"id": 5, "subject": "Building Material Delays", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["Are there any delays in the delivery of building materials?", "What alternatives do we have for material sourcing?"], "project_id": 505},
#     {"id": 6, "subject": "Software Testing Methodology", "priority_value": 3, "status": "Closed", "created_at": datetime.now() - timedelta(days=10), "due_date": datetime.now() + timedelta(days=5), "questions_body": ["What testing methodology should we follow for this project?", "How can we improve our testing efficiency?"], "project_id": 506},
#     {"id": 7, "subject": "HR Policy Updates", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=7), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What changes have been made to the HR policies?", "How can we effectively communicate policy updates?"], "project_id": 507},
#     {"id": 8, "subject": "Project Budget Adjustment", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["Why was the project budget adjusted?", "How will the changes affect the project timeline?"], "project_id": 508},
#     {"id": 9, "subject": "New Software Features", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=7), "questions_body": ["What are the new features being introduced in the software?", "How will the new features affect the end user?"], "project_id": 509},
#     {"id": 10, "subject": "Construction Timeline Adjustments", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=9), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What are the reasons for timeline adjustments?", "How do we plan to minimize project delays?"], "project_id": 510},
#     {"id": 11, "subject": "Quality Control Standards", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["What quality control standards should we follow?", "How can we ensure that all items meet these standards?"], "project_id": 511},
#     {"id": 12, "subject": "Risk Management Plan", "priority_value": 4, "status": "Closed", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What is the risk management plan for the project?", "How do we handle unforeseen risks during the project?"], "project_id": 512},
#     {"id": 13, "subject": "Client Feedback Analysis", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=6), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["What are the key points from client feedback?", "How can we address the client's concerns effectively?"], "project_id": 513},
#     {"id": 14, "subject": "Environmental Impact Assessment", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=7), "questions_body": ["What is the environmental impact of the project?", "How can we mitigate environmental damage?"], "project_id": 514},
#     {"id": 15, "subject": "Vendor Evaluation Process", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=5), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["How should we evaluate potential vendors?", "What criteria should be considered in the vendor evaluation?"], "project_id": 515},
#     {"id": 16, "subject": "Training Needs Assessment", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=8), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["What are the training needs for the project team?", "How can we address skill gaps effectively?"], "project_id": 516},
#     {"id": 17, "subject": "Project Risk Assessment", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=5), "questions_body": ["What are the risks associated with the project?", "How do we mitigate these risks effectively?"], "project_id": 517},
#     {"id": 18, "subject": "Stakeholder Engagement Plan", "priority_value": 2, "status": "Closed", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["How do we engage with stakeholders?", "What strategies do we use to keep them informed?"], "project_id": 518},
#     {"id": 19, "subject": "Contract Negotiation Clarifications", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=6), "questions_body": ["What are the terms and conditions of the contract?", "How do we handle contract amendments?"], "project_id": 519},
#     {"id": 20, "subject": "Post-Project Review", "priority_value": 4, "status": "Closed", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What went well in the project?", "What improvements can be made for future projects?"], "project_id": 520}
# ]

# class RfiAnalysis:
#     def __init__(self, rfi_data):
#         self.rfi_df = pd.DataFrame(rfi_data)

#     def calculate_sentiment(self, text):
#         """Analyze sentiment using TextBlob"""
#         return TextBlob(text).sentiment.polarity

#     def calculate_urgency(self, rfi):
#         """Determine urgency based on priority and due date proximity"""
#         time_left = (rfi['due_date'] - datetime.now()).days
#         urgency = rfi['priority_value'] * (1 / (time_left + 1))  # Simple formula
#         return urgency

#     def predict_resolution_time(self, rfi):
#         """Estimate resolution time based on priority and complexity of questions"""
#         complexity = len(rfi['questions_body'])  # Assume more questions means higher complexity
#         resolution_time = max(1, complexity * (6 - rfi['priority_value']))  # Time in hours
#         return resolution_time

#     def run_analysis(self):
#         """Analyze all RFIs and calculate sentiment, urgency, and predicted resolution time"""
#         analysis_results = []
#         for _, rfi in self.rfi_df.iterrows():
#             rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
#             sentiment_score = self.calculate_sentiment(rfi_text)
#             urgency_score = self.calculate_urgency(rfi)
#             resolution_time = self.predict_resolution_time(rfi)
            
#             analysis_results.append({
#                 "rfi_id": rfi['id'],
#                 "sentiment_score": sentiment_score,
#                 "urgency_score": urgency_score,
#                 "predicted_resolution_time": resolution_time
#             })
        
#         return pd.DataFrame(analysis_results).to_dict(orient="records")

# class AssignAssistance:
#     def __init__(self, user_data, rfi_data):
#         self.user_df = pd.DataFrame(user_data)
#         self.rfi_df = pd.DataFrame(rfi_data)

#     def calculate_similarity(self, rfi_text, job_title):
#         """Measure similarity between RFI text and user job title using TF-IDF and cosine similarity"""
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#         all_text = [rfi_text, job_title]
#         tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
#         similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
#         return similarity_matrix[0][0]

#     def extract_keywords(self, rfi_questions, job_title):
#         """Extract overlapping keywords between RFI questions and user job title"""
#         keywords = []
#         rfi_words = set(' '.join(rfi_questions).lower().split())
#         job_title_words = set(job_title.lower().split())
#         keywords = rfi_words.intersection(job_title_words)
#         return list(keywords)

#     def suggest_top_assignees(self, rfi):
#         """Rank top 3 potential assignees based on similarity to RFI text and current workload"""
#         rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
#         similarities = []
        
#         for _, user in self.user_df.iterrows():
#             similarity = self.calculate_similarity(rfi_text, user['job_title'])
#             keywords = self.extract_keywords(rfi['questions_body'], rfi['subject'])
#             similarities.append({
#                 'user_id': user['user_id'],
#                 'name': user['name'],
#                 'confidence': int(similarity * 100),  # Convert to integer percentage
#                 'workload': user['current_workload'],
#                 'reason': keywords
#             })
        
#         # Sort by similarity score and workload, prioritize users with lower workload
#         sorted_assignees = sorted(similarities, key=lambda x: (-x['confidence'], x['workload']))
#         top_assignees = sorted_assignees[:3]  # Get top 3 assignees
#         return top_assignees

# # Example Usage

# # Analyze all RFIs
# rfi_analysis = RfiAnalysis(rfi_data)
# analysis_dict = rfi_analysis.run_analysis()
# print("RFI Analysis Results:", analysis_dict)

# # Suggest assignees for an RFI (using the first RFI in the list as an example)
# assign_assistance = AssignAssistance(user_data, rfi_data)
# dashboard_dict = {
#     rfi['id']: assign_assistance.suggest_top_assignees(rfi) for rfi in rfi_data
# }
# print("\n\nSuggested Assignees Dashboard:", dashboard_dict)