import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from textblob import TextBlob

# Sample User Data
user_data = [
    {"user_id": 101, "name": "John Doe", "job_title": "Project Manager", "current_workload": 3, "historical_performance_score": 0.85, "expertise": ["project management", "material delay", "team leadership"]},
    {"user_id": 102, "name": "Alice Smith", "job_title": "Electrical Engineer", "current_workload": 2, "historical_performance_score": 0.92, "expertise": ["electrical wiring", "technical clarification", "circuit design"]},
    {"user_id": 103, "name": "Bob Johnson", "job_title": "Mechanical Engineer", "current_workload": 4, "historical_performance_score": 0.80, "expertise": ["mechanical design", "CAD", "construction"]},
    {"user_id": 104, "name": "Carol Williams", "job_title": "Civil Engineer", "current_workload": 5, "historical_performance_score": 0.75, "expertise": ["structural design", "construction", "site management"]},
    {"user_id": 105, "name": "David Lee", "job_title": "Quality Assurance", "current_workload": 1, "historical_performance_score": 0.95, "expertise": ["quality control", "testing", "process improvement"]},
    {"user_id": 106, "name": "Emma Moore", "job_title": "Software Developer", "current_workload": 6, "historical_performance_score": 0.78, "expertise": ["software development", "web applications", "JavaScript"]},
    {"user_id": 107, "name": "Frank Harris", "job_title": "Data Scientist", "current_workload": 3, "historical_performance_score": 0.90, "expertise": ["data analysis", "machine learning", "statistics"]},
    {"user_id": 108, "name": "Grace Clark", "job_title": "HR Manager", "current_workload": 4, "historical_performance_score": 0.88, "expertise": ["HR management", "employee relations", "recruitment"]},
    {"user_id": 109, "name": "Henry Lewis", "job_title": "Business Analyst", "current_workload": 2, "historical_performance_score": 0.85, "expertise": ["business analysis", "requirements gathering", "process optimization"]},
    {"user_id": 110, "name": "Ivy Walker", "job_title": "Marketing Specialist", "current_workload": 3, "historical_performance_score": 0.80, "expertise": ["marketing", "digital campaigns", "brand management"]}
]

# Sample RFI Data
rfi_data = [
    {"id": 1, "subject": "Project Scope Clarification", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What is the final scope of the project?", "Are there any changes to the project scope from the initial plan?"], "project_id": 501},
    {"id": 2, "subject": "Electrical Safety Standards", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=5), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What are the electrical safety standards for this project?", "How should we ensure compliance with the safety standards?"], "project_id": 502},
    {"id": 3, "subject": "Mechanical Equipment Specifications", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=6), "questions_body": ["What are the specifications for mechanical equipment?", "Do we need any special approvals for the equipment?"], "project_id": 503},
    {"id": 4, "subject": "Construction Site Logistics", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=8), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["What is the logistics plan for the construction site?", "What are the safety measures for construction equipment?"], "project_id": 504},
    {"id": 5, "subject": "Building Material Delays", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["Are there any delays in the delivery of building materials?", "What alternatives do we have for material sourcing?"], "project_id": 505},
    {"id": 6, "subject": "Software Testing Methodology", "priority_value": 3, "status": "Closed", "created_at": datetime.now() - timedelta(days=10), "due_date": datetime.now() + timedelta(days=5), "questions_body": ["What testing methodology should we follow for this project?", "How can we improve our testing efficiency?"], "project_id": 506},
    {"id": 7, "subject": "HR Policy Updates", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=7), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What changes have been made to the HR policies?", "How can we effectively communicate policy updates?"], "project_id": 507},
    {"id": 8, "subject": "Project Budget Adjustment", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["Why was the project budget adjusted?", "How will the changes affect the project timeline?"], "project_id": 508},
    {"id": 9, "subject": "New Software Features", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=7), "questions_body": ["What are the new features being introduced in the software?", "How will the new features affect the end user?"], "project_id": 509},
    {"id": 10, "subject": "Construction Timeline Adjustments", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=9), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What are the reasons for timeline adjustments?", "How do we plan to minimize project delays?"], "project_id": 510},
    {"id": 11, "subject": "Quality Control Standards", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["What quality control standards should we follow?", "How can we ensure that all items meet these standards?"], "project_id": 511},
    {"id": 12, "subject": "Risk Management Plan", "priority_value": 4, "status": "Closed", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["What is the risk management plan for the project?", "How do we handle unforeseen risks during the project?"], "project_id": 512},
    {"id": 13, "subject": "Client Feedback Analysis", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=6), "due_date": datetime.now() + timedelta(days=1), "questions_body": ["What are the key points from client feedback?", "How can we address the client's concerns effectively?"], "project_id": 513},
    {"id": 14, "subject": "Environmental Impact Assessment", "priority_value": 2, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=7), "questions_body": ["What is the environmental impact of the project?", "How can we mitigate environmental damage?"], "project_id": 514},
    {"id": 15, "subject": "Vendor Evaluation Process", "priority_value": 4, "status": "Open", "created_at": datetime.now() - timedelta(days=5), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["How should we evaluate potential vendors?", "What criteria should be considered in the vendor evaluation?"], "project_id": 515},
    {"id": 16, "subject": "Training Needs Assessment", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=8), "due_date": datetime.now() + timedelta(days=3), "questions_body": ["What are the training needs for the project team?", "How can we address skill gaps effectively?"], "project_id": 516},
    {"id": 17, "subject": "Project Risk Assessment", "priority_value": 5, "status": "Open", "created_at": datetime.now() - timedelta(days=2), "due_date": datetime.now() + timedelta(days=5), "questions_body": ["What are the risks associated with the project?", "How do we mitigate these risks effectively?"], "project_id": 517},
    {"id": 18, "subject": "Stakeholder Engagement Plan", "priority_value": 2, "status": "Closed", "created_at": datetime.now() - timedelta(days=1), "due_date": datetime.now() + timedelta(days=2), "questions_body": ["How do we engage with stakeholders?", "What strategies do we use to keep them informed?"], "project_id": 518},
    {"id": 19, "subject": "Contract Negotiation Clarifications", "priority_value": 3, "status": "Open", "created_at": datetime.now() - timedelta(days=4), "due_date": datetime.now() + timedelta(days=6), "questions_body": ["What are the terms and conditions of the contract?", "How do we handle contract amendments?"], "project_id": 519},
    {"id": 20, "subject": "Post-Project Review", "priority_value": 4, "status": "Closed", "created_at": datetime.now() - timedelta(days=3), "due_date": datetime.now() + timedelta(days=4), "questions_body": ["What went well in the project?", "What improvements can be made for future projects?"], "project_id": 520}
]

class RfiAnalysis:
    def __init__(self, rfi_data):
        self.rfi_df = pd.DataFrame(rfi_data)

    def calculate_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        return TextBlob(text).sentiment.polarity

    def calculate_urgency(self, rfi):
        """Determine urgency based on priority and due date proximity"""
        time_left = (rfi['due_date'] - datetime.now()).days
        urgency = rfi['priority_value'] * (1 / (time_left + 1))  # Simple formula
        return urgency

    def predict_resolution_time(self, rfi):
        """Estimate resolution time based on priority and complexity of questions"""
        complexity = len(rfi['questions_body'])  # Assume more questions means higher complexity
        resolution_time = max(1, complexity * (6 - rfi['priority_value']))  # Time in hours
        return resolution_time

    def run_analysis(self):
        """Analyze all RFIs and calculate sentiment, urgency, and predicted resolution time"""
        analysis_results = []
        for _, rfi in self.rfi_df.iterrows():
            rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
            sentiment_score = self.calculate_sentiment(rfi_text)
            urgency_score = self.calculate_urgency(rfi)
            resolution_time = self.predict_resolution_time(rfi)
            
            analysis_results.append({
                "rfi_id": rfi['id'],
                "sentiment_score": sentiment_score,
                "urgency_score": urgency_score,
                "predicted_resolution_time": resolution_time
            })
        
        return pd.DataFrame(analysis_results)

class AssignAssistance:
    def __init__(self, user_data, rfi_data):
        self.user_df = pd.DataFrame(user_data)
        self.rfi_df = pd.DataFrame(rfi_data)

    def calculate_similarity(self, rfi_text, expertise):
        """Measure similarity between RFI and user expertise using TF-IDF and cosine similarity"""
        all_expertise = [' '.join(expert) for expert in expertise]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        all_text = [rfi_text] + all_expertise
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarity_matrix[0]

    def suggest_top_assignees(self, rfi):
        """Rank top 3 potential assignees based on similarity to RFI text and current workload"""
        rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
        similarities = []
        
        for _, user in self.user_df.iterrows():
            similarity = self.calculate_similarity(rfi_text, user['expertise'])
            similarities.append({
                'user_id': user['user_id'],
                'similarity_score': np.mean(similarity),  # Averaging similarity score
                'workload': user['current_workload']
            })
        
        # Sort by similarity score and workload, prioritize users with lower workload
        sorted_assignees = sorted(similarities, key=lambda x: (x['similarity_score'], x['workload']))
        top_assignees = sorted_assignees[:3]  # Get top 3 assignees
        return top_assignees

# Example Usage

# Analyze all RFIs
rfi_analysis = RfiAnalysis(rfi_data)
analysis_df = rfi_analysis.run_analysis()
print("RFI Analysis Results:\n", analysis_df)

# Suggest assignees for an RFI (using the first RFI in the list as an example)
assign_assistance = AssignAssistance(user_data, rfi_data)
top_assignees = assign_assistance.suggest_top_assignees(rfi_data[0])  # First RFI as an example
print("Suggested Assignees:\n", top_assignees)
