import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from textblob import TextBlob

# Class for Analyzing RFIs
class RfiAnalysis:
    def __init__(self, rfi_data):
        self.rfi_df = pd.DataFrame(rfi_data)

    def calculate_sentiment(self, text):
        """Analyze sentiment using TextBlob (range: -1.0 to 1.0)."""
        return TextBlob(text).sentiment.polarity

    def calculate_urgency(self, rfi):
        """Determine urgency based on priority and proximity to due date."""
        time_left = max((rfi['due_date'] - datetime.now()).days, 0)  # Prevent negative time
        urgency = rfi['priority_value'] * (1 / (time_left + 1))  # Higher priority and closer due date = higher urgency
        return round(urgency, 2)

    def predict_resolution_time(self, rfi):
        """Estimate resolution time in hours based on priority and question complexity."""
        complexity = len(rfi['questions_body'])  # Number of questions signifies complexity
        resolution_time = max(1, complexity * (6 - rfi['priority_value']))  # Higher priority reduces time
        return resolution_time

    def run_analysis(self):
        """Analyze all RFIs and calculate sentiment, urgency, and predicted resolution time."""
        analysis_results = []
        for _, rfi in self.rfi_df.iterrows():
            # Combine subject and question text for sentiment analysis
            rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
            
            sentiment_score = self.calculate_sentiment(rfi_text)
            urgency_score = self.calculate_urgency(rfi)
            resolution_time = self.predict_resolution_time(rfi)
            
            # Append the results as a dictionary
            analysis_results.append({
                "rfi_id": rfi['id'],
                "sentiment_score": sentiment_score,
                "urgency_score": urgency_score,
                "predicted_resolution_time": resolution_time
            })
        
        return pd.DataFrame(analysis_results).to_dict(orient="records")

# Class for Assisting with RFI Assignments
class AssignAssistance:
    def __init__(self, user_data, rfi_data):
        self.user_df = pd.DataFrame(user_data)
        self.rfi_df = pd.DataFrame(rfi_data)

    def calculate_similarity(self, rfi_text, job_title):
        """Calculate similarity between RFI text and user job title using TF-IDF and cosine similarity."""
        if not job_title:  # Handle missing or empty job titles
            return 0
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform([rfi_text, job_title])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarity_matrix[0][0]

    def extract_keywords(self, rfi_questions, job_title):
        """Extract overlapping keywords between RFI questions and user job title."""
        rfi_words = set(' '.join(rfi_questions).lower().split())
        job_title_words = set(job_title.lower().split())
        return list(rfi_words.intersection(job_title_words))

    def calculate_experience_score(self, rfi_text, previous_rfi_data):
        """Calculate experience score based on user's previous RFIs."""
        if not previous_rfi_data:
            return 0
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        scores = []
        for prev_rfi in previous_rfi_data:
            combined_text = prev_rfi['subject'] + ' ' + ' '.join(prev_rfi['questions_body'])
            tfidf_matrix = tfidf_vectorizer.fit_transform([rfi_text, combined_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            scores.append(similarity[0][0])
        return np.mean(scores) if scores else 0

    def suggest_top_assignees(self, rfi):
        """Rank top 3 potential assignees based on similarity to RFI text, workload, and experience."""
        rfi_text = rfi['subject'] + ' ' + ' '.join(rfi['questions_body'])
        similarities = []
        
        for _, user in self.user_df.iterrows():
            # Calculate similarity and extract relevant keywords
            similarity = self.calculate_similarity(rfi_text, user['job_title'])
            experience_score = self.calculate_experience_score(rfi_text, user.get('previous_rfi_data', []))
            keywords = self.extract_keywords(rfi['questions_body'], user['job_title'])
            
            # Weighted score: similarity (70%), experience (20%), workload penalty (10%)
            weighted_score = 0.7 * similarity + 0.2 * experience_score - 0.1 * user['current_workload']
            
            # Append data for each user
            similarities.append({
                'user_id': user['user_id'],
                'name': user['name'],
                'confidence': int(weighted_score * 100),  # Convert to percentage
                'workload': user['current_workload'],
                'reason': keywords
            })
        
        # Sort users by confidence and workload (lower workload preferred)
        sorted_assignees = sorted(similarities, key=lambda x: (-x['confidence'], x['workload']))
        return sorted_assignees[:3]  # Return top 3 suggestions

# Example Usage

# Sample User Data
user_data = [
    {
        "user_id": 102,
        "name": "Jane Smith",
        "job_title": "Software Engineer",
        "current_workload": 4,
        "historical_performance_score": 0.90,
        "previous_rfi_data": [
            {
                "subject": "Code Optimization",
                "questions_body": [
                    "How can we improve the performance of this function?",
                    "What tools are available for code profiling?"
                ]
            }
        ]
    },
    {
        "user_id": 103,
        "name": "Michael Johnson",
        "job_title": "Data Analyst",
        "current_workload": 2,
        "historical_performance_score": 0.92,
        "previous_rfi_data": [
            {
                "subject": "Data Cleaning Techniques",
                "questions_body": [
                    "What methods can be used to handle missing values?",
                    "Are there any tools to automate data cleaning?"
                ]
            }
        ]
    },
    {
        "user_id": 104,
        "name": "Emily White",
        "job_title": "Quality Assurance Engineer",
        "current_workload": 3,
        "historical_performance_score": 0.88,
        "previous_rfi_data": [
            {
                "subject": "Testing Automation Tools",
                "questions_body": [
                    "What are the best automation tools for web testing?",
                    "How can we implement continuous testing in our CI/CD pipeline?"
                ]
            }
        ]
    },
    {
        "user_id": 105,
        "name": "Robert Brown",
        "job_title": "Product Manager",
        "current_workload": 5,
        "historical_performance_score": 0.87,
        "previous_rfi_data": [
            {
                "subject": "Product Launch Strategies",
                "questions_body": [
                    "What are the key stages of a product launch?",
                    "How do we measure the success of a product launch?"
                ]
            }
        ]
    },
    {
        "user_id": 106,
        "name": "Linda Green",
        "job_title": "Technical Writer",
        "current_workload": 2,
        "historical_performance_score": 0.93,
        "previous_rfi_data": [
            {
                "subject": "API Documentation Standards",
                "questions_body": [
                    "What are the best practices for API documentation?",
                    "How can we make our API guides more user-friendly?"
                ]
            }
        ]
    },
    {
        "user_id": 107,
        "name": "James Wilson",
        "job_title": "Network Engineer",
        "current_workload": 4,
        "historical_performance_score": 0.89,
        "previous_rfi_data": [
            {
                "subject": "Network Security Protocols",
                "questions_body": [
                    "What encryption methods are recommended for securing data?",
                    "How do we monitor for unauthorized network access?"
                ]
            }
        ]
    },
    {
        "user_id": 108,
        "name": "Sarah Taylor",
        "job_title": "Business Analyst",
        "current_workload": 3,
        "historical_performance_score": 0.91,
        "previous_rfi_data": [
            {
                "subject": "Market Research Techniques",
                "questions_body": [
                    "What are the primary tools for market analysis?",
                    "How can we gather competitor data effectively?"
                ]
            }
        ]
    },
    {
        "user_id": 109,
        "name": "William Martinez",
        "job_title": "UX Designer",
        "current_workload": 2,
        "historical_performance_score": 0.94,
        "previous_rfi_data": [
            {
                "subject": "UI Design Principles",
                "questions_body": [
                    "What are the key principles of effective UI design?",
                    "How do we conduct usability testing effectively?"
                ]
            }
        ]
    },
    {
        "user_id": 110,
        "name": "Elizabeth Clark",
        "job_title": "DevOps Engineer",
        "current_workload": 3,
        "historical_performance_score": 0.90,
        "previous_rfi_data": [
            {
                "subject": "CI/CD Pipeline Optimization",
                "questions_body": [
                    "What tools can help optimize the CI/CD pipeline?",
                    "How can we reduce build times in our pipeline?"
                ]
            }
        ]
    },
    {
        "user_id": 111,
        "name": "Daniel Lewis",
        "job_title": "AI Researcher",
        "current_workload": 1,
        "historical_performance_score": 0.96,
        "previous_rfi_data": [
            {
                "subject": "AI Model Training Techniques",
                "questions_body": [
                    "What strategies can improve model accuracy?",
                    "How do we prevent overfitting during model training?"
                ]
            }
        ]
    }
]


# Sample RFI Data
rfi_data = [
    {
        "id": 1,
        "subject": "Project Scope Clarification",
        "priority_value": 4,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=3),
        "due_date": datetime.now() + timedelta(days=2),
        "questions_body": [
            "What is the final scope of the project?",
            "Are there any changes to the project scope from the initial plan?"
        ],
        "project_id": 501
    },
    {
        "id": 2,
        "subject": "Material Procurement Timeline",
        "priority_value": 3,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=5),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": [
            "What is the expected delivery date for materials?",
            "Are there any potential delays in procurement?"
        ],
        "project_id": 502
    },
    {
        "id": 3,
        "subject": "Quality Assurance Standards",
        "priority_value": 5,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=7),
        "due_date": datetime.now() + timedelta(days=3),
        "questions_body": [
            "What QA standards will be applied?",
            "Is there a need for third-party QA verification?"
        ],
        "project_id": 503
    },
    {
        "id": 4,
        "subject": "Software Version Control",
        "priority_value": 4,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=2),
        "due_date": datetime.now() + timedelta(days=4),
        "questions_body": [
            "Which version control system should be used?",
            "Are there guidelines for repository structure?"
        ],
        "project_id": 504
    },
    {
        "id": 5,
        "subject": "Budget Allocation Clarification",
        "priority_value": 2,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=6),
        "due_date": datetime.now() + timedelta(days=6),
        "questions_body": [
            "What percentage of the budget is allocated for resources?",
            "Is there flexibility in budget allocation?"
        ],
        "project_id": 505
    },
    {
        "id": 6,
        "subject": "Risk Management Plan",
        "priority_value": 5,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=10),
        "due_date": datetime.now() + timedelta(days=8),
        "questions_body": [
            "What are the identified project risks?",
            "How will risk mitigation be handled?"
        ],
        "project_id": 506
    },
    {
        "id": 7,
        "subject": "Team Roles and Responsibilities",
        "priority_value": 3,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=4),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": [
            "What are the defined roles for the project team?",
            "Who is responsible for approvals?"
        ],
        "project_id": 507
    },
    {
        "id": 8,
        "subject": "Client Feedback Mechanism",
        "priority_value": 4,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=8),
        "due_date": datetime.now() + timedelta(days=7),
        "questions_body": [
            "How frequently will feedback be collected?",
            "What platform will be used for feedback collection?"
        ],
        "project_id": 508
    },
    {
        "id": 9,
        "subject": "Software Testing Requirements",
        "priority_value": 5,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=3),
        "due_date": datetime.now() + timedelta(days=4),
        "questions_body": [
            "What are the minimum testing requirements?",
            "Is there a need for manual testing or only automation?"
        ],
        "project_id": 509
    },
    {
        "id": 10,
        "subject": "Vendor Selection Criteria",
        "priority_value": 3,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=6),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": [
            "What are the criteria for selecting vendors?",
            "Is there a preferred vendor list available?"
        ],
        "project_id": 510
    },
    {
        "id": 11,
        "subject": "Data Privacy Compliance",
        "priority_value": 5,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=5),
        "due_date": datetime.now() + timedelta(days=3),
        "questions_body": [
            "How will sensitive data be handled?",
            "Are there any specific compliance standards to follow?"
        ],
        "project_id": 511
    },
    {
        "id": 12,
        "subject": "Server Capacity Planning",
        "priority_value": 4,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=4),
        "due_date": datetime.now() + timedelta(days=6),
        "questions_body": [
            "What is the expected server load capacity?",
            "Are there backup servers in case of failure?"
        ],
        "project_id": 512
    },
    {
        "id": 13,
        "subject": "Legal Contract Review",
        "priority_value": 5,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=7),
        "due_date": datetime.now() + timedelta(days=5),
        "questions_body": [
            "Who will be responsible for contract review?",
            "Are there any standard contract templates available?"
        ],
        "project_id": 513
    },
    {
        "id": 14,
        "subject": "Hardware Specification Clarification",
        "priority_value": 3,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=3),
        "due_date": datetime.now() + timedelta(days=3),
        "questions_body": [
            "What are the minimum hardware specifications required?",
            "Are there any preferred hardware vendors?"
        ],
        "project_id": 514
    },
    {
        "id": 15,
        "subject": "Training Program Schedule",
        "priority_value": 4,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=5),
        "due_date": datetime.now() + timedelta(days=4),
        "questions_body": [
            "When will the training sessions start?",
            "How long will each session last?"
        ],
        "project_id": 515
    },
    {
        "id": 16,
        "subject": "Project Deadline Extension",
        "priority_value": 5,
        "status": "Open",
        "created_at": datetime.now() - timedelta(days=2),
        "due_date": datetime.now() + timedelta(days=2),
        "questions_body": [
            "What factors justify the extension?",
            "How will the new deadline impact project milestones?"
        ],
        "project_id": 516
    },
    {
        "id": 17,
        "subject": "User Acceptance Testing (UAT)",
        "priority_value": 5,
        "status": "In Progress",
        "created_at": datetime.now() - timedelta(days=6),
        "due_date": datetime.now() + timedelta(days=7),
        "questions_body": [
            "What criteria define a successful UAT?",
            "Who will participate in the testing process?"
        ],
        "project_id": 517
    }
]

# Run RFI Analysis
rfi_analysis = RfiAnalysis(rfi_data)
analysis_results = rfi_analysis.run_analysis()
print("RFI Analysis Results:", analysis_results)

# Suggest Assignees for Each RFI
assign_assistance = AssignAssistance(user_data, rfi_data)
assignments = {rfi['id']: assign_assistance.suggest_top_assignees(rfi) for rfi in rfi_data}
print("\nSuggested Assignments Dashboard:", assignments)
