
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
                "resolution_time": resolution_time
            })
        
        return pd.DataFrame(analysis_results).to_dict(orient="records")

# Class for Assisting with RFI Assignments
class AssignAssistance:
    def __init__(self, user_data, rfi_data):
        self.user_df = pd.DataFrame(user_data)
        self.rfi_df = pd.DataFrame(rfi_data)

    def calculate_rfi_status(self):
        """
        Calculate the number of RFIs that are:
        - On time: Resolved before or on the predicted resolution date.
        - Delayed: Resolved after the predicted resolution date.
        - At risk of delay: Still unresolved but close to exceeding the predicted resolution date.
        """
        status_counts = {"on_time": 0, "delayed": 0, "risk_of_delay": 0}
        current_date = datetime.now()

        for _, rfi in self.rfi_df.iterrows():
            # Calculate the predicted resolution deadline
            predicted_deadline = rfi['created_at'] + timedelta(days=rfi['resolution_time'])

            if pd.notna(rfi['updated_at']):  # If the RFI is resolved
                if rfi['updated_at'] <= predicted_deadline:
                    status_counts["on_time"] += 1
                else:
                    status_counts["delayed"] += 1
            else:  # If the RFI is unresolved
                days_left = (predicted_deadline - current_date).days
                if days_left <= 2:  # Arbitrary threshold for "at risk of delay"
                    status_counts["risk_of_delay"] += 1

        return status_counts

    def calculate_similarity(self, rfi_text, job_title):
        """Calculate similarity between RFI text and user job title using TF-IDF and cosine similarity."""
        if not job_title:  # Handle missing or empty job titles
            return 0
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform([rfi_text, job_title])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarity_matrix[0][0]

    def extract_keywords(self, rfi_questions, job_title):
        """Extract overlapping keywords between RFI questions and user job title using TF-IDF."""
        # Combine the questions into a single string
        rfi_text = ' '.join(rfi_questions)
        
        # Create a TF-IDF vectorizer to get the most important terms
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Use unigrams and bigrams
        tfidf_matrix = tfidf_vectorizer.fit_transform([rfi_text, job_title])
        
        # Get the feature names (terms)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get the importance scores for each term
        importance_scores = tfidf_matrix.sum(axis=0).A1  # Sum over the rows
        
        # Create a dictionary of term -> importance score
        term_importance = dict(zip(feature_names, importance_scores))
        
        # Sort terms by importance (descending)
        sorted_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top 5 most important terms as keywords
        top_keywords = [term for term, _ in sorted_terms[:5]]
        return top_keywords

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
        
        # Fetch workload bounds
            max_workload = max(user['current_workload'] for _, user in self.user_df.iterrows())
            min_workload = min(user['current_workload'] for _, user in self.user_df.iterrows())
        
            weighted_scores = []  # Collect scores for normalization
        
            for _, user in self.user_df.iterrows():
                # Calculate similarity and extract relevant keywords
                similarity = self.calculate_similarity(rfi_text, user['job_title'])
                experience_score = self.calculate_experience_score(rfi_text, user.get('previous_rfi_data', []))
                keywords = self.extract_keywords(rfi['questions_body'], user['job_title'])
                normalized_workload = (user['current_workload'] - min_workload) / (max_workload - min_workload)
            
            # Weighted score: similarity (70%), experience (20%), workload penalty (10%)
                weighted_score = (0.6 * similarity * 100) + (0.395 * experience_score * 100) + (0.005 * (1 - normalized_workload))
                weighted_scores.append(weighted_score)  # Collect scores for normalization
            
            # Append raw data for normalization later
                similarities.append({
                    'id': user['id'],
                    'name': user['name'],
                    'confidence': weighted_score,
                    'workload': user['current_workload'],
                    'reason': keywords
            })
        
        # Normalize scores to 50-100 range
            min_score = min(weighted_scores)
            max_score = max(weighted_scores)
        
            for sim in similarities:
                raw_score = sim['confidence']
                confidence = 50 + ((raw_score - min_score) / (max_score - min_score)) * 50
                sim['confidence'] = float(round(confidence, 2))  # Update confidence
        
        # Sort users by confidence (descending order)
            sorted_by_confidence = sorted(similarities, key=lambda x: x['confidence'], reverse=True)
        
        # Return the top 3 users
            return sorted_by_confidence[:3]




# Example Usage

# Sample User Data
user_data = [
    {
        "id": 101,
        "name": "Alice Johnson",
        "job_title": "Software Engineer",
        "current_workload": 3,
        "historical_performance_score": 78.89,
        "previous_rfi_data": [
            {
                "subject": "Code Refactoring Techniques",
                "questions_body": [
                    "How can we improve code readability?",
                    "What tools can be used for code profiling?"
                ]
            }
        ]
    },
    {
        "id": 102,
        "name": "Bob Smith",
        "job_title": "Software Engineer",
        "current_workload": 2,
        "historical_performance_score": 56.67,
        "previous_rfi_data": [
            {
                "subject": "API Integration Methods",
                "questions_body": [
                    "What is the best practice for REST API integration?",
                    "How do we handle API rate limiting?"
                ]
            }
        ]
    },
    {
        "id": 103,
        "name": "Charlie Davis",
        "job_title": "Software Engineer",
        "current_workload": 1,
        "historical_performance_score": 45.35,
        "previous_rfi_data": [
            {
                "subject": "Error Handling Strategies",
                "questions_body": [
                    "What are common error handling strategies in Python?",
                    "How can we log errors effectively?"
                ]
            }
        ]
    },
    {
        "id": 104,
        "name": "David Brown",
        "job_title": "Project Manager",
        "current_workload": 4,
        "historical_performance_score": .20,
        "previous_rfi_data": [
            {
                "subject": "Project Timeline Management",
                "questions_body": [
                    "What tools can be used for Gantt charts?",
                    "How do we track task dependencies effectively?"
                ]
            }
        ]
    },
    {
        "id": 105,
        "name": "Eve White",
        "job_title": "Project Manager",
        "current_workload": 3,
        "historical_performance_score": 0.99,
        "previous_rfi_data": [
            {
                "subject": "Budget Allocation",
                "questions_body": [
                    "How do we ensure budget distribution across teams?",
                    "What tools assist in tracking expenses in real time?"
                ]
            }
        ]
    },
    {
        "id": 106,
        "name": "Fay Clark",
        "job_title": "Project Manager",
        "current_workload": 2,
        "historical_performance_score": 0.43,
        "previous_rfi_data": [
            {
                "subject": "Stakeholder Communication",
                "questions_body": [
                    "What methods can be used to streamline stakeholder updates?",
                    "How do we manage conflicts between stakeholders?"
                ]
            }
        ]
    },
    {
        "id": 107,
        "name": "Grace Lewis",
        "job_title": "Data Analyst",
        "current_workload": 1,
        "historical_performance_score": 0.64,
        "previous_rfi_data": [
            {
                "subject": "Data Cleaning Techniques",
                "questions_body": [
                    "What tools are effective for data deduplication?",
                    "How can we handle missing data effectively?"
                ]
            }
        ]
    },
    {
        "id": 108,
        "name": "Hank Wilson",
        "job_title": "Data Analyst",
        "current_workload": 2,
        "historical_performance_score": 0.82,
        "previous_rfi_data": [
            {
                "subject": "Data Visualization Tools",
                "questions_body": [
                    "What visualization libraries are available in Python?",
                    "How do we compare multiple datasets visually?"
                ]
            }
        ]
    },
    {
        "id": 109,
        "name": "Ivy Martinez",
        "job_title": "Data Analyst",
        "current_workload": 3,
        "historical_performance_score": 0.90,
        "previous_rfi_data": [
            {
                "subject": "Statistical Analysis Methods",
                "questions_body": [
                    "What are common statistical tests for correlation?",
                    "How can we ensure data normalization?"
                ]
            }
        ]
    },
    {
        "id": 110,
        "name": "Jack Taylor",
        "job_title": "Quality Assurance Engineer",
        "current_workload": 3,
        "historical_performance_score": 0.78,
        "previous_rfi_data": [
            {
                "subject": "Automated Testing Frameworks",
                "questions_body": [
                    "What are the best tools for UI testing?",
                    "How can we integrate Selenium with Jenkins?"
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

# # Instantiate the AssignAssistance class
# assign_assist = AssignAssistance(user_data, rfi_data)

# # Calculate RFI status
# rfi_status = assign_assist.calculate_rfi_status()
# print(rfi_status)

# Run RFI Analysis
rfi_analysis = RfiAnalysis(rfi_data)
analysis_results = rfi_analysis.run_analysis()
print("RFI Analysis Results:", analysis_results)

# Suggest Assignees for Each RFI
assign_assistance = AssignAssistance(user_data, rfi_data)
assignments = {rfi['id']: assign_assistance.suggest_top_assignees(rfi) for rfi in rfi_data}
print("\nSuggested Assignments Dashboard:", assignments)
