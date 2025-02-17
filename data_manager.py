import json
import re
from pathlib import Path
import pandas as pd

class DataManager:
    def __init__(self):
        self.cs_patterns = [
            r"customer success",
            r"customer experience",
            r"customer satisfaction",
            r"customer retention",
            r"churn reduction",
            r"NPS",
            r"CSAT",
            r"customer journey",
            r"customer onboarding",
            r"relationship management",
            r"voice of customer",
            r"VoC",
            r"first-call resolution",
            r"customer-centric",
            r"customer feedback",
            r"customer engagement",
            r"customer support",
            r"customer service",
            r"customer lifecycle",
            r"customer analytics"
        ]
        
        self.cs_patterns.extend([
            # Member Services Specific
            r"member services",
            r"member satisfaction",
            r"member experience",
            
            # Leadership & Development
            r"talent development",
            r"team leadership",
            r"coaching",
            r"supervisors",
            
            # Operational Excellence
            r"operational excellence",
            r"service delivery",
            r"performance measures",
            r"critical KPIs",
            
            # Business Operations
            r"utilization",
            r"conversion rate",
            r"cross-functional",
            r"operational processes"
        ])
        
        self.cs_tools = [
            "Salesforce",
            "Gainsight",
            "Zendesk",
            "Intercom",
            "ChurnZero",
            "Totango",
            "HubSpot",
            "Freshdesk",
            "RingCentral",
            "Power BI",
            "Tableau",
            "SQL",
            "Python",
            "TensorFlow",
            "scikit-learn",
            "Asana",
            "Jira",
            "Slack",
            "Microsoft Teams"
        ]
        
        self.cs_metrics = [
            r"\d+%\s*(increase|improvement|reduction)",
            r"NPS",
            r"CSAT",
            r"first-call resolution",
            r"customer satisfaction",
            r"churn rate",
            r"response time",
            r"handling time",
            r"conversion rate"
        ]

    def extract_keywords(self, job_description):
        # Direct keyword mapping for this role
        key_terms = {
            'leadership': [
                'director', 'lead', 'manage', 'develop', 'inspire',
                'coaching', 'supervisors', 'talent development'
            ],
            'member_service': [
                'member services', 'member satisfaction', 'customer service',
                'service delivery', 'customer support', 'service excellence'
            ],
            'operations': [
                'operational excellence', 'kpis', 'metrics', 'performance measures',
                'utilization', 'conversion rate', 'processes'
            ]
        }
        
        keywords = []
        for category, terms in key_terms.items():
            for term in terms:
                if term.lower() in job_description.lower():
                    keywords.append(term)
        
        # Add exact phrases from job description
        phrases = re.findall(r'"([^"]*)"|\b([\w\s]+)\b', job_description)
        keywords.extend([p[0] or p[1] for p in phrases if len(p[0] or p[1]) > 5])
        
        return list(set(keywords))

    def _extract_key_phrases(self, text):
        phrases = []
        phrase_patterns = [
            r"customer success",
            r"customer experience",
            r"artificial intelligence",
            r"machine learning",
            r"data analytics",
            r"voice of customer",
            r"first call resolution"
        ]
        
        for pattern in phrase_patterns:
            if re.search(pattern, text.lower()):
                phrases.append(pattern)
        
        return phrases

    def _prioritize_cs_terms(self, keywords):
        # Move CS/CX specific terms to the front of the list
        cs_specific = []
        others = []
        
        for keyword in keywords:
            if any(pattern.lower() in keyword.lower() for pattern in self.cs_patterns):
                cs_specific.append(keyword)
            else:
                others.append(keyword)
                
        return cs_specific + others

    def extract_skills(self, job_description):
        # Enhanced skill extraction
        skill_patterns = [
            r"proficient in (.*?)\.",
            r"experience with (.*?)\.",
            r"knowledge of (.*?)\.",
            r"skills: (.*?)\.",
            r"requirements: (.*?)\.",
            r"expertise in (.*?)\.",
            r"familiarity with (.*?)\.",
            r"demonstrated ability to (.*?)\."
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            skills.extend(matches)
        
        # Add tool-specific skills
        for tool in self.cs_tools:
            if tool.lower() in job_description.lower():
                skills.append(tool)
        
        return list(set(skills))

    def get_cs_patterns(self):
        return self.cs_patterns

    def get_cs_tools(self):
        return self.cs_tools

    def get_cs_metrics(self):
        return self.cs_metrics 

class JobDataManager:
    def __init__(self):
        self.job_data = self._load_job_data()
        
    def _load_job_data(self):
        job_data = []
        for file in ['Generated_Job_Descriptions.csv', 'Generated_Job_Descriptions (1).csv', 'Generated_Job_Descriptions (3).csv']:
            df = pd.read_csv(file)
            job_data.extend(df.to_dict('records'))
        return job_data
        
    def get_common_skills(self):
        skills = []
        for job in self.job_data:
            skills.extend(job['Skills Required'].split(','))
        return list(set([s.strip() for s in skills])) 