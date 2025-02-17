import PyPDF2
import docx
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import logging
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from keyword_density_ats import ResumeOptimizer
import os
from openai_client import client
from functools import lru_cache

# Set up NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    match_score: float
    keyword_density: float
    domain_alignment: float
    technical_score: float
    leadership_score: float
    customer_service_score: float
    operations_score: float
    missing_keywords: List[str]
    skill_gaps: List[str]
    text_content: str
    metrics_found: List[str]
    tool_matches: Dict[str, bool]
    improvement_areas: List[str]

class EnhancedAnalyzer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
        # Calibrate weights directly
        weights = self._calibrate_weights()
        if weights:
            self.criteria = {
                'technical': {
                    'terms': [
                        'ai', 'machine learning', 'predictive analytics', 'data visualization',
                        'automation', 'crm', 'salesforce', 'zendesk'
                    ],
                    'weight': weights['technical'],
                    'required_matches': 4
                },
                'leadership': {
                    'terms': [
                        'director', 'leadership', 'manage', 'lead', 'develop',
                        'mentor', 'coach', 'oversee'
                    ],
                    'weight': weights['leadership'],
                    'required_matches': 5
                },
                'customer_service': {
                    'terms': [
                        'customer success', 'customer experience', 'cx strategy',
                        'customer satisfaction', 'customer journey'
                    ],
                    'weight': weights['customer_service'],
                    'required_matches': 4
                },
                'operations': {
                    'terms': [
                        'operational efficiency', 'process improvement', 'workflow',
                        'kpi management', 'metrics'
                    ],
                    'weight': weights['operations'],
                    'required_matches': 3
                }
            }
        else:
            # Default weights if calibration fails
            self.criteria = {
                'technical': {
                    'terms': [
                        'ai', 'machine learning', 'predictive analytics', 'data visualization',
                        'automation', 'crm', 'salesforce', 'zendesk'
                    ],
                    'weight': 0.25,
                    'required_matches': 4
                },
                'leadership': {
                    'terms': [
                        'director', 'leadership', 'manage', 'lead', 'develop',
                        'mentor', 'coach', 'oversee'
                    ],
                    'weight': 0.30,
                    'required_matches': 5
                },
                'customer_service': {
                    'terms': [
                        'customer success', 'customer experience', 'cx strategy',
                        'customer satisfaction', 'customer journey'
                    ],
                    'weight': 0.25,
                    'required_matches': 4
                },
                'operations': {
                    'terms': [
                        'operational efficiency', 'process improvement', 'workflow',
                        'kpi management', 'metrics'
                    ],
                    'weight': 0.20,
                    'required_matches': 3
                }
            }
        
    def analyze(self, resume_file, job_description):
        # Extract text from resume
        resume_text = self._extract_text(resume_file)
        
        # Calculate core metrics
        match_score = self._calculate_match_score(resume_text, job_description)
        keyword_density = self._calculate_keyword_density(resume_text, job_description)
        domain_alignment = self._calculate_domain_alignment(resume_text)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(resume_text)
        
        # Detailed analysis
        missing_keywords = self._identify_missing_keywords(resume_text, job_description)
        skill_gaps = self._identify_skill_gaps(resume_text, job_description)
        metrics_found = self._extract_metrics(resume_text)
        tool_matches = self._analyze_tool_coverage(resume_text)
        improvement_areas = self._identify_improvement_areas(resume_text, job_description)
        
        return AnalysisResult(
            match_score=match_score,
            keyword_density=keyword_density,
            domain_alignment=domain_alignment,
            technical_score=category_scores['technical'],
            leadership_score=category_scores['leadership'],
            customer_service_score=category_scores['customer_service'],
            operations_score=category_scores['operations'],
            missing_keywords=missing_keywords,
            skill_gaps=skill_gaps,
            text_content=resume_text,
            metrics_found=metrics_found,
            tool_matches=tool_matches,
            improvement_areas=improvement_areas
        )

    def _extract_text(self, file):
        try:
            if isinstance(file, StringIO):
                return file.getvalue()
            
            text = ""
            if file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file.name.endswith('.docx'):
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            
            logger.info(f"Extracted text length: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

    @lru_cache(maxsize=32)
    def _calculate_match_score(self, resume_text, job_description):
        if not resume_text or not job_description:
            return 0.0
        
        try:
            # First, have GPT analyze the job description to identify key requirements
            requirements_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Analyze this job description and break down the key requirements into these categories:
                    1. Core Technical Skills
                    2. Domain Expertise
                    3. Leadership Requirements
                    4. Operational Skills
                    
                    Return only a JSON object with the categories and their specific requirements."""},
                    {"role": "user", "content": f"Job Description:\n{job_description}"}
                ],
                temperature=0.1
            )
            
            # Then score the resume against these specific requirements
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Score how well the resume matches these exact job requirements.
                    Consider only the specific skills and experience mentioned in the job description.
                    Provide a numerical score (0-100) based solely on matching the stated requirements."""},
                    {"role": "user", "content": f"Job Requirements:\n{requirements_response.choices[0].message.content}\n\nResume:\n{resume_text}"}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(''.join(filter(str.isdigit, score_text)))
                return score
            except ValueError:
                logger.error("Could not parse score from OpenAI response")
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating match score: {str(e)}")
            return 0.0

    def _get_similar_terms(self, keyword):
        # Dictionary of similar terms for common requirements
        similarity_dict = {
            'lead': ['manage', 'direct', 'oversee', 'supervise'],
            'develop': ['train', 'coach', 'mentor', 'grow'],
            'customer': ['client', 'member', 'user', 'patient'],
            'service': ['support', 'assistance', 'help', 'care'],
            'metrics': ['kpi', 'measurement', 'performance', 'analytics'],
            'process': ['procedure', 'workflow', 'operation', 'system']
        }
        
        for key, values in similarity_dict.items():
            if key in keyword.lower():
                return values
        return []

    def _calculate_keyword_density(self, resume_text, job_description):
        if not resume_text or not job_description:
            return 0.0
        
        keywords = self.data_manager.extract_keywords(job_description)
        if not keywords:
            return 0.0
        
        words = [w for w in resume_text.split() if w.strip()]
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        keyword_count = sum(resume_text.lower().count(keyword.lower()) for keyword in keywords)
        return (keyword_count / total_words * 100)  # Should naturally approach 6.5% for good density

    def _calculate_domain_alignment(self, resume_text):
        resume_lower = resume_text.lower()
        
        # Simplified domain criteria with tighter scoring
        domain_criteria = {
            'cx_expertise': {
                'terms': [
                    'customer experience', 'customer success', 'cx strategy',
                    'customer satisfaction', 'customer journey'
                ],
                'weight': 0.30,
                'required_matches': 3
            },
            'technical_skills': {
                'terms': [
                    'ai', 'machine learning', 'predictive analytics',
                    'crm', 'automation'
                ],
                'weight': 0.25,
                'required_matches': 2
            },
            'leadership': {
                'terms': [
                    'director', 'lead', 'manage', 'develop',
                    'strategic'
                ],
                'weight': 0.25,
                'required_matches': 3
            },
            'operations': {
                'terms': [
                    'operational', 'process improvement',
                    'kpi', 'metrics', 'workflow'
                ],
                'weight': 0.20,
                'required_matches': 2
            }
        }
        
        total_alignment = 0
        max_possible = 0
        
        for category, data in domain_criteria.items():
            matches = sum(1 for term in data['terms'] if term in resume_lower)
            category_score = min(1.0, matches / (data['required_matches'] * 1.2))  # 20% more matches needed
            total_alignment += category_score * data['weight']
            max_possible += data['weight']
        
        # Normalize to 85-point scale like match score
        return (total_alignment / max_possible) * 85

    def _extract_metrics(self, resume_text):
        metrics = []
        metric_patterns = self.data_manager.get_cs_metrics()
        
        for pattern in metric_patterns:
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                metrics.append(match.group(0))
                
        return metrics

    def _analyze_tool_coverage(self, resume_text):
        tool_matches = {}
        for tool in self.data_manager.get_cs_tools():
            tool_matches[tool] = tool.lower() in resume_text.lower()
        return tool_matches

    def _identify_improvement_areas(self, resume_text, job_description):
        areas = []
        
        # Check metric presence
        if len(self._extract_metrics(resume_text)) < 3:
            areas.append("Add more quantifiable achievements and metrics")
            
        # Check tool coverage
        tool_matches = self._analyze_tool_coverage(resume_text)
        if sum(tool_matches.values()) < len(tool_matches) * 0.3:
            areas.append("Include more relevant CS/CX tools and technologies")
            
        # Check keyword density
        if self._calculate_keyword_density(resume_text, job_description) < 4.5:
            areas.append("Increase relevant keyword density while maintaining readability")
            
        return areas

    def _identify_missing_keywords(self, resume_text, job_description):
        keywords = self.data_manager.extract_keywords(job_description)
        return [keyword for keyword in keywords 
                if keyword.lower() not in resume_text.lower()]

    def _identify_skill_gaps(self, resume_text, job_description):
        required_skills = self.data_manager.extract_skills(job_description)
        return [skill for skill in required_skills 
                if skill.lower() not in resume_text.lower()]

    def _calculate_category_scores(self, resume_text):
        resume_lower = resume_text.lower()
        scores = {}
        
        try:
            for category, data in self.criteria.items():
                # Count matching terms
                matches = sum(1 for term in data['terms'] if term in resume_lower)
                
                # Calculate base score (0-1 range)
                base_score = matches / data['required_matches']
                
                # Convert to percentage and cap at 100
                score = min(100.0, base_score * 100)
                
                # Store final score
                scores[category] = float(max(0, score))
                
                logger.info(f"{category}: {matches} matches, score: {scores[category]:.1f}%")
                
        except Exception as e:
            logger.error(f"Error calculating category scores: {str(e)}")
            scores = {
                'technical': 0.0,
                'leadership': 0.0,
                'customer_service': 0.0,
                'operations': 0.0
            }
        
        return scores

    def _calibrate_weights(self):
        """Return default weights for analysis"""
        return {
            'technical': 0.25,
            'leadership': 0.30,
            'customer_service': 0.25,
            'operations': 0.20
        }