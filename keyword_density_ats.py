import os
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai_client import client

# Handle optional dependencies with try/except
try:
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI and/or python-dotenv not available. Some features will be disabled.")

# Debug print at start of file
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

class ResumeOptimizer:
    def __init__(self):
        self.vectorizer = self._train_vectorizer()

    def _train_vectorizer(self):
        try:
            # Load training data from job descriptions
            training_texts = []
            csv_files = [
                'Generated_Job_Descriptions.csv',
                'Generated_Job_Descriptions (1).csv',
                'Generated_Job_Descriptions (3).csv'
            ]
            
            print("Looking for files:", csv_files)  # Debug print
            
            for csv_file in csv_files:
                file_path = os.path.join(os.getcwd(), csv_file)
                print(f"Trying to load: {file_path}")  # Debug print
                try:
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        print(f"Successfully loaded {len(df)} records from {csv_file}")
                        training_texts.extend(df['Description'].tolist())
                        training_texts.extend(df['Key Responsibilities'].tolist())
                        training_texts.extend(df['Skills Required'].tolist())
                    else:
                        print(f"File not found: {file_path}")
                except Exception as e:
                    print(f"Error loading {csv_file}: {str(e)}")

            # Clean and prepare texts
            training_texts = [text for text in training_texts if isinstance(text, str)]
            print(f"Total training texts: {len(training_texts)}")  # Debug print
            training_texts = [self.preprocess_text(text) for text in training_texts]

            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words='english'
            )
            vectorizer.fit(training_texts)
            return vectorizer
        except Exception as e:
            print(f"Error training vectorizer: {str(e)}")
            return None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'(\d+)(\s*-\s*\d+|\+)', r'\1plus', text)
        text = re.sub(r'(\d+)%', r'\1percent', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def calculate_match(self, job_desc, resume):
        if not self.vectorizer:
            return 0.0, set(), set()
            
        job_text = self.preprocess_text(job_desc)
        resume_text = self.preprocess_text(resume)
        
        vectors = self.vectorizer.transform([job_text, resume_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        terms = self.vectorizer.get_feature_names_out()
        job_terms = set(terms[vectors[0].nonzero()[1]])
        resume_terms = set(terms[vectors[1].nonzero()[1]])
        
        matches = job_terms & resume_terms
        missing = job_terms - resume_terms
        
        return round(similarity * 100, 2), matches, missing

    def optimize_resume(self, resume, job_desc):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert resume optimizer. Enhance the resume to match the job description while maintaining professional integrity."},
                    {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume:\n{resume}\n\nOptimize the resume to improve ATS match and highlight relevant skills and experiences."}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Optimization error: {str(e)}"

    def analyze_job_requirements(self):
        try:
            analysis_results = {
                'skill_weights': {},
                'experience_levels': {},
                'tools': {},
                'responsibilities': {}
            }
            
            skill_categories = {
                'technical': [
                    'ai', 'machine learning', 'analytics', 'data visualization', 
                    'automation', 'predictive analytics', 'crm', 'salesforce', 
                    'zendesk'
                ],
                'leadership': [
                    'lead', 'manage', 'direct', 'strategic', 'develop', 
                    'mentor', 'coach', 'oversee', 'drive'
                ],
                'customer_service': [
                    'customer success', 'customer experience', 'cx strategy',
                    'customer satisfaction', 'customer journey', 'voice of customer'
                ],
                'operations': [
                    'operational efficiency', 'process improvement', 'workflow',
                    'kpi management', 'metrics', 'optimization'
                ]
            }
            
            for csv_file in ['Generated_Job_Descriptions.csv', 'Generated_Job_Descriptions (1).csv', 'Generated_Job_Descriptions (3).csv']:
                df = pd.read_csv(csv_file)
                
                # Combine text for analysis
                df['all_text'] = df['Description'] + ' ' + df['Key Responsibilities'] + ' ' + df['Skills Required']
                
                # Analyze skill category frequencies
                total_jobs = len(df)
                category_counts = {cat: 0 for cat in skill_categories}
                
                for idx, row in df.iterrows():
                    text = row['all_text'].lower()
                    
                    # Count category occurrences
                    for category, terms in skill_categories.items():
                        if any(term in text for term in terms):
                            category_counts[category] += 1
                    
                    # Extract experience requirements
                    exp_match = re.search(r'(\d+)\+?\s*years?', str(row['Preferred Experience']))
                    if exp_match:
                        years = int(exp_match.group(1))
                        analysis_results['experience_levels'][years] = analysis_results['experience_levels'].get(years, 0) + 1
                    
                    # Count tool mentions
                    tools = ['Salesforce', 'Zendesk', 'Power BI', 'Tableau', 'SQL', 'Python']
                    for tool in tools:
                        if tool.lower() in text:
                            analysis_results['tools'][tool] = analysis_results['tools'].get(tool, 0) + 1
                
                # Calculate category weights
                category_weights = {cat: count/total_jobs for cat, count in category_counts.items()}
                
                # Update or initialize skill weights
                for cat, weight in category_weights.items():
                    if cat in analysis_results['skill_weights']:
                        analysis_results['skill_weights'][cat] = (analysis_results['skill_weights'][cat] + weight) / 2
                    else:
                        analysis_results['skill_weights'][cat] = weight
                
                print(f"\nAnalysis of {csv_file}:")
                print(f"Category weights: {category_weights}")
                print(f"Experience levels: {dict(sorted(analysis_results['experience_levels'].items()))}")
                print(f"Tool requirements: {dict(sorted(analysis_results['tools'].items(), key=lambda x: x[1], reverse=True))}")
            
            return analysis_results
            
        except Exception as e:
            print(f"Error analyzing job requirements: {str(e)}")
            return None

    def calibrate_weights(self):
        """Analyze job descriptions to determine accurate category weights"""
        try:
            total_weights = {
                'technical': 0,
                'leadership': 0,
                'customer_service': 0,
                'operations': 0
            }
            job_count = 0
            
            for csv_file in ['Generated_Job_Descriptions.csv', 'Generated_Job_Descriptions (1).csv', 'Generated_Job_Descriptions (3).csv']:
                df = pd.read_csv(csv_file)
                job_count += len(df)
                
                for _, row in df.iterrows():
                    text = ' '.join([
                        str(row['Description']), 
                        str(row['Key Responsibilities']), 
                        str(row['Skills Required'])
                    ]).lower()
                    
                    # Count occurrences in each category
                    if any(term in text for term in ['ai', 'machine learning', 'analytics', 'data']):
                        total_weights['technical'] += 1
                    if any(term in text for term in ['lead', 'manage', 'director']):
                        total_weights['leadership'] += 1
                    if any(term in text for term in ['customer', 'cx', 'service']):
                        total_weights['customer_service'] += 1
                    if any(term in text for term in ['operations', 'process', 'efficiency']):
                        total_weights['operations'] += 1
            
            # Calculate percentages
            weights = {k: v/job_count for k, v in total_weights.items()}
            print("Calibrated weights:", weights)
            return weights
            
        except Exception as e:
            print(f"Error calibrating weights: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    st.title("AI Resume Optimizer")
    
    optimizer = ResumeOptimizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        resume = st.text_area("Current Resume:", height=400)
    with col2:
        job_desc = st.text_area("Job Description:", height=400)
        
    if st.button("Optimize Resume"):
        if not resume.strip() or not job_desc.strip():
            st.warning("Please provide both resume and job description.")
            return
            
        # Calculate initial match
        initial_score, matches, missing = optimizer.calculate_match(job_desc, resume)
        
        # Optimize resume
        optimized_resume = optimizer.optimize_resume(resume, job_desc)
        
        # Calculate new match score
        new_score, new_matches, new_missing = optimizer.calculate_match(job_desc, optimized_resume)
        
        # Display results
        st.subheader("Resume Optimization Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Initial ATS Match", f"{initial_score}%")
        with col2:
            st.metric("Optimized ATS Match", f"{new_score}%")
        
        with st.expander("Original Resume"):
            st.text(resume)
        
        with st.expander("Optimized Resume"):
            st.text(optimized_resume)
        
        with st.expander("Keyword Analysis"):
            st.write("Initial Matching Keywords:")
            st.write(", ".join(sorted(list(matches))[:15]) or "No matching keywords")
            
            st.write("\nOptimized Matching Keywords:")
            st.write(", ".join(sorted(list(new_matches))[:15]) or "No matching keywords")
            
            if new_missing:
                st.write("\nRemaining Suggested Keywords:")
                st.write(", ".join(sorted(list(new_missing))[:10]))

if __name__ == "__main__":
    main()
