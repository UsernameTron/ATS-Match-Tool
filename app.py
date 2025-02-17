import streamlit as st
from analyzer import EnhancedAnalyzer
from optimizer import OptimizationEngine
from data_manager import DataManager
import pandas as pd
from io import StringIO
import logging
import sys
import altair as alt
import openai
from openai_client import client
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import sklearn
    import nltk
    import numpy as np
    logger.info("All required packages loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import required package: {str(e)}")
    st.error("Missing required packages. Please install requirements.txt")

def get_stored_resume():
    return """Director of Customer Experience & AI Strategist
Dallas-Fort Worth Metroplex
Phone: 682-500-5159
Email: cpeteconnor@gmail.com
LinkedIn: linkedin.com/in/cpeteconnor

Summary
Accomplished Customer Experience (CX) leader with expertise in leveraging Artificial Intelligence (AI), Machine Learning (ML), and Data Analytics to optimize customer experiences and operational performance. Proven track record in driving significant improvements in key performance metrics, including:
93% First-Call Resolution Rate
70% Increase in Referral-to-Appointment Conversions
33% Improvement in Service Delivery Speed
25% Reduction in Operational Costs
20% Boost in Net Promoter Score (NPS)
20% Increase in Customer Satisfaction (CSAT)

Expert in designing and implementing Voice of the Customer (VoC) systems, advanced SaaS & CRM platforms, and data-driven decision-making frameworks. Recognized for strategic leadership, cross-functional collaboration, and a commitment to continuous improvement.

Core Competencies
AI & Machine Learning Applications
o Predictive Analytics
o Natural Language Processing (NLP)
o AI-Powered Tools Development

Customer Feedback Systems (VoC)
o Real-Time Sentiment Analysis
o VoC Program Design
o Customer Journey Mapping

Advanced SaaS & CRM Platforms
o Zendesk
o Totango
o RingCentral
o Salesforce

Data Analytics & Visualization
o Microsoft 365 & Excel
o Google Sheets
o Power BI
o SQL & Python

Cross-Functional Collaboration
o Sales, Marketing, & Product Alignment
o Team Leadership & Mentorship
o Stakeholder Engagement

KPI Management & Reporting
o Performance Monitoring Tools
o Dashboard Creation
o Strategic Planning & Execution

Process Optimization
o Standard Operating Procedures (SOPs)
o Workflow Automation
o Operational Efficiency

Strategic Leadership
o Customer Experience Roadmaps
o Retention Strategies
o Digital Transformation Initiatives

[... rest of resume content ...]"""

# Add caching for OpenAI responses
@lru_cache(maxsize=32)
def get_openai_analysis(job_description: str) -> str:
    return client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """Analyze this job description and provide:
            1. Required Skills (with importance level)
            2. Experience Requirements
            3. Technical Requirements
            4. Leadership Requirements
            
            Format as a clear markdown list."""},
            {"role": "user", "content": f"Job Description:\n{job_description}"}
        ],
        temperature=0.1
    )

def main():
    try:
        logger.info("Starting application...")
        st.set_page_config(
            page_title="ATS Resume Optimizer for CS/CX Roles",
            page_icon="📄",
            layout="wide"
        )

        st.title("ATS Resume Optimizer for Customer Success Roles")
        st.subheader("Optimize your resume for ATS systems with CS/CX domain expertise")

        # Initialize components
        logger.info("Initializing components...")
        data_manager = DataManager()
        analyzer = EnhancedAnalyzer(data_manager)
        optimizer = OptimizationEngine(data_manager)

        # Use stored resume
        logger.info("Loading stored resume...")
        resume_text = get_stored_resume()
        resume_file = StringIO(resume_text)
        resume_file.name = "stored_resume.txt"

        # Initialize session state
        if 'analyzed' not in st.session_state:
            st.session_state.analyzed = False
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""

        # Job description input
        job_description = st.text_area("Paste the Job Description", height=200)

        if st.button("Analyze Resume"):
            # Check if we need to rerun analysis
            if not st.session_state.analyzed or job_description != st.session_state.job_description:
                logger.info("Starting new analysis...")
                try:
                    analysis = analyzer.analyze(resume_file, job_description)
                    st.session_state.analysis_result = analysis
                    st.session_state.analyzed = True
                    st.session_state.job_description = job_description
                    display_results(analysis, job_description)
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
            else:
                logger.info("Using cached analysis...")
                display_results(st.session_state.analysis_result, job_description)

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

def display_results(original, job_description):
    try:
        # Use cached OpenAI analysis
        if 'requirements_analysis' not in st.session_state:
            requirements_response = get_openai_analysis(job_description)
            st.session_state.requirements_analysis = requirements_response.choices[0].message.content
        
        # Display job requirements first
        st.subheader("Job Requirements Analysis")
        st.markdown(st.session_state.requirements_analysis)
        
        # Display match scores in columns
        scores_col1, scores_col2 = st.columns(2)
        
        with scores_col1:
            st.subheader("📊 Overall Scores")
            st.metric("ATS Match Score", f"{original.match_score:.1f}%")
            st.metric("Keyword Density", f"{original.keyword_density:.1f}%")
            st.metric("Domain Alignment", f"{original.domain_alignment:.1f}%")
            
        with scores_col2:
            st.subheader("🎯 Category Scores")
            for category in ['technical', 'leadership', 'customer_service', 'operations']:
                score = getattr(original, f"{category}_score")
                st.metric(category.replace('_', ' ').title(), f"{score:.1f}%")
        
        # Show improvement suggestions if any
        if original.improvement_areas:
            st.subheader("💡 Suggested Improvements")
            for area in original.improvement_areas:
                st.markdown(f"- {area}")
            
        # Add optimization section
        st.divider()
        st.subheader("📝 Optimized Resume")
        
        # Get optimized version using OpenAI
        optimized_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert resume optimizer. 
                Enhance this resume to better match the job description while:
                1. Maintaining authenticity of experience
                2. Targeting 90% ATS match score
                3. Keeping keyword density below 4.5%
                4. Preserving the original format
                
                Return only the optimized resume text."""},
                {"role": "user", "content": f"Job Description:\n{job_description}\n\nOriginal Resume:\n{original.text_content}"}
            ],
            temperature=0.7
        )
        
        # Show optimized resume in a copyable text area
        optimized_text = optimized_response.choices[0].message.content
        st.text_area("Optimized Resume (Copy from here)", optimized_text, height=400)
        
        # Show optimization improvements
        st.markdown("### ✨ Optimization Changes")
        st.markdown("- Adjusted keyword density to optimal level")
        st.markdown("- Enhanced role descriptions to better match requirements")
        st.markdown("- Aligned skills section with job requirements")
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        st.error("Error analyzing requirements. Please try again.")

if __name__ == "__main__":
    main() 