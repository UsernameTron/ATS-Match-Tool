from dataclasses import dataclass
from typing import List, Dict
import re

@dataclass
class OptimizationResult:
    match_score: float
    keyword_density: float
    domain_alignment: float
    suggestions: List[Dict[str, str]]
    section_recommendations: Dict[str, List[str]]
    priority_improvements: List[str]

class OptimizationEngine:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def optimize(self, analysis_result):
        suggestions = []
        
        # Generate section-specific recommendations
        section_recommendations = self._generate_section_recommendations(analysis_result)
        
        # Generate prioritized improvements
        priority_improvements = self._generate_priority_improvements(analysis_result)
        
        # Generate detailed suggestions
        if analysis_result.keyword_density < 4.5:
            suggestions.extend(self._generate_keyword_suggestions(analysis_result))
            
        if analysis_result.match_score < 90:
            suggestions.extend(self._generate_skill_suggestions(analysis_result))
            
        if analysis_result.domain_alignment < 85:
            suggestions.extend(self._generate_domain_suggestions(analysis_result))
            
        # Add tool-specific suggestions
        suggestions.extend(self._generate_tool_suggestions(analysis_result))
        
        # Add metric-focused suggestions
        suggestions.extend(self._generate_metric_suggestions(analysis_result))
            
        return OptimizationResult(
            match_score=self._estimate_optimized_score(analysis_result),
            keyword_density=self._estimate_optimized_density(analysis_result),
            domain_alignment=self._estimate_optimized_alignment(analysis_result),
            suggestions=suggestions,
            section_recommendations=section_recommendations,
            priority_improvements=priority_improvements
        )

    def _generate_keyword_suggestions(self, analysis):
        suggestions = []
        for keyword in analysis.missing_keywords:
            context = self._determine_keyword_context(keyword)
            suggestions.append({
                'type': 'keyword',
                'item': keyword,
                'suggestion': f"Include the term '{keyword}' in your {context}",
                'priority': 'high' if keyword in self.data_manager.get_cs_patterns() else 'medium'
            })
        return suggestions

    def _determine_keyword_context(self, keyword):
        if any(pattern.lower() in keyword.lower() for pattern in self.data_manager.get_cs_patterns()):
            return "experience section or summary"
        elif keyword in self.data_manager.get_cs_tools():
            return "technical skills section"
        else:
            return "relevant sections"

    def _generate_skill_suggestions(self, analysis):
        suggestions = []
        for skill in analysis.skill_gaps:
            suggestions.append({
                'type': 'skill',
                'item': skill,
                'suggestion': f"Highlight experience with '{skill}' through specific achievements",
                'priority': 'high'
            })
        return suggestions

    def _generate_domain_suggestions(self, analysis):
        suggestions = []
        cs_patterns = self.data_manager.get_cs_patterns()
        
        domain_suggestions = [
            {
                'type': 'domain',
                'item': 'Customer Success Metrics',
                'suggestion': "Quantify achievements with CS/CX metrics (NPS, CSAT, churn rate)",
                'priority': 'high'
            },
            {
                'type': 'domain',
                'item': 'Customer Journey',
                'suggestion': "Describe experience with customer journey mapping and optimization",
                'priority': 'medium'
            },
            {
                'type': 'domain',
                'item': 'Stakeholder Management',
                'suggestion': "Emphasize cross-functional collaboration and stakeholder management",
                'priority': 'medium'
            }
        ]
        suggestions.extend(domain_suggestions)
        return suggestions

    def _generate_tool_suggestions(self, analysis):
        suggestions = []
        missing_tools = [tool for tool, present in analysis.tool_matches.items() if not present]
        
        if missing_tools:
            key_tools = self._prioritize_tools(missing_tools)
            for tool in key_tools[:3]:  # Suggest top 3 missing tools
                suggestions.append({
                    'type': 'tool',
                    'item': tool,
                    'suggestion': f"Add experience with {tool} if applicable",
                    'priority': 'medium'
                })
        return suggestions

    def _generate_metric_suggestions(self, analysis):
        suggestions = []
        if len(analysis.metrics_found) < 3:
            suggestions.append({
                'type': 'metric',
                'item': 'Quantifiable Achievements',
                'suggestion': "Add more quantifiable achievements (%, improvements, metrics)",
                'priority': 'high'
            })
        return suggestions

    def _generate_section_recommendations(self, analysis):
        return {
            'Summary': [
                "Lead with CS/CX focus and key achievements",
                "Highlight years of experience and industry impact"
            ],
            'Experience': [
                "Quantify achievements with specific metrics",
                "Emphasize customer-centric initiatives",
                "Showcase cross-functional collaboration"
            ],
            'Skills': [
                "List relevant CS/CX tools and technologies",
                "Include both technical and soft skills",
                "Highlight certifications and methodologies"
            ]
        }

    def _generate_priority_improvements(self, analysis):
        priorities = []
        
        if analysis.match_score < 85:
            priorities.append("Enhance keyword alignment with job requirements")
        if len(analysis.metrics_found) < 3:
            priorities.append("Add more quantifiable achievements")
        if analysis.domain_alignment < 80:
            priorities.append("Strengthen CS/CX specific experience")
            
        return priorities

    def _prioritize_tools(self, tools):
        # Prioritize tools based on industry relevance
        tool_priority = {
            "Salesforce": 10,
            "Gainsight": 9,
            "Zendesk": 8,
            "Totango": 7,
            "HubSpot": 6
        }
        return sorted(tools, key=lambda x: tool_priority.get(x, 0), reverse=True)

    def _estimate_optimized_score(self, analysis):
        potential_improvement = min(15, 90 - analysis.match_score)
        return min(100, analysis.match_score + potential_improvement)

    def _estimate_optimized_density(self, analysis):
        return min(4.5, analysis.keyword_density + 1.0)

    def _estimate_optimized_alignment(self, analysis):
        return min(100, analysis.domain_alignment + 10) 