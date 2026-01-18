# core/computerscience/pipeline.py - Computer Science Pipeline with Qwen Model Integration

import asyncio
import time
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
import logging
import asyncio

from .state import ExecutionState, AnalysisType, CSDomain, TheoreticalFramework
from .loaders import _load_qwen_model, get_reward_model, generate_codeqwen_response, _fallback_cs_response
from core.analytics import (
    run_code_analysis,
    run_performance_analysis,
    run_security_analysis,
    run_algorithm_analysis,
    run_shap_analysis,
    run_lime_analysis,
    run_bayesian_optimization,
    run_causal_analysis
)
import os

logger = logging.getLogger("cs.pipeline")

class ComputerSciencePipeline:
    """
    Pipeline for processing computer science queries with support for both
    theoretical and practical aspects of computer science.
    """
    
    def __init__(self):
        self.state: Optional[ExecutionState] = None
        self.reward_model = None
        self.logger = logging.getLogger("cs.pipeline")
    
    async def initialize(self):
        """Initialize the pipeline and load required models"""
        try:
            self.reward_model = await get_reward_model()
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    async def process_query(
        self, 
        query: str, 
        session_id: str = "",
        analysis_type: Union[AnalysisType, str] = AnalysisType.HYBRID,
        domains: Optional[List[Union[CSDomain, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a computer science query with support for both
        theoretical and practical aspects.
        
        Args:
            query: The input query or problem statement
            session_id: Optional session identifier
            analysis_type: Type of analysis to perform (theoretical, practical, or hybrid)
            domains: List of computer science domains this query relates to
            
        Returns:
            Dictionary containing the response and analysis metadata
        """
        # Initialize state
        self.state = ExecutionState(
            query=query,
            session_id=session_id or f"cs_{int(datetime.now().timestamp())}",
            analysis_type=analysis_type
        )
        
        try:
            # Add specified domains if any
            if domains:
                for domain in domains:
                    self.state.add_domain(domain)
            
            # Step 1: Analyze query if domains not specified
            if not self.state.domains:
                await self._analyze_query_domains()
            
            self.logger.info(
                f"Processing {self.state.analysis_type.value} query "
                f"in domains: {[d.value for d in self.state.domains]}"
            )
            
            # Step 2: Load appropriate model configuration
            model_config = await _load_qwen_model(self.state.analysis_type)
            
            # Step 3: Process the query based on analysis type
            if self.state.analysis_type == AnalysisType.THEORETICAL:
                await self._process_theoretical_analysis(model_config)
            elif self.state.analysis_type == AnalysisType.PRACTICAL:
                await self._process_practical_analysis(model_config)
            else:  # HYBRID
                await self._process_hybrid_analysis(model_config)
            
            # Step 4: Generate final response
            response = await self._generate_response()
            
            # Step 5: Log interaction for RLHF
            await self._log_interaction(response)
            
            return {
                "response": response,
                "analysis_type": self.state.analysis_type.value,
                "domains": [d.value for d in self.state.domains],
                "confidence": self.state.confidence,
                "theoretical_frameworks": [
                    {"name": f.name, "description": f.description}
                    for f in self.state.theoretical_frameworks
                ],
                "code_artifacts": [
                    {"name": a["name"], "language": a["language"]}
                    for a in self.state.code_artifacts
                ]
            }
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            self.state.final_response = _fallback_cs_response(self.state.query)
            return {"error": str(e)}
    
    async def _generate_response(self) -> str:
        """
        Generate a comprehensive response based on the analysis performed.
        In a real implementation, this would use the Qwen model to generate
        a natural language response based on the state.
        """
        response_parts = []
        
        # Add introduction
        domain_names = ", ".join(d.value.replace("_", " ").title() 
                               for d in self.state.domains)
        response_parts.append(
            f"# Analysis of '{self.state.query}'\n"
            f"**Domains**: {domain_names}\n"
            f"**Analysis Type**: {self.state.analysis_type.value.title()}\n"
        )
        
        # Add theoretical components if present
        if hasattr(self.state, 'theoretical_frameworks') and self.state.theoretical_frameworks:
            response_parts.append("## Theoretical Framework\n")
            for framework in self.state.theoretical_frameworks:
                framework_text = f"### {framework.name}\n{framework.description}\n"
                if hasattr(framework, 'complexity_class') and framework.complexity_class:
                    framework_text += f"**Complexity Classes**: {framework.complexity_class}\n"
                response_parts.append(framework_text)
        
        # Add practical components if present
        if hasattr(self.state, 'code_artifacts') and self.state.code_artifacts:
            response_parts.append("## Implementation\n")
            for artifact in self.state.code_artifacts:
                if isinstance(artifact, dict) and 'name' in artifact and 'language' in artifact and 'content' in artifact:
                    code_block = (
                        f"### {artifact['name'].replace('_', ' ').title()} ({artifact['language']})\n"
                        f"```{artifact['language']}\n"
                        f"{artifact['content']}\n"
                        f"```\n"
                    )
                    response_parts.append(code_block)
        
        # Add parameters if present
        if hasattr(self.state, 'parameters') and self.state.parameters:
            response_parts.append("## Parameters\n")
            for key, value in self.state.parameters.items():
                response_parts.append(f"- **{key.replace('_', ' ').title()}**: `{value}`\n")
        
        # Add any implications or conclusions
        if hasattr(self.state, 'implications') and self.state.implications:
            response_parts.append("## Analysis and Implications\n")
            for implication in self.state.implications:
                response_parts.append(f"- {implication}\n")
        
        return "\n".join(response_parts)
    
    def _format_analytics_summary(self, analytics: dict) -> str:
        """Format analytics results with XAI insights for the response"""
        if not analytics:
            return "No detailed analysis available."

        summary = []
        
        # XAI Insights Section
        xai_insights = []
        
        # SHAP Analysis
        if 'shap_analysis' in analytics and 'feature_importance' in analytics['shap_analysis']:
            shap_imp = analytics['shap_analysis']['feature_importance']
            if shap_imp:
                top_features = sorted(shap_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                xai_insights.append("Feature Importance (SHAP):")
                for feat, imp in top_features:
                    xai_insights.append(f"- {feat}: {imp:.3f}")
        
        # LIME Analysis
        if 'lime_analysis' in analytics and 'explanations' in analytics['lime_analysis']:
            lime_exp = analytics['lime_analysis']['explanations']
            if lime_exp and isinstance(lime_exp, dict):
                top_explanations = sorted(lime_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                xai_insights.append("\nLocal Explanations (LIME):")
                for feat, weight in top_explanations:
                    xai_insights.append(f"- {feat}: {weight:.3f}")
        
        # Causal Analysis
        if 'causal_analysis' in analytics:
            causal = analytics['causal_analysis']
            if 'ate' in causal:
                ate = causal['ate']
                ci = causal.get('confidence_interval', [0, 0])
                p_value = causal.get('p_value', 0.05)
                
                causal_text = [
                    f"\n🔗 Causal Analysis:",
                    f"- Average Treatment Effect (ATE): {ate:.3f}",
                    f"- 95% Confidence Interval: [{ci[0]:.3f}, {ci[1]:.3f}]",
                    f"- p-value: {p_value:.4f}"
                ]
                
                if 'confounder_effects' in causal:
                    causal_text.append("\nConfounder Effects:")
                    for conf, effect in causal['confounder_effects'].items():
                        causal_text.append(f"- {conf}: {effect:.3f}")
                
                xai_insights.append("\n".join(causal_text))
        
        # Bayesian Optimization
        if 'bayesian_optimization' in analytics:
            bayes = analytics['bayesian_optimization']
            if 'suggested_improvement' in bayes:
                xai_insights.append(f"\nOptimization Suggestion: {bayes['suggested_improvement']}")
            if 'improvement_pct' in bayes:
                xai_insights.append(f"Expected Improvement: {bayes['improvement_pct']:.1f}%")
        
        if xai_insights:
            summary.append("🔍 Explainable AI Insights:" + "\n" + "\n".join(xai_insights))
        
        # Traditional Analysis Section
        summary.append("\n📊 Technical Analysis:")
        
        # Code Analysis
        if 'code_analysis' in analytics:
            issues = analytics['code_analysis'].get('issues', [])
            if issues:
                critical_issues = [i for i in issues if i.get('severity') == 'critical'][:3]
                if critical_issues:
                    summary.append("\n🔧 Critical Code Issues:")
                    for issue in critical_issues:
                        summary.append(f"- {issue.get('description')}")
                        if 'line' in issue:
                            summary[-1] += f" (line {issue['line']})"

        # Performance Analysis
        if 'performance_analysis' in analytics:
            perf = analytics['performance_analysis']
            perf_summary = []
            if 'bottleneck' in perf:
                perf_summary.append(f"Bottleneck: {perf['bottleneck']}")
            if 'complexity' in perf:
                perf_summary.append(f"Complexity: {perf['complexity']}")
            if perf_summary:
                summary.append("\n⚡ Performance Analysis:" + "\n- " + "\n- ".join(perf_summary))

        # Security Analysis
        if 'security_analysis' in analytics:
            vulns = analytics['security_analysis'].get('vulnerabilities', [])
            if vulns:
                summary.append("\n🔒 Security Findings:")
                for vuln in vulns[:3]:
                    sev = vuln.get('severity', 'Medium').title()
                    summary.append(f"- {sev} severity: {vuln.get('description')}")

        return "\n".join(summary) if summary else "No significant findings in analysis."
    
    def _calculate_confidence(self):
        """Calculate confidence score based on XAI and analysis results"""
        # Base confidence
        confidence = 0.4
        
        # Get analysis results
        analysis = getattr(self.state, 'structured_analysis', {})
        
        # Boost for XAI components (up to 0.25)
        xai_components = ['shap_analysis', 'lime_analysis', 'causal_analysis', 'bayesian_optimization']
        xai_present = [c for c in xai_components if c in analysis]
        if xai_present:
            confidence += min(0.25, 0.05 * len(xai_present))
        
        # Boost for agreement between XAI methods (up to 0.15)
        if len(xai_present) >= 2:
            # Check if SHAP and LIME agree on top features
            if 'shap_analysis' in analysis and 'lime_analysis' in analysis:
                shap_top = sorted(analysis['shap_analysis'].get('feature_importance', {}).items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:3]
                lime_top = sorted(analysis['lime_analysis'].get('explanations', {}).items(),
                                key=lambda x: abs(x[1]), reverse=True)[:3]
                
                # Check for overlap in top features
                shap_features = {f[0] for f in shap_top}
                lime_features = {f[0] for f in lime_top}
                overlap = len(shap_features.intersection(lime_features))
                
                if overlap > 0:
                    confidence += min(0.15, 0.05 * overlap)
        
        # Boost for causal evidence (up to 0.15)
        if 'causal_analysis' in analysis:
            causal = analysis['causal_analysis']
            if 'ate' in causal and 'p_value' in causal:
                ate = abs(causal['ate'])
                p_value = causal['p_value']
                
                # Only boost if effect is statistically significant
                if p_value < 0.05 and ate > 0.1:
                    # Base boost on effect size and precision (inverse of CI width)
                    ci = causal.get('confidence_interval', [0, 0])
                    ci_width = ci[1] - ci[0] if len(ci) == 2 else 1.0
                    precision = 1.0 / (ci_width + 1e-6)  # Avoid division by zero
                    
                    # Calculate boost considering both effect size and precision
                    effect_boost = min(0.1, ate * 0.5)
                    precision_boost = min(0.05, precision * 0.02)
                    confidence += effect_boost + precision_boost
        
        # Boost for Bayesian optimization (up to 0.1)
        if 'bayesian_optimization' in analysis and 'improvement_pct' in analysis['bayesian_optimization']:
            improvement = analysis['bayesian_optimization']['improvement_pct']
            confidence += min(0.1, improvement * 0.01)  # Scale with improvement percentage

        # Increase confidence based on successful analyses (up to 0.2)
        analysis_count = len(analysis)
        if analysis_count > 0:
            confidence += min(0.2, 0.03 * analysis_count)

        # Increase confidence based on evidence (up to 0.15)
        evidence_count = len(getattr(self.state, 'retrieved_evidence', []))
        if evidence_count > 0:
            confidence += min(0.15, 0.03 * evidence_count)

        # Penalize for errors (up to -0.3)
        error_count = len(getattr(self.state, 'errors', []))
        if error_count > 0:
            confidence -= min(0.3, 0.05 * error_count)

        # Ensure confidence is within [0.1, 0.99] range
        self.state.confidence = max(0.1, min(0.99, confidence))

        return self.state.confidence
    
    async def _emergency_fallback(self):
        """Fallback response generation when pipeline fails"""
        try:
            response = await _fallback_cs_response(self.state.query)
            if hasattr(self.state, 'final_response'):
                self.state.final_response = response
            if hasattr(self.state, 'confidence'):
                self.state.confidence = 0.3
        except Exception as e:
            if hasattr(self.state, 'final_response'):
                self.state.final_response = "I apologize, but I encountered an error processing your request. Please try again later."
            if hasattr(self.state, 'confidence'):
                self.state.confidence = 0.1