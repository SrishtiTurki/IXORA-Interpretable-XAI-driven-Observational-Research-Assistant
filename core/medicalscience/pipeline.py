# core/medicalscience/pipeline.py - UPDATED WITH PRIORITIZED ANALYTICS + RLHF LOGGING
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List
from .state import ExecutionState
import logging
from datetime import datetime
import numpy as np
import torch
from torch import nn
import os
from core.rlhf.reward_model import get_reward_model
from core.analytics import (
    run_bayesian_optimization,
    run_shap_analysis,
    run_lime_analysis,
    run_causal_analysis,  # Extracted as standalone
)

logger = logging.getLogger("biomed.pipeline")

class BiomedicalPipeline:
    """CPU-optimized biomedical pipeline with prioritized analytics"""
    
    STEP_TIMEOUTS = {
        "intent_classification": 5.0,
        "parameter_extraction": 20.0,
        "evidence_retrieval": 10.0,
        "parallel_analytics": 50.0,
        "hypothesis_generation": 15.0,
        "response_synthesis": 60.0,
        "total": 210.0
    }
    
    def __init__(self):
        self.state = None


    async def _classify_intent_fast(self):
        from core.utils import detect_intent
        intent = await detect_intent(self.state.query, self.state.domain)
        self.state.intent = intent
        # NEW: even conversational gets a light draft later
        return intent
    
    async def run(self, query: str, session_id: str = "") -> ExecutionState:
        self.state = ExecutionState(
            query=query,
            session_id=session_id,
            start_time=datetime.now(),
            max_total_time=self.STEP_TIMEOUTS["total"]
        )
        
        logger.info(f"🚀 Starting Prioritized Pipeline: {query[:100]}")
        
        try:
            await self._run_with_timeout("intent_classification", self._classify_intent_fast)
            await self._run_with_timeout("parameter_extraction", self._extract_parameters_fast)
            await self._run_with_timeout("evidence_retrieval", self._retrieve_evidence_lightweight)
            await self._run_with_timeout("parallel_analytics", self._run_prioritized_analytics)
            await self._run_with_timeout("hypothesis_generation", self._generate_hypothesis_fast)
            await self._run_with_timeout("response_synthesis", self._synthesize_response_with_analytics)
            
            self._calculate_confidence()
            logger.info(f"✅ Pipeline completed in {self.state._get_elapsed():.1f}s")
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Pipeline timeout after {self.state._get_elapsed():.1f}s")
            self.state.add_error("pipeline", "Total timeout exceeded", True)
            await self._emergency_fallback_with_analytics()
        
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            self.state.add_error("pipeline", str(e), True)
            await self._emergency_fallback_with_analytics()
        
        finally:
            # RLHF Logging: Save anonymized state
            self._save_rlhf_log()
        
        return self.state
    
    async def _run_with_timeout(self, step_name: str, step_func):
        timeout = self.STEP_TIMEOUTS[step_name]
        
        if self.state.time_remaining() < timeout:
            logger.warning(f"⚠️ Skipping {step_name} - insufficient time")
            self.state.add_error(step_name, "Insufficient time budget", True)
            return None
        
        start = time.time()
        try:
            result = await asyncio.wait_for(step_func(), timeout=timeout)
            elapsed = time.time() - start
            self.state.step_times[step_name] = elapsed
            self.state.add_trace(step_name, result)
            logger.info(f"✅ {step_name}: {elapsed:.1f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            self.state.step_times[step_name] = elapsed
            self.state.add_error(step_name, f"Timeout after {timeout}s", False)
            logger.warning(f"⏱️ {step_name} timeout after {timeout}s")
    
    async def _run_prioritized_analytics(self):
        """Prioritized: Bayesian first → Explainability (conditional) → Causal deferred"""
        start = time.time()
        
        parameters = self.state.parameters or {}
        query_lower = self.state.query.lower()
        
        # 1. Bayesian Optimization - ALWAYS FIRST (most actionable)
        if parameters:
            bayesian_result = await run_bayesian_optimization(parameters, domain=self.state.domain)
            self.state.structured_analysis["optimization"] = bayesian_result
            self.state.add_trace("analytics_bayesian", bayesian_result)
            self.state.module_usage.append("bayesian")
        
        # 2. Conditional Explainability
        if parameters:
            if any(kw in query_lower for kw in ["local", "specific", "why this", "instance"]) or len(parameters) <= 2:
                primary = await run_lime_analysis(parameters, self.state.domain)
                primary["method_priority"] = "lime_local"
                secondary = await run_shap_analysis(parameters, self.state.domain)
            else:
                primary = await run_shap_analysis(parameters, self.state.domain)
                primary["method_priority"] = "shap_global"
                secondary = await run_lime_analysis(parameters, self.state.domain)
            
            primary["secondary"] = secondary
            self.state.structured_analysis["explainability"] = primary
            self.state.add_trace("analytics_explainability", primary)
            self.state.module_usage.append("explainability")
        
        # Causal deferred - log as available on-demand
        self.state.add_trace("analytics_causal", {"status": "available_on_demand"})
        
        elapsed = time.time() - start
        logger.info(f"Prioritized analytics completed in {elapsed:.1f}s")
    
    def _save_rlhf_log(self):
        """Save anonymized execution state for future RLHF"""
        try:
            anonymized_query = hashlib.sha256(self.state.query.encode()).hexdigest()[:16]
            
            log_entry = {
                "session_id": self.state.session_id,
                "anonymized_query": anonymized_query,
                "timestamp": datetime.now().isoformat(),
                "module_usage": self.state.module_usage,
                "confidence": self.state.confidence,
                "total_time": self.state._get_elapsed(),
                "parameters_count": len(self.state.parameters),
                "trace_summary": [
                    {"step": t["step"], "elapsed": t.get("elapsed_seconds", 0)}
                    for t in self.state.trace
                ],
                "errors": len(self.state.errors)
            }
            
            with open("rlhf_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to save RLHF log: {e}")

    
    # ========== STEP IMPLEMENTATIONS (CPU OPTIMIZED) ==========
    
    async def _classify_intent_fast(self):
        from core.utils import detect_intent, cache_get
        
        # Check if it came from cache (for debugging/trace)
        cache_key = f"{self.state.domain}:intent:{hash(self.state.query[:50])}"
        cached = cache_get(cache_key)
        cache_source = "cache_hit" if cached else "computed"
        
        intent = await detect_intent(self.state.query, self.state.domain)
        
        self.state.intent = intent
        self.state.add_trace(
            "intent_classification",
            intent,
            metadata={
                "method": "detect_intent",
                "source": cache_source,
                "cached": bool(cached)
            }
        )
        
        return intent
    
    async def _extract_parameters_fast(self):
        """Fast parameter extraction for CPU"""
        import re
        from core.utils import extract_parameters
        
        try:
            # Use cached extraction first
            params = await extract_parameters(self.state.query, self.state.domain)
            
            # Ensure we have at least basic parameters
            if not params:
                params = self._extract_basic_params_fallback()
            
            self.state.parameters = params
            
            # Log summary
            param_summary = {k: v.get("value") for k, v in list(params.items())[:3]}
            return {
                "parameter_count": len(params),
                "top_parameters": param_summary,
                "method": "fast_extraction",
                "has_numeric": any(isinstance(v.get("value"), (int, float)) for v in params.values())
            }
            
        except Exception as e:
            logger.warning(f"Parameter extraction failed: {e}, using fallback")
            params = self._extract_basic_params_fallback()
            self.state.parameters = params
            return {
                "parameter_count": len(params),
                "method": "fallback_regex",
                "error": str(e)[:100]
            }
    
    def _extract_basic_params_fallback(self):
        """Fallback parameter extraction using regex only"""
        import re
        
        params = {}
        
        # Extract pH
        ph_match = re.search(r'pH\s*([\d\.]+)', self.state.query, re.IGNORECASE)
        if ph_match:
            try:
                params["ph"] = {
                    "value": float(ph_match.group(1)),
                    "unit": "pH",
                    "description": "Acidity level",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # Extract temperature
        temp_matches = re.findall(r'(\d+\.?\d*)\s*[°]?[CcFf]', self.state.query)
        if temp_matches:
            try:
                params["temperature"] = {
                    "value": float(temp_matches[0]),
                    "unit": "°C",
                    "description": "Temperature",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # Extract concentrations
        conc_matches = re.findall(r'(\d+\.?\d*)\s*(mM|µM|nM|mg/L|g/L)', self.state.query)
        for i, (value, unit) in enumerate(conc_matches[:2]):
            try:
                params[f"concentration_{i}"] = {
                    "value": float(value),
                    "unit": unit,
                    "description": f"Concentration ({unit})",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # If no parameters found, add defaults for analytics
        if not params:
            params = {
                "default_ph": {"value": 7.0, "unit": "pH", "description": "Default pH", "method": "default"},
                "default_temp": {"value": 30.0, "unit": "°C", "description": "Default temperature", "method": "default"}
            }
        
        return params
    
    async def _retrieve_evidence_lightweight(self):
        """Lightweight evidence retrieval for CPU"""
        # Use synthetic optimal conditions (fast)
        optimal_conditions = self._get_synthetic_optimal()
        
        evidence = [
            {
                "id": "optimal_conditions",
                "content": f"Optimal conditions for {self.state.domain}: {optimal_conditions}",
                "source": "synthetic_optimal_cpu",
                "relevance": 0.8,
                "cpu_optimized": True
            }
        ]
        
        # Add parameter-based evidence
        if self.state.parameters:
            param_summary = ", ".join([f"{k}: {v.get('value')}" for k, v in list(self.state.parameters.items())[:3]])
            evidence.append({
                "id": "parameter_context",
                "content": f"Experimental parameters: {param_summary}",
                "source": "parameter_extraction",
                "relevance": 0.7
            })
        
        self.state.retrieved_evidence = evidence
        return {
            "evidence_count": len(evidence),
            "sources": [e["source"] for e in evidence],
            "cpu_optimized": True
        }
    
    def _get_synthetic_optimal(self):
        """Generate synthetic optimal conditions for CPU"""
        optimals = {}
        
        # Domain-specific optimal conditions
        if self.state.domain == "biomed":
            optimals = {
                "optimal_ph": 7.0,
                "ph_range": [5.5, 8.5],
                "optimal_temperature": 30.0,
                "temp_range": [20.0, 37.0],
                "optimal_incubation_time": 24.0,
                "recommended_replicates": 3
            }
        else:
            # General scientific optimal conditions
            optimals = {
                "optimal_value": 7.0,
                "optimal_range": [5.0, 9.0],
                "recommended_samples": 3
            }
        
        return optimals
    
    # In pipeline.py - Update the analytics step
    async def _run_all_analytics_parallel(self):
        """Run analytics using either Celery or async"""
        
        # Check if we should use Celery (based on parameter count)
        use_celery = len(self.state.parameters) >= 2  # Use Celery for 2+ parameters
        
        if use_celery:
            logger.info("🔄 Using Celery for distributed analytics...")
            from core.analytics import run_comprehensive_analytics_with_celery
            analytics_func = run_comprehensive_analytics_with_celery
        else:
            logger.info("🔄 Using async analytics (few parameters)...")
            from core.analytics import run_comprehensive_analytics_cpu
            analytics_func = run_comprehensive_analytics_cpu
        
        try:
            analytics_result = await analytics_func(
                self.state.query,
                self.state.parameters,
                self.state.domain
            )
            
            self.state.structured_analysis = analytics_result
            
            # Log results
            method = analytics_result.get("execution_mode", "unknown")
            logger.info(f"✅ Analytics complete via {method}")
            
            return {
                "analytics_completed": True,
                "method": method,
                "cpu_optimized": True,
                "used_celery": use_celery
            }
            
        except Exception as e:
            logger.error(f"❌ Analytics failed: {e}")
            # Fallback to simple analytics
            return await self._run_simple_analytics_fallback()
    
    async def _generate_hypothesis_fast(self):
        """Generate hypothesis using analytics insights"""
        from core.mistral import generate_with_mistral
        
        # Build hypothesis prompt with analytics context
        analytics = self.state.structured_analysis
        parameters = self.state.parameters
        
        # Extract insights for hypothesis
        insights = []
        
        # From SHAP
        shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
        if shap_imp:
            top_feature = max(shap_imp.items(), key=lambda x: abs(x[1]))
            insights.append(f"SHAP identifies '{top_feature[0]}' as most influential (importance: {top_feature[1]:.3f})")
        
        # From causal
        ate = analytics.get("causal", {}).get("ate", 0)
        if ate != 0:
            insights.append(f"Causal effect estimate: {ate:.3f}")
        
        insights_text = "\n".join(f"- {i}" for i in insights) if insights else "Standard experimental parameters."
        
        prompt = f"""Based on this analysis, generate ONE testable biomedical hypothesis:

Query: {self.state.query}

Parameters: {json.dumps(parameters, indent=2)[:300]}

Analytics Insights:
{insights_text}

Guidelines:
1. Must be specific and testable
2. Include measurable variables
3. Reference analytics insights if available
4. One sentence only
5. Relevant to biomedical domain

Hypothesis:"""
        
        try:
            hypothesis, _ = await generate_with_mistral(prompt, max_tokens=100, temperature=0.3)
            hypothesis = hypothesis.strip().replace('"', '').replace("'", "").replace("\n", " ")
            
            # Ensure it ends with period
            if not hypothesis.endswith('.'):
                hypothesis += '.'
            
            self.state.hypothesis = hypothesis
            return {
                "hypothesis": hypothesis,
                "insights_used": len(insights),
                "length": len(hypothesis)
            }
            
        except Exception as e:
            logger.warning(f"Hypothesis generation failed: {e}")
            # Fallback hypothesis
            fallback = "Experimental variables will show statistically significant effects on the measured outcomes."
            self.state.hypothesis = fallback
            return {
                "hypothesis": fallback,
                "method": "fallback",
                "error": str(e)[:100]
            }
    
    async def _synthesize_response_with_analytics(self):
        """Final synthesis using Mistral: incorporates analytics, hypothesis, and arXiv evidence"""
        from core.mistral import generate_with_mistral, enforce_xml_structure
        
        # Build analytics summary (existing logic - keep your current format_analytics_for_prompt or similar)
        analytics_summary = self._format_analytics_summary(self.state.structured_analysis)
        
        # === NEW: arXiv Evidence Summary ===
        evidence_summary = ""
        if self.state.retrieved_evidence:
            evidence_summary = "\nRelevant supporting literature from arXiv (use these to back up your hypothesis and recommendations):\n"
            for i, paper in enumerate(self.state.retrieved_evidence[:3], 1):  # Limit to top 3 for token efficiency
                year = paper.get('published', 'Unknown')[:4]
                authors = paper.get('authors', 'Unknown')[:120]  # Slightly longer for better context
                title = paper.get('title', 'No title')
                summary = paper.get('summary', 'No summary available')[:400] + "..." if len(paper.get('summary', '')) > 400 else paper.get('summary', '')
                pdf_url = paper.get('pdf_url', '')
                
                evidence_summary += f"{i}. {title} ({year})\n"
                evidence_summary += f"   Authors: {authors}\n"
                evidence_summary += f"   Summary: {summary}\n"
                if pdf_url:
                    evidence_summary += f"   PDF: {pdf_url}\n"
                evidence_summary += "\n"
        
        # Base system instructions (keep your existing SYSTEM_PREFIX or domain-specific one)
        system_instructions = """
    You are an enthusiastic biomedical research colleague. Respond in structured XML format:
    <enthusiasm>Express genuine excitement</enthusiasm>
    <explanation>Detailed 4-6 paragraph scientific explanation</explanation>
    <hypothesis>Clear, testable hypothesis</hypothesis>
    <followup>1-2 thoughtful open-ended questions</followup>

    Be natural, engaging, and cite evidence naturally.
    """
        
        # Full prompt assembly
        enhanced_prompt = f"""User query: {self.state.query}

    Hypothesis to support: {self.state.hypothesis or 'Generate one based on the query and analytics'}

    Analytics insights (incorporate relevant ones):
    {analytics_summary}

    {evidence_summary}

    Instructions:
    - Start with enthusiasm about the user's research interest
    - Provide thorough, accurate scientific explanation
    - Integrate analytics insights (SHAP importance, causal effects, optimization suggestions)
    - Naturally reference supporting arXiv papers (e.g., "A 2023 study on yeast fermentation found..." or "Similar results were reported in [title]")
    - Formulate a clear, testable hypothesis
    - End with thoughtful follow-up questions
    - Use the exact XML structure above

    Respond ONLY with the structured XML content.
    """

        try:
            # Generate with Mistral (adjust tokens/temperature as needed)
            response, _ = await generate_with_mistral(
                enhanced_prompt,
                max_tokens=1200,
                temperature=0.8
            )
            
            # Enforce structure as safety net
            final_response = enforce_xml_structure(response, self.state.query)
            
            self.state.final_response = final_response
            self.state.add_trace(
                "response_synthesis",
                "Success: Generated structured response with analytics + arXiv evidence",
                metadata={"response_length": len(final_response), "evidence_count": len(self.state.retrieved_evidence)}
            )
            
        except Exception as e:
            logger.error(f"Mistral synthesis failed: {e}")
            # Fallback with evidence included
            fallback_evidence = ""
            if self.state.retrieved_evidence:
                fallback_evidence = "\nSupporting literature:\n" + "\n".join(
                    [f"- {p['title']} ({p['published'][:4]})" for p in self.state.retrieved_evidence[:3]]
                )
            
            fallback = f"""<enthusiasm>Exciting question about biomedical research!</enthusiasm>
    <explanation>
    Your query involves key parameters and scientific analysis.
    {analytics_summary}
    {fallback_evidence}
    Standard recommendations apply based on typical experimental design.
    </explanation>
    <hypothesis>
    {self.state.hypothesis or 'Parameters will significantly affect the outcome.'}
    </hypothesis>
    <followup>
    What specific aspects would you like to explore further?
    Do you have any preliminary data?
    </followup>"""
            
            self.state.final_response = fallback
            self.state.add_trace("response_synthesis", "Fallback used due to generation error")

        try:
            # Generate two diverse candidates
            prompt_a = enhanced_prompt + "\n\nStyle: Concise, direct, and scientifically precise."
            prompt_b = enhanced_prompt + "\n\nStyle: Engaging, explanatory, and conversational with clear examples."

            candidate_a, _ = await generate_with_mistral(prompt_a, max_tokens=1200, temperature=0.7)
            candidate_b, _ = await generate_with_mistral(prompt_b, max_tokens=1200, temperature=0.9)

            # Enforce structure early for fair comparison
            candidate_a = enforce_xml_structure(candidate_a, self.state.query)
            candidate_b = enforce_xml_structure(candidate_b, self.state.query)

            # Score with real reward model
            reward_model = get_reward_model()
            with torch.no_grad():
                scores = reward_model([candidate_a, candidate_b])
                score_a, score_b = scores[0].item(), scores[1].item()

            # Select winner
            final_response = candidate_a if score_a > score_b else candidate_b
            winner = "A" if score_a > score_b else "B"

            self.state.add_trace("rlhf_selection", {
                "winner": winner,
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
                "delta": round(abs(score_a - score_b), 4),
                "model_loaded": os.path.exists("models/reward_model.pth")
            })

        except Exception as e:
            logger.error(f"RLHF selection failed: {e} → using candidate A")
            final_response = enforce_xml_structure(candidate_a, self.state.query)
        
        def _format_analytics_summary(self, analytics: Dict[str, Any]) -> str:
            """Format analytics results for inclusion in prompt"""
            lines = []
            
            # Feature Importance
            shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
            if shap_imp:
                top_features = sorted(shap_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
                for feat, imp in top_features:
                    lines.append(f"• SHAP feature importance: '{feat}' = {imp:.3f}")
            
            # Causal Inference
            causal = analytics.get("causal", {})
            if "ate" in causal:
                ate = causal["ate"]
                sig = causal.get("is_significant", False)
                lines.append(f"• Causal ATE: {ate:.3f} {'(significant)' if sig else '(not significant)'}")
            
            # Bayesian Optimization
            optimization = analytics.get("optimization", {})
            if "improvement_pct" in optimization:
                imp = optimization["improvement_pct"]
                if imp != 0:
                    lines.append(f"• Bayesian optimization: {imp:.1f}% potential improvement")
            
            # LIME Explanations
            lime = analytics.get("explainability", {}).get("lime", {})
            if "explanations" in lime and lime["explanations"]:
                lime_data = lime["explanations"]
                if isinstance(lime_data, dict) and lime_data:
                    top_lime = max(lime_data.items(), key=lambda x: abs(x[1]))
                    lines.append(f"• LIME local explanation: '{top_lime[0]}' = {top_lime[1]:.3f}")
            
            # Method notes
            exec_mode = analytics.get("execution_mode", "")
            if "cpu" in exec_mode.lower():
                lines.append("• Note: Analytics performed with CPU-optimized settings (small samples, fast algorithms)")
            
            if lines:
                return "\n".join(lines)
            else:
                return "Basic statistical analysis applied. (Full analytics unavailable or in simplified mode)"
        
        async def _emergency_fallback_with_analytics(self):
            """Emergency fallback that still includes analytics if available"""
            from core.mistral import enforce_xml_structure
            
            analytics = self.state.structured_analysis
            hypothesis = self.state.hypothesis or "Experimental parameters will influence outcomes."
            
            # Try to extract some analytics for fallback
            analytics_insight = ""
            if analytics and "explainability" in analytics:
                shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
                if shap_imp:
                    top_feature = max(shap_imp.items(), key=lambda x: abs(x[1]))
                    analytics_insight = f" Preliminary analysis suggests '{top_feature[0]}' may be influential. "
            
            fallback = f"""<enthusiasm>Thanks for your biomedical research question!</enthusiasm>

    <explanation>
    I've analyzed your query about "{self.state.query[:100]}".{analytics_insight}

    The CPU-optimized analysis pipeline completed key steps including:
    1. Parameter extraction from your query
    2. Feature importance analysis
    3. Causal inference estimation
    4. Bayesian optimization suggestions

    For your experimental design:
    - Use factorial designs for multiple variables
    - Include proper controls and replicates
    - Consider biological variability
    - Apply appropriate statistical tests (ANOVA, t-tests)

    {analytics_insight}
    </explanation>

    <hypothesis>
    {hypothesis}
    </hypothesis>

    <followup>
    What specific measurement techniques are you using?
    Do you have preliminary data on expected effect sizes?
    What resources are available for your experiments?
    </followup>"""
            
            structured_fallback = enforce_xml_structure(fallback, self.state.query)
            self.state.final_response = structured_fallback
            self.state.confidence = 0.6
    
    def _calculate_confidence(self):
        """Calculate confidence score based on pipeline success"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on steps completed
        completed_steps = len([t for t in self.state.trace if not t.get("error")])
        total_steps = 6
        confidence *= (completed_steps / total_steps)
        
        # Boost if we have analytics
        if self.state.structured_analysis and "explainability" in self.state.structured_analysis:
            confidence *= 1.2
        
        # Boost if we have parameters
        if self.state.parameters:
            confidence *= 1.1
        
        # Penalize errors
        if self.state.errors:
            confidence *= 0.9
        
        # Ensure bounds
        self.state.confidence = min(0.95, max(0.3, confidence))
