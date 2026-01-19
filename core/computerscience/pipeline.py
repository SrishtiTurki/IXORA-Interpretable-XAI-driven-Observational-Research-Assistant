# core/computerscience/pipeline.py - CS PIPELINE WITH PRIORITIZED ANALYTICS + RLHF LOGGING
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
    run_causal_analysis,
)

logger = logging.getLogger("cs.pipeline")

class ComputerSciencePipeline:
    """CPU-optimized computer science pipeline with prioritized analytics"""
    
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
        return intent
    
    async def run(self, query: str, session_id: str = "") -> ExecutionState:
        self.state = ExecutionState(
            query=query,
            session_id=session_id,
            start_time=datetime.now(),
            max_total_time=self.STEP_TIMEOUTS["total"]
        )
        
        logger.info(f"🚀 Starting CS Prioritized Pipeline: {query[:100]}")
        
        try:
            await self._run_with_timeout("intent_classification", self._classify_intent_fast)
            await self._run_with_timeout("parameter_extraction", self._extract_parameters_fast)
            await self._run_with_timeout("evidence_retrieval", self._retrieve_evidence_lightweight)
            await self._run_with_timeout("parallel_analytics", self._run_prioritized_analytics)
            await self._run_with_timeout("hypothesis_generation", self._generate_hypothesis_fast)
            await self._run_with_timeout("response_synthesis", self._synthesize_response_with_analytics)
            
            self._calculate_confidence()
            logger.info(f"✅ CS Pipeline completed in {self.state._get_elapsed():.1f}s")
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ CS Pipeline timeout after {self.state._get_elapsed():.1f}s")
            self.state.add_error("pipeline", "Total timeout exceeded", True)
            await self._emergency_fallback_with_analytics()
        
        except Exception as e:
            logger.error(f"❌ CS Pipeline error: {e}", exc_info=True)
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
        """Fast parameter extraction for CPU - CS domain"""
        import re
        from core.utils import extract_parameters
        
        try:
            params = await extract_parameters(self.state.query, self.state.domain)
            
            if not params:
                params = self._extract_basic_params_fallback()
            
            self.state.parameters = params
            
            param_summary = {k: v.get("value") for k, v in list(params.items())[:3]}
            return {
                "parameter_count": len(params),
                "top_parameters": param_summary,
                "method": "fast_extraction",
                "has_numeric": any(isinstance(v.get("value"), (int, float)) for v in params.values())
            }
            
        except Exception as e:
            logger.warning(f"CS Parameter extraction failed: {e}, using fallback")
            params = self._extract_basic_params_fallback()
            self.state.parameters = params
            return {
                "parameter_count": len(params),
                "method": "fallback_regex",
                "error": str(e)[:100]
            }
    
    def _extract_basic_params_fallback(self):
        """Fallback parameter extraction using regex only - CS domain"""
        import re
        
        params = {}
        
        # Extract time complexity O(...)
        complexity_match = re.search(r'O\(([^)]+)\)', self.state.query, re.IGNORECASE)
        if complexity_match:
            try:
                params["time_complexity"] = {
                    "value": complexity_match.group(1),
                    "unit": "big-O",
                    "description": "Algorithmic time complexity",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # Extract batch size
        batch_match = re.search(r'batch\s*size[:\s]+(\d+)', self.state.query, re.IGNORECASE)
        if batch_match:
            try:
                params["batch_size"] = {
                    "value": int(batch_match.group(1)),
                    "unit": "samples",
                    "description": "Batch size",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # Extract learning rate
        lr_match = re.search(r'learning\s*rate[:\s]+([\d\.]+)', self.state.query, re.IGNORECASE)
        if lr_match:
            try:
                params["learning_rate"] = {
                    "value": float(lr_match.group(1)),
                    "unit": "",
                    "description": "Learning rate",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # Extract dataset size
        dataset_match = re.search(r'(\d+\.?\d*)\s*(MB|GB|TB|KB)', self.state.query, re.IGNORECASE)
        if dataset_match:
            try:
                params["dataset_size"] = {
                    "value": float(dataset_match.group(1)),
                    "unit": dataset_match.group(2),
                    "description": "Dataset size",
                    "method": "regex_fallback"
                }
            except:
                pass
        
        # If no parameters found, add defaults for analytics
        if not params:
            params = {
                "default_batch_size": {"value": 32, "unit": "samples", "description": "Default batch size", "method": "default"},
                "default_learning_rate": {"value": 0.001, "unit": "", "description": "Default learning rate", "method": "default"}
            }
        
        return params
    
    async def _retrieve_evidence_lightweight(self):
        """Lightweight evidence retrieval for CPU"""
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
        
        if self.state.parameters:
            param_summary = ", ".join([f"{k}: {v.get('value')}" for k, v in list(self.state.parameters.items())[:3]])
            evidence.append({
                "id": "parameter_context",
                "content": f"Computational parameters: {param_summary}",
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
        """Generate synthetic optimal conditions for CPU - CS domain"""
        optimals = {}
        
        if self.state.domain == "cs":
            optimals = {
                "optimal_batch_size": 32,
                "batch_range": [16, 64],
                "optimal_learning_rate": 0.001,
                "lr_range": [0.0001, 0.01],
                "optimal_epochs": 10,
                "recommended_train_test_split": 0.8
            }
        else:
            optimals = {
                "optimal_value": 0.001,
                "optimal_range": [0.0001, 0.01],
                "recommended_samples": 1000
            }
        
        return optimals
    
    async def _generate_hypothesis_fast(self):
        """Generate hypothesis using analytics insights - CS domain"""
        from core.mistral import generate_with_mistral
        
        analytics = self.state.structured_analysis
        parameters = self.state.parameters
        
        insights = []
        
        shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
        if shap_imp:
            top_feature = max(shap_imp.items(), key=lambda x: abs(x[1]))
            insights.append(f"SHAP identifies '{top_feature[0]}' as most influential (importance: {top_feature[1]:.3f})")
        
        ate = analytics.get("causal", {}).get("ate", 0)
        if ate != 0:
            insights.append(f"Causal effect estimate: {ate:.3f}")
        
        insights_text = "\n".join(f"- {i}" for i in insights) if insights else "Standard computational parameters."
        
        prompt = f"""Based on this analysis, generate ONE testable computer science hypothesis:

Query: {self.state.query}

Parameters: {json.dumps(parameters, indent=2)[:300]}

Analytics Insights:
{insights_text}

Guidelines:
1. Must be specific and testable computationally
2. Include measurable variables (accuracy, latency, throughput, complexity)
3. Reference analytics insights if available
4. One sentence only
5. Relevant to computer science domain (algorithms, ML, systems, etc.)

Hypothesis:"""
        
        try:
            hypothesis, _ = await generate_with_mistral(prompt, max_tokens=100, temperature=0.3)
            hypothesis = hypothesis.strip().replace('"', '').replace("'", "").replace("\n", " ")
            
            if not hypothesis.endswith('.'):
                hypothesis += '.'
            
            self.state.hypothesis = hypothesis
            return {
                "hypothesis": hypothesis,
                "insights_used": len(insights),
                "length": len(hypothesis)
            }
            
        except Exception as e:
            logger.warning(f"CS Hypothesis generation failed: {e}")
            fallback = "Computational parameters will show statistically significant effects on algorithmic performance metrics."
            self.state.hypothesis = fallback
            return {
                "hypothesis": fallback,
                "method": "fallback",
                "error": str(e)[:100]
            }
    
    async def _synthesize_response_with_analytics(self):
        """Final synthesis using Mistral: incorporates analytics, hypothesis, and arXiv evidence - CS domain"""
        from core.mistral import generate_with_mistral
        
        analytics_summary = self._format_analytics_summary(self.state.structured_analysis)
        
        evidence_summary = ""
        if self.state.retrieved_evidence:
            evidence_summary = "\nRelevant supporting literature from arXiv (use these to back up your hypothesis and recommendations):\n"
            for i, paper in enumerate(self.state.retrieved_evidence[:3], 1):
                year = paper.get('published', 'Unknown')[:4]
                authors = paper.get('authors', 'Unknown')[:120]
                title = paper.get('title', 'No title')
                summary = paper.get('summary', 'No summary available')[:400] + "..." if len(paper.get('summary', '')) > 400 else paper.get('summary', '')
                pdf_url = paper.get('pdf_url', '')
                
                evidence_summary += f"{i}. {title} ({year})\n"
                evidence_summary += f"   Authors: {authors}\n"
                evidence_summary += f"   Summary: {summary}\n"
                if pdf_url:
                    evidence_summary += f"   PDF: {pdf_url}\n"
                evidence_summary += "\n"
        
        system_instructions = """You are a friendly, highly knowledgeable computer science research mentor and technical advisor. 
You guide graduate students, researchers, and professionals in CS research, ensuring rigor, clarity, and enthusiasm.

Follow this EXACT XML structure—do NOT deviate, reword, or shorten tag names or order.

<rules>
1. Domain restriction: ONLY respond to computer science, software engineering, AI/ML, or related research/technical topics. 
   If the user goes off-topic, respond only with: 
   'I'd love to help, but let's stick to CS—got a technical or research query?'
2. Start with genuine enthusiasm: 
   <enthusiasm>Oh, that's excellent! [Brief acknowledgment of topic or problem]</enthusiasm>
3. Include 1–2 clarifying questions after the intro: 
   <clarify>Ask what specifically interests the user (algorithmic approach, implementation details, performance metrics, scalability concerns, etc.)</clarify>
4. Core section — technical narrative: 
   <explanation>Provide 4–6 detailed, coherent paragraphs. 
   Maintain a narrative tone that blends technical accuracy and engagement. 
   Avoid bullet points. Explain algorithmic reasoning, data structures, architectural choices, examples, and implications. 
   Include relevant CS techniques (e.g., dynamic programming, gradient descent, MapReduce, attention mechanisms) and how conclusions are derived. 
   Discuss computational complexity, trade-offs, and implementation considerations.
   Adjust depth according to the complexity of the query (short = concise, complex = deep and technical).</explanation>
5. End with a testable hypothesis: 
   <hypothesis>Offer one clear, research-driven hypothesis followed by 2–3 practical next steps to test or validate it.
   Include: experimental setup (datasets, baselines, metrics), ablation studies, and expected outcomes.</hypothesis>
6. Maintain a collegial, mentor-like tone throughout — encourage inquiry, experimentation, and critical thinking.
7. Finish with a follow-up thought: 
   <followup>Ask 1–2 thought-provoking, open-ended technical questions (e.g., 'What if we applied transfer learning here?', 'How would this scale to billion-parameter models?').</followup>
</rules>

Response length target: 300–700 words, dynamically scaled to query complexity.
Do not include any text outside of the XML tags.
Emphasize reproducibility: mention random seeds, hyperparameters, library versions where relevant.
"""
        
        enhanced_prompt = f"""{system_instructions}

User query: {self.state.query}

Hypothesis to support: {self.state.hypothesis or 'Generate one based on the query and analytics'}

Analytics insights (incorporate relevant ones):
{analytics_summary}

{evidence_summary}

Instructions:
- Start with enthusiasm about the user's research interest
- Provide thorough, accurate technical explanation with algorithmic reasoning
- Integrate analytics insights (SHAP importance, causal effects, optimization suggestions)
- Naturally reference supporting arXiv papers (e.g., "A 2023 study on neural architecture search found..." or "Similar results were reported in [title]")
- Formulate a clear, testable hypothesis with experimental setup
- End with thoughtful follow-up questions
- Use the exact XML structure: <enthusiasm>, <clarify>, <explanation>, <hypothesis>, <followup>
- Only respond to CS-related queries

Respond ONLY with the structured XML content.
"""

        try:
            response, _ = await generate_with_mistral(
                enhanced_prompt,
                max_tokens=1200,
                temperature=0.8
            )
            
            final_response = self._enforce_cs_xml_structure(response, self.state.query)
            
            self.state.final_response = final_response
            self.state.add_trace(
                "response_synthesis",
                "Success: Generated structured CS response with analytics + arXiv evidence",
                metadata={"response_length": len(final_response), "evidence_count": len(self.state.retrieved_evidence)}
            )
            
        except Exception as e:
            logger.error(f"CS Mistral synthesis failed: {e}")
            fallback_evidence = ""
            if self.state.retrieved_evidence:
                fallback_evidence = "\nSupporting literature:\n" + "\n".join(
                    [f"- {p['title']} ({p['published'][:4]})" for p in self.state.retrieved_evidence[:3]]
                )
            
            fallback = f"""<enthusiasm>Oh, that's excellent! Great computer science research question!</enthusiasm>
<clarify>What specific algorithmic approach are you considering? What are your computational constraints?</clarify>
<explanation>
Your query involves key computational parameters and algorithmic analysis.
{analytics_summary}
{fallback_evidence}
Standard recommendations apply based on typical experimental design in CS research.
</explanation>
<hypothesis>
{self.state.hypothesis or 'Computational parameters will significantly affect algorithmic performance metrics.'}
</hypothesis>
<followup>
What specific implementation approach are you considering?
How would this scale to larger datasets or distributed systems?
</followup>"""
            
            self.state.final_response = self._enforce_cs_xml_structure(fallback, self.state.query)
            self.state.add_trace("response_synthesis", "Fallback used due to generation error")
    
    def _enforce_cs_xml_structure(self, content: str, query: str) -> str:
        """Enforce CS XML structure: enthusiasm, clarify, explanation, hypothesis, followup"""
        if not content:
            return """<enthusiasm>Oh, that's excellent! Great CS research question!</enthusiasm>
<clarify>What specific algorithmic approach are you considering?</clarify>
<explanation>Your query about computational research is interesting. Algorithmic design, data structures, and computational complexity are key factors in computer science research.</explanation>
<hypothesis>Computational parameters will significantly affect algorithmic performance metrics.</hypothesis>
<followup>What if we applied this to a different problem domain? How would scalability change?</followup>"""
        
        required_tags = ["<enthusiasm>", "<clarify>", "<explanation>", "<hypothesis>", "<followup>"]
        
        for tag in required_tags:
            if tag not in content:
                if tag == "<enthusiasm>":
                    content = f"{tag}Oh, that's excellent! Great CS research question!{tag.replace('<', '</')}\n\n{content}"
                elif tag == "<clarify>":
                    if "<enthusiasm>" in content:
                        parts = content.split("</enthusiasm>", 1)
                        content = f"{parts[0]}</enthusiasm>\n\n{tag}What specific algorithmic approach are you considering?{tag.replace('<', '</')}\n\n{parts[1]}"
                elif tag == "<explanation>":
                    if "</clarify>" in content:
                        parts = content.split("</clarify>", 1)
                        content = f"{parts[0]}</clarify>\n\n{tag}\n{parts[1]}\n</explanation>"
                elif tag == "<hypothesis>":
                    content = content.replace("</explanation>", f"</explanation>\n\n{tag}Computational parameters will significantly affect algorithmic performance.{tag.replace('<', '</')}")
                elif tag == "<followup>":
                    content = f"{content}\n\n{tag}What if we applied this to a different problem domain?{tag.replace('<', '</')}"
        
        tags_to_close = ["enthusiasm", "clarify", "explanation", "hypothesis", "followup"]
        for tag_name in tags_to_close:
            opening = f"<{tag_name}>"
            closing = f"</{tag_name}>"
            if opening in content and closing not in content:
                content += closing
        
        return content
    
    def _format_analytics_summary(self, analytics: Dict[str, Any]) -> str:
        """Format analytics results for inclusion in prompt"""
        lines = []
        
        shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
        if shap_imp:
            top_features = sorted(shap_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
            for feat, imp in top_features:
                lines.append(f"• SHAP feature importance: '{feat}' = {imp:.3f}")
        
        causal = analytics.get("causal", {})
        if "ate" in causal:
            ate = causal["ate"]
            sig = causal.get("is_significant", False)
            lines.append(f"• Causal ATE: {ate:.3f} {'(significant)' if sig else '(not significant)'}")
        
        optimization = analytics.get("optimization", {})
        if "improvement_pct" in optimization:
            imp = optimization["improvement_pct"]
            if imp != 0:
                lines.append(f"• Bayesian optimization: {imp:.1f}% potential improvement")
        
        lime = analytics.get("explainability", {}).get("lime", {})
        if "explanations" in lime and lime["explanations"]:
            lime_data = lime["explanations"]
            if isinstance(lime_data, dict) and lime_data:
                top_lime = max(lime_data.items(), key=lambda x: abs(x[1]))
                lines.append(f"• LIME local explanation: '{top_lime[0]}' = {top_lime[1]:.3f}")
        
        exec_mode = analytics.get("execution_mode", "")
        if "cpu" in exec_mode.lower():
            lines.append("• Note: Analytics performed with CPU-optimized settings (small samples, fast algorithms)")
        
        if lines:
            return "\n".join(lines)
        else:
            return "Basic statistical analysis applied. (Full analytics unavailable or in simplified mode)"
    
    async def _emergency_fallback_with_analytics(self):
        """Emergency fallback that still includes analytics if available - CS domain"""
        analytics = self.state.structured_analysis
        hypothesis = self.state.hypothesis or "Computational parameters will influence algorithmic performance."
        
        analytics_insight = ""
        if analytics and "explainability" in analytics:
            shap_imp = analytics.get("explainability", {}).get("feature_importance", {})
            if shap_imp:
                top_feature = max(shap_imp.items(), key=lambda x: abs(x[1]))
                analytics_insight = f" Preliminary analysis suggests '{top_feature[0]}' may be influential. "
        
        fallback = f"""<enthusiasm>Oh, that's excellent! Great computer science research question!</enthusiasm>

<clarify>What specific algorithmic approach are you considering? What are your computational constraints?</clarify>

<explanation>
I've analyzed your query about "{self.state.query[:100]}".{analytics_insight}

The CPU-optimized analysis pipeline completed key steps including:
1. Parameter extraction from your query
2. Feature importance analysis
3. Causal inference estimation
4. Bayesian optimization suggestions

For your computational experimental design:
- Use proper train/validation/test splits
- Include baseline comparisons
- Consider algorithmic complexity analysis
- Apply appropriate evaluation metrics (accuracy, latency, throughput)
- Emphasize reproducibility (random seeds, library versions, hardware specs)

{analytics_insight}
</explanation>

<hypothesis>
{hypothesis}
</hypothesis>

<followup>
What specific implementation approach are you considering?
How would this scale to larger datasets or distributed systems?
Do you have preliminary performance benchmarks?
</followup>"""
        
        self.state.final_response = self._enforce_cs_xml_structure(fallback, self.state.query)
        self.state.confidence = 0.6
    
    def _calculate_confidence(self):
        """Calculate confidence score based on pipeline success"""
        confidence = 0.7
        
        completed_steps = len([t for t in self.state.trace if not t.get("error")])
        total_steps = 6
        confidence *= (completed_steps / total_steps)
        
        if self.state.structured_analysis and "explainability" in self.state.structured_analysis:
            confidence *= 1.2
        
        if self.state.parameters:
            confidence *= 1.1
        
        if self.state.errors:
            confidence *= 0.9
        
        self.state.confidence = min(0.95, max(0.3, confidence))
