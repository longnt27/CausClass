"""
LLM Agent for Causal Graph Optimization - CausClass Project
---------------------------------------
This module interfaces with the DeepSeek LLM API to propose causal graph edits.
It includes mathematical guardrails to prevent hallucinations and an exponential 
backoff retry mechanism to handle API rate limits (e.g., HTTP 429).
"""

import json
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


def _is_transient(exc):
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    return (isinstance(exc, requests.HTTPError) and exc.response is not None
            and (exc.response.status_code == 429 or 500 <= exc.response.status_code < 600))

class LLMGraphAgent:
    """
    LLM-based optimization agent that interacts with the causal discovery pipeline.
    Utilizes context, raw structural priors, and tabular history to propose valid DAG edits.
    """
    def __init__(self, api_key, variables, context, *, model=None, timeout=(10, 90)):
        if not api_key or not str(api_key).strip():
            raise ValueError("DEEPSEEK_API_KEY is required for LLM proposals")
        if len(set(variables)) != len(variables):
            raise ValueError("variable names must be unique")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.variables = variables 
        self.context = context
        self.var_str = ', '.join([f"'{v}'" for v in self.variables])

    def format_current_graph(self, edges):
        """
        Formats the current edge list into a readable string for the LLM prompt,
        including weight magnitudes to guide deletion proposals.
        """
        if not edges: 
            return "[CURRENTLY EMPTY]"
            
        lines = []
        for e in edges:
            lines.append(f"- {e['source']} -> {e['target']}")
        return "\n".join(lines)

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient),
        reraise=True
    )
    def _call_llm_api(self, payload):
        """
        Isolated API execution method to cleanly apply the tenacity retry decorator.
        Raises HTTPError on bad status codes to trigger the retry logic.
        """
        res = requests.post(
            self.api_url, 
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        res.raise_for_status() 
        return res.json()

    def propose_edits(self, current_edges, local_tabu, feedback_momentum, raw_matrix):
        """
        Constructs the prompt, calls the LLM, and rigidly filters the proposed edits.
        """
        missing_potentials = []
        existing_pairs = set()
        
        for e in current_edges:
            existing_pairs.add((e["source"], e["target"]))
            existing_pairs.add((e["target"], e["source"]))

        for i, src in enumerate(self.variables):
            for j, tgt in enumerate(self.variables):
                if src != tgt and (src, tgt) not in existing_pairs:
                    score = float(raw_matrix[i, j])
                    missing_potentials.append((src, tgt, score))

        missing_potentials.sort(key=lambda x: x[2], reverse=True)
        hints = [f"- {s} -> {t} (Signal Score: {sc:.4f})" for s, t, sc in missing_potentials[:10]]
        hint_str = "\n".join(hints) if hints else "[NO_AVAILABLE_SIGNALS]"

        # =========================================================
        # PROMPT ENGINEERING
        # =========================================================
        system_prompt = f"""PROJECT CONTEXT: You are the core 'Pruning Agent' for "CausClass", a system designed to discover macro-level classroom behavior graphs from noisy Time-Series data. The baseline mathematical engine (AERCA) has extracted a messy graph full of spurious correlations. YOUR JOB IS TO CLEAN IT UP.

CRITICAL MACRO-LEVEL DEFINITIONS (MUST USE THESE):
- 'Talk': Widespread student-to-student chatting (Contagious noise).
- 'Read': A high percentage of the class focusing on reading.
- 'Phone': Epidemic of covert smartphone usage (Signals contagious isolation).
- 'Write': Multiple students writing.
- 'Lean': Widespread slouching/leaning (Contagious fatigue/low energy).
- 'Bow': Some students keeping their heads down, maybe tired or distracted.

CRITICAL CAUSALITY RULES (READ CAREFULLY BEFORE DELETING):
1. STRUCTURAL GRAPH ONLY: The edges provided to you represent STRUCTURAL CONNECTIONS ONLY. They DO NOT imply a positive or negative sign.
2. NEGATIVE CAUSALITY IS VALID: If A suppresses/decreases B (e.g., Talking suppresses Reading), the structural link EXISTS and is VALID. 
3. DO NOT BE THE SIGN POLICE: NEVER propose deleting an edge by arguing "it suppresses B, so the positive sign is wrong". If A impacts B in ANY WAY (increasing OR suppressing), YOU MUST KEEP THE EDGE. DO NOT DELETE IT!

CRITICAL PRUNING AND ADDING RULES:
1. BEHAVIORS CAUSE BEHAVIORS (NO LATENT STATES): You MUST assume that behaviors directly influence other behaviors through peer contagion or physical disruption. NEVER delete an edge by arguing that both behaviors are just "symptoms" of a hidden internal state (like 'boredom', 'engagement', 'fatigue', 'energy', or 'motivation'). 
2. NEGATIVE CAUSALITY = KEEP: If behavior A creates an environment that suppresses behavior B (e.g., Talking creates noise that stops Reading; Phone use isolates students and stops Talking), YOU MUST KEEP THE EDGE. 
3. DELETE ONLY ABSURD PHYSICAL SEQUENCES: Only delete an edge if the physical sequence is absurd (e.g., Reading causes Standing, or Standing causes Leaning). 
4. NO HALLUCINATIONS: Evaluate ONLY the nodes provided. Do not invent root causes.
5. THE 'ADD' ACTION (RESURRECTION): You will be provided with 'SUSPECTED MISSING LINKS' (strong mathematical signals ignored by the baseline). If one of these missing links makes PERFECT MACRO-LOGICAL SENSE (e.g., Talk suppresses Read, Phone suppresses Talk), output an 'add' action! Do not add blindly, only add if the behavior clearly drives or suppresses the target.

OUTPUT FORMAT: Return ONLY a valid JSON object containing an "edits" key mapped to an array of 1 to 3 objects.
EXAMPLE:
{{
  "edits": [
    {{"action": "delete", "source": "Read", "target": "Stand", "reasoning": "Reading is stationary, it does not physically cause standing."}},
    {{"action": "add", "source": "Talk", "target": "Read", "reasoning": "Widespread talking creates a disruptive noise environment that heavily suppresses reading."}}
  ]
}}

"""

        user_prompt = f"""
=== CURRENT GRAPH TO OPTIMIZE ===
{self.format_current_graph(current_edges)}

=== SUSPECTED MISSING LINKS (MATH SIGNALS) ===
{hint_str}

TABU LIST (DO NOT SUGGEST THESE EDGES): {json.dumps(local_tabu)}
SYSTEM FEEDBACK (Prior MSE impacts): {feedback_momentum}

TASK: Return a JSON object with the "edits" array containing 1 to 3 strictly evaluated edits.
"""
        
        payload = {
            "model": self.model, 
            "temperature": 0.1, 
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            response_data = self._call_llm_api(payload)
            content = response_data['choices'][0]['message']['content'].strip()
            
            parsed_data = json.loads(content)
            if not isinstance(parsed_data, dict) or not isinstance(parsed_data.get("edits"), list):
                return []
            raw_edits = parsed_data["edits"]
            
            final_edits = []
            seen = set()
            current_pairs = {(e["source"], e["target"]) for e in current_edges}
            for edit in raw_edits:
                if not isinstance(edit, dict):
                    continue
                s = edit.get("source")
                t = edit.get("target")
                
                # Hallucination Guardrail[cite: 2]
                action = edit.get("action")
                if (not isinstance(s, str) or not isinstance(t, str)
                        or not isinstance(action, str)):
                    continue
                key = (action, s, t)
                if (s in self.variables and t in self.variables and s != t
                        and action in {"add", "delete"} and key not in seen
                        and ((action == "add" and (s, t) not in current_pairs)
                             or (action == "delete" and (s, t) in current_pairs))):
                    seen.add(key)
                    final_edits.append(edit)
                else:
                    print(f"  [Security Guard] Trashed hallucinated edit: {s} -> {t}")
            
            return final_edits[:3]
            
        except requests.exceptions.HTTPError as e:
            print(f"  [LLM FATAL] API Rate Limit or Network Error after retries: {e}")
            return []
        except Exception as e:
            print(f"  [LLM Logic Error] Failed to parse response: {e}")
            return []
