from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


USAGE_KEYS = ("calls", "input_tokens", "output_tokens")


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class LLMConfig:
    backend: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = ""
    omit_temperature: bool = False
    token_limit_param: str = ""
    timeout_seconds: int = 120
    llama_cli_path: str = "llama-cli"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: str = ""
    max_input_tokens: int = 0
    truncation_side: str = "left"
    enable_thinking: bool = False
    local_files_only: bool = True
    anthropic_version: str = "2023-06-01"
    inference_geo: str = ""
    provider_order: tuple[str, ...] = ()
    provider_only: tuple[str, ...] = ()
    provider_sort: str = ""
    provider_allow_fallbacks: bool | None = None
    provider_require_parameters: bool | None = None
    openrouter_referrer: str = ""
    openrouter_title: str = ""


def build_llm(config: dict[str, Any]) -> LLM:
    llm_config = LLMConfig(
        backend=str(config.get("backend", "mock")),
        model=str(config.get("model", "")),
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        api_key_env=str(config.get("api_key_env", "")),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 1024)),
        system_prompt=str(config.get("system_prompt", "")),
        omit_temperature=bool(config.get("omit_temperature", False)),
        token_limit_param=str(config.get("token_limit_param", "")),
        timeout_seconds=int(config.get("timeout_seconds", 120)),
        llama_cli_path=str(config.get("llama_cli_path", "llama-cli")),
        torch_dtype=str(config.get("torch_dtype", "float16")),
        device_map=str(config.get("device_map", "auto")),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        attn_implementation=str(config.get("attn_implementation", "")),
        max_input_tokens=int(config.get("max_input_tokens", 0)),
        truncation_side=str(config.get("truncation_side", "left")),
        enable_thinking=bool(config.get("enable_thinking", False)),
        local_files_only=bool(config.get("local_files_only", True)),
        anthropic_version=str(config.get("anthropic_version", "2023-06-01")),
        inference_geo=str(config.get("inference_geo", "")),
        provider_order=tuple_str(config.get("provider_order", [])),
        provider_only=tuple_str(config.get("provider_only", [])),
        provider_sort=str(config.get("provider_sort", "")),
        provider_allow_fallbacks=optional_bool(config.get("provider_allow_fallbacks")),
        provider_require_parameters=optional_bool(config.get("provider_require_parameters")),
        openrouter_referrer=str(config.get("openrouter_referrer", "")),
        openrouter_title=str(config.get("openrouter_title", "")),
    )
    backend = llm_config.backend.lower()
    if backend == "mock":
        return MockLLM()
    if backend == "local_hf":
        return LocalHFLLM(llm_config)
    if backend == "ollama":
        return OllamaLLM(llm_config)
    if backend == "openai_compatible":
        return OpenAICompatibleLLM(llm_config)
    if backend == "anthropic":
        return AnthropicLLM(llm_config)
    if backend == "llama_cpp_cli":
        return LlamaCppCliLLM(llm_config)
    raise ValueError(f"Unknown LLM backend: {llm_config.backend}")


def chat_messages(config: LLMConfig, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def rendered_prompt_for_usage(config: LLMConfig, prompt: str) -> str:
    if not config.system_prompt:
        return prompt
    return f"{config.system_prompt}\n\n{prompt}"


class MockLLM:
    """Deterministic backend for pipeline smoke tests."""

    def __init__(self) -> None:
        self.usage = empty_usage()

    def generate(self, prompt: str) -> str:
        response = self._generate(prompt)
        record_usage(self, estimated_token_count(prompt), estimated_token_count(response))
        return response

    def _generate(self, prompt: str) -> str:
        lower = prompt.lower()
        suspected_step = first_suspicious_step(prompt)
        agent = agent_for_step(prompt, suspected_step) or "Unknown"
        if "agentrx global constraint synthesis stage" in lower:
            return json.dumps(
                {
                    "constraints": [
                        {
                            "id": "G1",
                            "assertion_name": "ground_claims_in_observations",
                            "taxonomy_targets": ["Misinterpretation of Tool Output / Handoff Failure"],
                            "constraint_type": "PROVENANCE",
                            "event_trigger": {
                                "step_index": "*",
                                "substep_index": "*",
                                "role_name": "*",
                                "content_regex": "*",
                                "tool_name": "*",
                            },
                            "check_hint": "Compare the claim against available observations and task requirements.",
                            "examples": {
                                "pass_scenario": "The agent cites a relevant observation before deciding.",
                                "fail_scenario": "The agent draws a conclusion unsupported by prior evidence.",
                            },
                            "check_type": "nl_check",
                            "python_check": {},
                            "nl_check": {
                                "judge_scope_notes": "Use only the current event, prefix, and local window.",
                                "focus_steps_instruction": "Check the current event and the most relevant prior evidence.",
                                "judge_rubric": ["grounded in evidence", "no unsupported conclusion"],
                                "output_format_template": "Return SKIP, SAT, or VIOL with evidence.",
                            },
                        }
                    ]
                }
            )
        if "agentrx dynamic constraint synthesis stage" in lower:
            return json.dumps(
                {
                    "constraints": [
                        {
                            "id": "D1",
                            "assertion_name": "preserve_current_task_requirement",
                            "taxonomy_targets": ["Instruction/Plan Adherence Failure"],
                            "constraint_type": "TEMPORAL",
                            "event_trigger": {
                                "step_index": "*",
                                "substep_index": "*",
                                "role_name": "*",
                                "content_regex": "*",
                                "tool_name": "*",
                            },
                            "check_hint": "Check whether the current step changes or ignores the task.",
                            "examples": {
                                "pass_scenario": "The step preserves the active task requirement.",
                                "fail_scenario": "The step drops or changes an active task requirement.",
                            },
                            "check_type": "nl_check",
                            "python_check": {},
                            "nl_check": {
                                "judge_scope_notes": "Compare the current step to the task and prefix.",
                                "focus_steps_instruction": "Check the current event and earlier task-setting events.",
                                "judge_rubric": ["preserves active requirement", "does not ignore required work"],
                                "output_format_template": "Return SKIP, SAT, or VIOL with evidence.",
                            },
                        }
                    ]
                }
            )
        if "agentrx guarded constraint evaluation stage" in lower:
            current_step = current_step_from_prompt(prompt) or suspected_step
            current_agent = agent_for_step(prompt, current_step) or agent
            has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            violations = []
            checks = []
            if has_error:
                violation = {
                    "step": current_step,
                    "agent": current_agent,
                    "assertion_name": "ground_claims_in_observations",
                    "constraint_type": "PROVENANCE",
                    "check_type": "nl_check",
                    "severity": 0.8,
                    "evidence": "Mock suspicious keyword.",
                    "taxonomy_targets": ["Misinterpretation of Tool Output / Handoff Failure"],
                    "reason": "Mock guarded evaluator found a suspicious step.",
                }
                violations.append(violation)
                checks.append(
                    {
                        "constraint_id": "G1",
                        "assertion_name": "ground_claims_in_observations",
                        "trigger_applies": True,
                        "verdict": "VIOL",
                        "evidence": "Mock suspicious keyword.",
                        "taxonomy_targets": ["Misinterpretation of Tool Output / Handoff Failure"],
                        "reason": "Mock guarded evaluator found a suspicious step.",
                    }
                )
            return json.dumps({"step_index": current_step, "checks": checks, "violations": violations})
        if "agentrx-like diagnostic baseline" in lower:
            return json.dumps(
                {
                    "constraints": [
                        {
                            "id": "ARX1",
                            "type": "evidence",
                            "description": "Agent claims must be grounded in relevant evidence or tool outputs.",
                            "violation_criteria": "The agent relies on irrelevant, unsupported, or contradictory evidence.",
                        },
                        {
                            "id": "ARX2",
                            "type": "task",
                            "description": "Agent actions must preserve the user's task requirements.",
                            "violation_criteria": "The agent changes or ignores the requested task.",
                        },
                    ]
                }
            )
        if "choosing among a2p counterfactual failure candidates" in lower:
            candidate_id, candidate_step, candidate_agent = first_candidate_from_prompt(prompt)
            return json.dumps(
                {
                    "candidate_id": candidate_id,
                    "step": candidate_step,
                    "agent": candidate_agent or agent,
                    "causal_score": 0.8 if candidate_step is not None else 0.2,
                    "reason": "Mock A2P reranker chose the earliest candidate.",
                }
            )
        if "a2p scaffolding instructions" in lower and "candidate attributions" in lower:
            candidate_id, candidate_step, candidate_agent = first_candidate_from_prompt(prompt)
            return json.dumps(
                {
                    "candidate_id": candidate_id,
                    "step": candidate_step,
                    "agent": candidate_agent or agent,
                    "causal_score": 0.8 if candidate_step is not None else 0.2,
                    "reason": "Mock A2P scaffold reranker chose the earliest candidate.",
                }
            )
        if "a2p scaffolding instructions" in lower and "chunk_id, contains_counterfactual_error" in lower:
            return json.dumps(
                {
                    "chunk_id": None,
                    "contains_counterfactual_error": suspected_step is not None,
                    "agent": agent if suspected_step is not None else None,
                    "step": suspected_step,
                    "abduction": "Mock A2P scaffold hidden root cause.",
                    "action": "Mock A2P scaffold minimal corrected action.",
                    "prediction": "Mock A2P scaffold predicts the correction would resolve the failure.",
                    "would_fix_failure": suspected_step is not None,
                    "causal_score": 0.8 if suspected_step is not None else 0.1,
                    "causal_mechanism": "Mock causal mechanism.",
                    "reason": "Mock A2P scaffold chunk evaluation.",
                }
            )
        if "a2p scaffolding instructions" in lower and "agent, step, abduction" in lower:
            return json.dumps(
                {
                    "agent": agent if suspected_step is not None else None,
                    "step": suspected_step,
                    "abduction": "Mock A2P scaffold hidden root cause.",
                    "action": "Mock A2P scaffold minimal corrected action.",
                    "prediction": "Mock A2P scaffold predicts the correction would resolve the failure.",
                    "would_fix_failure": suspected_step is not None,
                    "causal_score": 0.8 if suspected_step is not None else 0.1,
                    "causal_mechanism": "Mock causal mechanism.",
                    "reason": "Mock A2P scaffold attribution.",
                }
            )
        if "a2p scaffolding instructions" in lower and "candidate id:" in lower:
            candidate_id, candidate_step, candidate_agent = first_candidate_from_prompt(prompt)
            return (
                f"Candidate ID: {candidate_id if candidate_id is not None else 1}\n"
                f"Agent Name: {candidate_agent or agent}\n"
                f"Step Number: {candidate_step if candidate_step is not None else (suspected_step if suspected_step is not None else 0)}\n"
                "Causal Score: 0.8\n"
                "Reason for Mistake: Mock official A2P reranker chose the first candidate."
            )
        if "a2p scaffolding instructions" in lower and "chunk contains root cause:" in lower:
            contains = suspected_step is not None
            return (
                f"Chunk Contains Root Cause: {'Yes' if contains else 'No'}\n"
                f"Agent Name: {agent if contains else 'NONE'}\n"
                f"Step Number: {suspected_step if contains else -1}\n"
                f"Causal Score: {0.8 if contains else 0.0}\n"
                "Reason for Mistake: Mock official A2P chunk response selected the first suspicious step."
            )
        if "a2p scaffolding instructions" in lower and "segment contains root cause:" in lower:
            contains = suspected_step is not None
            return (
                f"Segment Contains Root Cause: {'Yes' if contains else 'No'}\n"
                f"Agent Name: {agent if contains else 'NONE'}\n"
                f"Step Number: {suspected_step if contains else -1}\n"
                f"Causal Score: {0.8 if contains else 0.0}\n"
                "Reason for Mistake: Mock official A2P reread response selected the first suspicious step."
            )
        if "a2p scaffolding instructions" in lower and "candidate is root cause:" in lower:
            current_step = current_step_under_review(prompt)
            current_agent = agent_for_step(prompt, current_step) or agent
            current_has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            return (
                f"Candidate Is Root Cause: {'Yes' if current_has_error else 'No'}\n"
                f"Agent Name: {current_agent if current_has_error else 'NONE'}\n"
                f"Step Number: {current_step if current_has_error else -1}\n"
                f"Causal Score: {0.8 if current_has_error else 0.0}\n"
                "Reason for Mistake: Mock official A2P step scan."
            )
        if "chunk table:" in lower and '"selected_chunk_ids"' in lower:
            chunk_ranges = [
                (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                for match in re.finditer(r"Chunk\s+(\d+):\s+steps\s+(-?\d+)-(-?\d+)", prompt, re.IGNORECASE)
            ]
            beam_match = re.search(r"select (?:exactly )?the top\s+(\d+)\s+chunk", prompt, re.IGNORECASE)
            beam_k = int(beam_match.group(1)) if beam_match else min(2, len(chunk_ranges))
            selected_ids: list[int] = []
            if suspected_step is not None:
                selected_ids.extend(
                    chunk_id
                    for chunk_id, start_step, end_step in chunk_ranges
                    if start_step <= suspected_step <= end_step
                )
            selected_ids.extend(chunk_id for chunk_id, _, _ in chunk_ranges if chunk_id not in selected_ids)
            return (
                "<json>"
                + json.dumps(
                    {
                        "selected_chunk_ids": selected_ids[:beam_k],
                        "rationale": "Mock whole-trace router selected chunks containing suspicious evidence.",
                    }
                )
                + "</json>"
            )
        if "a2p scaffolding instructions" in lower and "critical output format" in lower:
            return (
                f"Agent Name: {agent}\n"
                f"Step Number: {suspected_step if suspected_step is not None else 0}\n"
                "Reason for Mistake: Mock official A2P text response selected the first suspicious step."
            )
        if "localizing a suspected chunk using the a2p" in lower:
            return json.dumps(
                {
                    "agent": agent,
                    "step": suspected_step,
                    "abduction": "Mock hidden cause for the suspicious action.",
                    "action": "Mock minimal corrected action.",
                    "prediction": "Mock prediction says the correction would resolve the failure.",
                    "would_fix_failure": suspected_step is not None,
                    "causal_score": 0.8 if suspected_step is not None else 0.1,
                    "reason": "Mock A2P step localization.",
                }
            )
        if "a2p-style failure attribution method" in lower and "current chunk:" in lower:
            return json.dumps(
                {
                    "chunk_id": None,
                    "contains_counterfactual_error": suspected_step is not None,
                    "agent": agent,
                    "step": suspected_step,
                    "abduction": "Mock hidden cause for the suspicious action.",
                    "action": "Mock minimal corrected action.",
                    "prediction": "Mock prediction says the correction would resolve the failure.",
                    "would_fix_failure": suspected_step is not None,
                    "causal_score": 0.8 if suspected_step is not None else 0.1,
                    "reason": "Mock A2P chunk evaluation.",
                }
            )
        if "a2p-style failure attribution method" in lower:
            return json.dumps(
                {
                    "agent": agent,
                    "step": suspected_step,
                    "abduction": "Mock hidden cause for the suspicious action.",
                    "action": "Mock minimal corrected action.",
                    "prediction": "Mock prediction says the correction would resolve the failure.",
                    "would_fix_failure": suspected_step is not None,
                    "causal_score": 0.8 if suspected_step is not None else 0.1,
                    "reason": "Mock A2P all-at-once attribution.",
                }
            )
        if "produce an auditable validation log" in lower:
            violations = []
            if suspected_step is not None:
                violations.append(
                    {
                        "step": suspected_step,
                        "agent": agent,
                        "violated_constraint": "ARX1",
                        "failure_category": "Evidence Grounding Failure",
                        "severity": 0.8,
                        "confidence": 0.8,
                        "recoverable": False,
                        "evidence": "Mock suspicious keyword.",
                        "reason": "Mock validation log found a suspicious step.",
                    }
                )
            return json.dumps({"chunk_id": None, "violations": violations, "chunk_summary": "Mock validation."})
        if "final judge in an agentrx-like diagnostic framework" in lower:
            log_step = validation_log_step(prompt) or suspected_step
            log_agent = validation_log_agent(prompt) or agent_for_step(prompt, log_step) or agent
            return json.dumps(
                {
                    "step": log_step,
                    "agent": log_agent,
                    "failure_category": "Evidence Grounding Failure",
                    "confidence": 0.8 if log_step is not None else 0.2,
                    "reason": "Mock AGENTRX-like judge chose the first validation violation.",
                }
            )
        if "expert failure-categorization judge" in lower and "failure taxonomy" in lower:
            step = validation_log_step(prompt) or suspected_step
            judge_agent = validation_log_agent(prompt) or agent_for_step(prompt, step) or agent
            return json.dumps(
                {
                    "reason_for_failure": "Mock AGENTRX official judge selected the first suspicious unresolved failure.",
                    "failure_case": 4,
                    "reason_for_index": "Mock decision procedure found the first unrecovered candidate.",
                    "index": step,
                    "agent": judge_agent,
                    "confidence": 0.8 if step is not None else 0.2,
                }
            )
        if "applying echo-style objective analysis as a first-pass chunk selector" in lower:
            return json.dumps(
                {
                    "chunk_id": None,
                    "contains_attribution_evidence": suspected_step is not None,
                    "agent": agent if suspected_step is not None else "NONE",
                    "step": suspected_step if suspected_step is not None else -1,
                    "confidence": 0.8 if suspected_step is not None else 0.0,
                    "reasoning": "Mock ECHO chunk selector found a suspicious step.",
                }
            )
        if "objective analysis agent conducting an impartial investigation" in lower:
            step = suspected_step
            echo_agent = agent_for_step(prompt, step) or agent
            return (
                "<json>"
                + json.dumps(
                    {
                        "analysis_summary": "Mock ECHO objective analysis.",
                        "agent_evaluations": [
                            {
                                "agent_name": echo_agent,
                                "step_index": step,
                                "error_likelihood": 0.8 if step is not None else 0.1,
                                "reasoning": "Mock analyst evidence.",
                                "evidence": "Mock suspicious keyword.",
                            }
                        ],
                        "primary_conclusion": {
                            "type": "single_agent",
                            "attribution": [echo_agent],
                            "mistake_step": step,
                            "confidence": 0.8 if step is not None else 0.1,
                            "reasoning": "Mock ECHO consensus candidate.",
                        },
                        "alternative_hypotheses": [],
                    }
                )
                + "</json>"
            )
        if "chunk contains responsible error:" in lower:
            contains = suspected_step is not None
            return (
                f"Chunk Contains Responsible Error: {'Yes' if contains else 'No'}\n"
                f"Agent Name: {agent if contains else 'NONE'}\n"
                f"Step Number: {suspected_step if contains else -1}\n"
                f"Causal Score: {0.8 if contains else 0.0}\n"
                "Reason for Mistake: Mock official Who&When chunk response selected the first suspicious step."
            )
        if "selected conversation chunks:" in lower and "agent name:" in lower and "reason for mistake:" in lower:
            return (
                f"Agent Name: {agent}\n"
                f"Step Number: {suspected_step if suspected_step is not None else 0}\n"
                "Reason for Mistake: Mock official Who&When joint reread selected the first suspicious selected step."
            )
        if "based on this conversation, please predict the following" in lower and "agent name:" in lower:
            return (
                f"Agent Name: {agent}\n"
                f"Step Number: {suspected_step if suspected_step is not None else 0}\n"
                "Reason for Mistake: Mock official Who&When all-at-once selected the first suspicious step."
            )
        if "the most recent step (" in lower and "respond only in the format" in lower:
            current_step = current_step_under_review(prompt)
            current_has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            return (
                f"1. {'Yes' if current_has_error else 'No'}.\n"
                f"2. Reason: Mock official Who&When step evaluation for step {current_step}."
            )
        if "upper half" in lower and "lower half" in lower and "responding with only" in lower:
            return "upper half"
        if "selected evidence chunks" in lower and "contains_decisive_error" in lower:
            return json.dumps(
                {
                    "contains_decisive_error": suspected_step is not None,
                    "agent": agent if suspected_step is not None else None,
                    "step": suspected_step,
                    "reason": "Mock Who&When joint reread found the first suspicious selected step.",
                    "confidence": 0.8 if suspected_step is not None else 0.2,
                }
            )
        if "blame_score" in lower and "rereading a high-scoring chunk" in lower:
            return json.dumps(
                {
                    "agent": agent if suspected_step is not None else "NONE",
                    "step": suspected_step,
                    "blame_score": 4 if suspected_step is not None else 0,
                }
            )
        if "blame_score" in lower and "given a chunk from a multi-agent execution trace" in lower:
            return json.dumps(
                {
                    "agent": agent if suspected_step is not None else "NONE",
                    "step": suspected_step,
                    "blame_score": 4 if suspected_step is not None else 0,
                }
            )
        if "blame_score" in lower and "root-cause constraint violation" in lower and "you are now reviewing chunk" in lower:
            return json.dumps(
                {
                    "agent": agent if suspected_step is not None else "NONE",
                    "step": suspected_step,
                    "blame_score": 4 if suspected_step is not None else 0,
                }
            )
        if "identify whether this chunk contains the decisive error" in lower and "do not output confidence" in lower:
            return json.dumps(
                {
                    "likely_contains_decisive_error": suspected_step is not None,
                    "agent": agent,
                    "step": suspected_step,
                    "reason": "Mock bool chunk vote found the first suspicious step.",
                }
            )
        if "identify whether this chunk contains the decisive error" in lower:
            return json.dumps(
                {
                    "likely_contains_decisive_error": suspected_step is not None,
                    "agent": agent,
                    "step": suspected_step,
                    "reason": "Mock chunk vote found the first suspicious step.",
                    "confidence": 0.8 if suspected_step is not None else 0.2,
                }
            )
        if "candidate step under review" in lower and "blame_score" in lower:
            current_step = current_step_under_review(prompt)
            current_agent = agent_for_step(prompt, current_step) or agent
            current_has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            return json.dumps(
                {
                    "agent": current_agent if current_has_error else "NONE",
                    "step": current_step if current_has_error else None,
                    "blame_score": 4 if current_has_error else 0,
                }
            )
        if "target-agent hybrid localization" in lower and "selected_chunk_ids" in lower:
            return json.dumps(
                {
                    "selected_chunk_ids": [1, 2],
                    "rationale": "Mock paper-hybrid global router selected the first chunks for target-agent rereading.",
                }
            )
        if "target-agent candidate steps" in lower and "hybrid failure attribution method" in lower:
            return json.dumps(
                {
                    "agent": agent,
                    "step": suspected_step,
                    "reason": "Mock paper-hybrid joint target-agent reread selected the first suspicious step.",
                }
            )
        if "candidate step under review" in lower and "do not output confidence" in lower:
            current_step = current_step_under_review(prompt)
            current_agent = agent_for_step(prompt, current_step) or agent
            current_has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            return json.dumps(
                {
                    "contains_error": current_has_error,
                    "agent": current_agent,
                    "step": current_step,
                    "reason": "Mock bool step scan.",
                }
            )
        if "name of the agent who made a mistake that should be directly responsible" in lower:
            return json.dumps(
                {
                    "agent_name": agent,
                    "step_number": suspected_step,
                    "reason": "Mock TraceElephant All-at-Once selected the first suspicious step.",
                }
            )
        if "identify which agent made the decisive error" in lower:
            return json.dumps(
                {
                    "agent": agent,
                    "step": suspected_step,
                    "reason": "Mock all-at-once found the first suspicious step.",
                    "confidence": 0.8 if suspected_step is not None else 0.2,
                }
            )
        if "most recent step under review" in lower or "candidate step under review" in lower:
            current_step = current_step_under_review(prompt)
            current_agent = agent_for_step(prompt, current_step) or agent
            current_has_error = current_step is not None and step_has_suspicious_text(prompt, current_step)
            return json.dumps(
                {
                    "contains_error": current_has_error,
                    "agent": current_agent,
                    "step": current_step,
                    "reason": "Mock step scan.",
                    "confidence": 0.8 if current_has_error else 0.2,
                }
            )
        if "choose whether the earliest decisive error" in lower and "earlier half" in lower and "later half" in lower:
            earlier_range = range_after_label(prompt, "Earlier Half:")
            if suspected_step is not None and earlier_range and earlier_range[0] <= suspected_step <= earlier_range[1]:
                half = "earlier"
            elif suspected_step is not None:
                half = "later"
            else:
                half = "earlier"
            return json.dumps({"half": half, "reason": "Mock binary-search choice.", "confidence": 0.7})
        if "generate task-specific constraints" in lower:
            return json.dumps(
                {
                    "constraints": [
                        {
                            "id": "C1",
                            "type": "evidence",
                            "description": "Use the correct and verified source before deriving the answer.",
                            "violation_criteria": "An agent uses unrelated, irrelevant, or unverified evidence.",
                        },
                        {
                            "id": "C2",
                            "type": "calculation",
                            "description": "Preserve correct units and numerical conversions.",
                            "violation_criteria": "An agent changes units or numbers without justification.",
                        },
                    ]
                }
            )
        if "constraint-guided failure localization" in lower and "full conversation" in lower:
            return json.dumps(
                {
                    "step": suspected_step,
                    "agent": agent,
                    "violated_constraint": "C1",
                    "violation_type": "evidence",
                    "confidence": 0.8 if suspected_step is not None else 0.2,
                    "reason": "Mock full-trace CCV selected the first suspicious step.",
                }
            )
        if (
            "directly attributing a failed multi-agent conversation" in lower
            and "task-success requirements" in lower
            and "full conversation" in lower
        ):
            return json.dumps(
                {
                    "agent": agent,
                    "step": suspected_step,
                    "reason": "Mock direct attribution selected the first suspicious step.",
                    "confidence": 0.8 if suspected_step is not None else 0.2,
                }
            )
        if "rereading selected evidence chunks" in lower and "constraint-guided failure localization" in lower:
            return json.dumps(
                {
                    "step": suspected_step,
                    "agent": agent,
                    "violated_constraint": "C1",
                    "violation_type": "evidence",
                    "reason": "Mock joint selected-chunk reread selected the first suspicious step.",
                }
            )
        if "comparing two candidate" in lower:
            a_match = re.search(r"Candidate A:\s*Step\s+(-?\d+)", prompt, re.IGNORECASE)
            b_match = re.search(r"Candidate B:\s*Step\s+(-?\d+)", prompt, re.IGNORECASE)
            a_step = int(a_match.group(1)) if a_match else 10**9
            b_step = int(b_match.group(1)) if b_match else 10**9
            return json.dumps({"winner": "A" if a_step <= b_step else "B", "confidence": 0.6, "reason": "Mock chooses the earlier candidate."})
        if (
            (
                "localizing the first constraint violation" in lower
                or "localizing the first decisive constraint violation" in lower
                or "localizing the first a2p-decisive constraint violation" in lower
            )
            and "do not output confidence" in lower
        ):
            return json.dumps(
                {
                    "step": suspected_step,
                    "agent": agent,
                    "violated_constraint": "C1",
                    "violation_type": "evidence",
                    "reason": "Mock bool constraint localization.",
                }
            )
        if (
            "localizing the first constraint violation" in lower
            or "localizing the first decisive constraint violation" in lower
            or "localizing the first a2p-decisive constraint violation" in lower
        ):
            return json.dumps(
                {
                    "step": suspected_step,
                    "agent": agent,
                    "violated_constraint": "C1",
                    "violation_type": "evidence",
                    "confidence": 0.8,
                    "reason": "Mock found a suspicious evidence-related keyword.",
                }
            )
        if (
            (
                "choosing among candidate constraint-violation failure steps" in lower
                or "choosing among candidate a2p-decisive constraint-violation failure steps" in lower
            )
            and "do not output confidence" in lower
        ):
            candidate_id, candidate_step, candidate_agent = first_candidate_from_prompt(prompt)
            return json.dumps(
                {
                    "candidate_id": candidate_id,
                    "step": candidate_step,
                    "agent": candidate_agent or agent,
                    "reason": "Mock bool reranker chose the earliest candidate.",
                }
            )
        if (
            "choosing among candidate constraint-violation failure steps" in lower
            or "choosing among candidate a2p-decisive constraint-violation failure steps" in lower
        ):
            candidate_id, candidate_step, candidate_agent = first_candidate_from_prompt(prompt)
            return json.dumps(
                {
                    "candidate_id": candidate_id,
                    "step": candidate_step,
                    "agent": candidate_agent or agent,
                    "confidence": 0.8 if candidate_step is not None else 0.2,
                    "reason": "Mock reranker chose the earliest candidate.",
                }
            )
        if "single scalar root-cause score" in lower:
            contains = suspected_step is not None
            return json.dumps(
                {
                    "contains_violation": contains,
                    "earliest_suspected_step": suspected_step,
                    "agent": agent,
                    "root_cause_score": 0.8 if contains else 0.1,
                    "reason": "Mock scalar violation scoring.",
                }
            )
        if (
            "checking whether a chunk violates" in lower
            or "checking whether a chunk contains a decisive" in lower
            or "checking whether a focal chunk contains a decisive" in lower
            or "checking whether a chunk contains the decisive root-cause" in lower
            or "checking whether a chunk contains an a2p-decisive" in lower
            or "checking whether a focal chunk contains an a2p-decisive" in lower
        ):
            contains = suspected_step is not None
            return json.dumps(
                {
                    "contains_violation": contains,
                    "earliest_suspected_step": suspected_step,
                    "agent": agent,
                    "severity": 0.8 if contains else 0.1,
                    "irreversibility": 0.7 if contains else 0.1,
                    "evidence_strength": 0.75 if contains else 0.1,
                    "downstream_symptom_penalty": 0.1,
                    "confidence": 0.8 if contains else 0.2,
                    "reason": "Mock violation scoring.",
                }
            )
        if "localizing the earliest decisive error" in lower:
            contains = suspected_step is not None
            return json.dumps(
                {
                    "contains_decisive_error": contains,
                    "candidate_step": suspected_step,
                    "agent": agent,
                    "root_cause_score": 0.8 if contains else 0.1,
                    "is_downstream_symptom": False,
                    "confidence": 0.8 if contains else 0.2,
                    "reason": "Mock window refinement.",
                }
            )
        return json.dumps(
            {
                "likely_contains_decisive_error": suspected_step is not None,
                "onset_score": 0.8 if suspected_step is not None else 0.1,
                "causal_impact_score": 0.8 if suspected_step is not None else 0.1,
                "answer_contrast_score": 0.7 if suspected_step is not None else 0.1,
                "agent_specificity_score": 0.8 if suspected_step is not None else 0.1,
                "symptom_penalty": 0.1,
                "confidence": 0.8 if suspected_step is not None else 0.2,
                "candidate_steps": [
                    {"step": suspected_step, "agent": agent, "reason": "Mock suspicious keyword."}
                ]
                if suspected_step is not None
                else [],
            }
        )


class LocalHFLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = empty_usage()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "backend=local_hf requires torch and transformers in the Python environment."
            ) from exc

        dtype = dtype_from_name(torch, config.torch_dtype)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        if config.truncation_side:
            self.tokenizer.truncation_side = config.truncation_side
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = chat_messages(self.config, prompt)
        chat_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3 supports this switch; Llama tokenizers ignore neither unknown kwargs
        # nor always accept it, so retry without it if needed.
        if self.config.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self.config.enable_thinking
        try:
            text = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            text = self.tokenizer.apply_chat_template(messages, **chat_kwargs)

        tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.config.max_input_tokens > 0:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": self.config.max_input_tokens,
                }
            )
        inputs = self.tokenizer([text], **tokenizer_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature if self.config.temperature > 0 else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        try:
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_kwargs)
        except Exception:
            record_usage(self, input_token_count, 0)
            raise
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        output_token_count = int(new_tokens.shape[-1])
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        record_usage(self, input_token_count, output_token_count)
        return response


class OllamaLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = empty_usage()

    def generate(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": chat_messages(self.config, prompt),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        data = post_json(url, payload, timeout=self.config.timeout_seconds)
        response = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count") or estimated_token_count(rendered_prompt_for_usage(self.config, prompt))
        output_tokens = data.get("eval_count") or estimated_token_count(response)
        record_usage(self, input_tokens, output_tokens)
        return response


class OpenAICompatibleLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = empty_usage()
        self.api_key = config.api_key or (os.environ.get(config.api_key_env) if config.api_key_env else "")

    def generate(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        token_param = openai_token_limit_param(self.config)
        payload = self._payload(prompt, token_param)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.config.openrouter_referrer:
            headers["HTTP-Referer"] = self.config.openrouter_referrer
        if self.config.openrouter_title:
            headers["X-OpenRouter-Title"] = self.config.openrouter_title
        for _ in range(3):
            try:
                data = post_json(url, payload, headers=headers, timeout=self.config.timeout_seconds)
                break
            except RuntimeError as exc:
                retry_payload = openai_retry_payload(payload, str(exc))
                if retry_payload is None:
                    raise
                payload = retry_payload
        else:
            data = post_json(url, payload, headers=headers, timeout=self.config.timeout_seconds)
        response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens") or estimated_token_count(rendered_prompt_for_usage(self.config, prompt))
        output_tokens = usage.get("completion_tokens") or estimated_token_count(response)
        record_usage(self, input_tokens, output_tokens)
        return response

    def _payload(self, prompt: str, token_param: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": chat_messages(self.config, prompt),
            token_param: self.config.max_tokens,
        }
        if not self.config.omit_temperature:
            payload["temperature"] = self.config.temperature
        provider = openrouter_provider_preferences(self.config)
        if provider:
            payload["provider"] = provider
        return payload


class AnthropicLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = empty_usage()
        self.api_key = config.api_key or (os.environ.get(config.api_key_env) if config.api_key_env else "")

    def generate(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") or "https://api.anthropic.com"
        url = url + "/v1/messages"
        payload = self._payload(prompt)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.config.anthropic_version,
        }
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Configure it as an environment variable before running."
            )
        for _ in range(3):
            try:
                data = post_json(url, payload, headers=headers, timeout=self.config.timeout_seconds)
                break
            except RuntimeError as exc:
                retry_payload = anthropic_retry_payload(payload, str(exc))
                if retry_payload is None:
                    raise
                payload = retry_payload
        else:
            data = post_json(url, payload, headers=headers, timeout=self.config.timeout_seconds)
        response = anthropic_response_text(data)
        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens") or estimated_token_count(rendered_prompt_for_usage(self.config, prompt))
        output_tokens = usage.get("output_tokens") or estimated_token_count(response)
        record_usage(self, input_tokens, output_tokens)
        return response

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.system_prompt:
            payload["system"] = self.config.system_prompt
        if not self.config.omit_temperature:
            payload["temperature"] = self.config.temperature
        return payload


class LlamaCppCliLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = empty_usage()

    def generate(self, prompt: str) -> str:
        full_prompt = rendered_prompt_for_usage(self.config, prompt)
        cmd = [
            self.config.llama_cli_path,
            "-m",
            self.config.model,
            "-p",
            full_prompt,
            "-n",
            str(self.config.max_tokens),
            "--temp",
            str(self.config.temperature),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )
        response = completed.stdout
        record_usage(self, estimated_token_count(full_prompt), estimated_token_count(response))
        return response


def empty_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def record_usage(llm: Any, input_tokens: Any, output_tokens: Any, calls: int = 1) -> None:
    if not hasattr(llm, "usage"):
        llm.usage = empty_usage()
    llm.usage["calls"] = int(llm.usage.get("calls", 0)) + int(calls)
    llm.usage["input_tokens"] = int(llm.usage.get("input_tokens", 0)) + safe_int(input_tokens)
    llm.usage["output_tokens"] = int(llm.usage.get("output_tokens", 0)) + safe_int(output_tokens)


def get_usage_snapshot(llm: Any) -> dict[str, int]:
    usage = getattr(llm, "usage", None)
    if not isinstance(usage, dict):
        return empty_usage()
    return {key: safe_int(usage.get(key, 0)) for key in USAGE_KEYS}


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta = {key: max(0, safe_int(after.get(key, 0)) - safe_int(before.get(key, 0))) for key in USAGE_KEYS}
    delta["total_tokens"] = delta["input_tokens"] + delta["output_tokens"]
    return delta


def estimated_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def openrouter_provider_preferences(config: LLMConfig) -> dict[str, Any]:
    provider: dict[str, Any] = {}
    if config.provider_order:
        provider["order"] = list(config.provider_order)
    if config.provider_only:
        provider["only"] = list(config.provider_only)
    if config.provider_sort:
        provider["sort"] = config.provider_sort
    if config.provider_allow_fallbacks is not None:
        provider["allow_fallbacks"] = config.provider_allow_fallbacks
    if config.provider_require_parameters is not None:
        provider["require_parameters"] = config.provider_require_parameters
    return provider


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"LLM request failed for {url}: HTTP {exc.code}: {body_text}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == 3:
                raise last_error from exc
            time.sleep(2 * attempt)
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"LLM request failed for {url}: {exc}")
            if attempt == 3:
                raise last_error from exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"LLM request failed for {url}: {last_error}")


def openai_token_limit_param(config: LLMConfig) -> str:
    override = config.token_limit_param.strip()
    if override in {"max_tokens", "max_completion_tokens"}:
        return override
    model = config.model.lower()
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def openai_retry_token_limit_param(error_text: str, current_param: str) -> str | None:
    lower = error_text.lower()
    if "unsupported parameter" not in lower:
        return None
    if current_param == "max_tokens" and "max_completion_tokens" in lower:
        return "max_completion_tokens"
    if current_param == "max_completion_tokens" and "max_tokens" in lower:
        return "max_tokens"
    return None


def openai_retry_payload(payload: dict[str, Any], error_text: str) -> dict[str, Any] | None:
    lower = error_text.lower()
    current_param = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
    retry_param = openai_retry_token_limit_param(error_text, current_param)
    if retry_param is not None:
        updated = dict(payload)
        value = updated.pop(current_param)
        updated[retry_param] = value
        return updated
    if "temperature" in payload and ("unsupported parameter" in lower or "unsupported value" in lower) and "temperature" in lower:
        updated = dict(payload)
        updated.pop("temperature", None)
        return updated
    return None


def anthropic_response_text(data: dict[str, Any]) -> str:
    content = data.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def anthropic_retry_payload(payload: dict[str, Any], error_text: str) -> dict[str, Any] | None:
    lower = error_text.lower()
    if "temperature" in payload and ("unsupported" in lower or "invalid" in lower) and "temperature" in lower:
        updated = dict(payload)
        updated.pop("temperature", None)
        return updated
    if "max_tokens" in payload and "max_tokens" in lower and ("too" in lower or "exceed" in lower):
        updated = dict(payload)
        updated["max_tokens"] = max(256, int(updated["max_tokens"]) // 2)
        if updated["max_tokens"] != payload["max_tokens"]:
            return updated
    return None


def dtype_from_name(torch: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized == "auto":
        return "auto"
    raise ValueError(f"Unsupported torch_dtype for local_hf: {name}")


def first_suspicious_step(prompt: str) -> int | None:
    suspicious = [
        "unrelated",
        "irrelevant",
        "wrong",
        "incorrect",
        "mistake",
        "error",
        "unverified",
        "unsupported",
        "contradict",
    ]
    current_step: int | None = None
    for line in relevant_log_lines(prompt):
        match = re.match(r"\s*Step\s+(-?\d+)(?:\s+-\s*.*)?\s*$", line, flags=re.IGNORECASE)
        if match:
            current_step = int(match.group(1))
            if any(word in line.lower() for word in suspicious):
                return current_step
            continue
        if current_step is not None and any(word in line.lower() for word in suspicious):
            return current_step
    return None


def relevant_log_lines(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    start_markers = [
        "Current Chunk:",
        "Chunk:",
        "Log Window:",
        "Chunk Log:",
        "Chunk Under Validation:",
        "Conversation:",
        "History Up To Current Step:",
        "History Up To Current Candidate Step:",
        "Segment Under Review:",
    ]
    stop_markers = [
        "Next Chunk Summary:",
        "Immediate Previous Context:",
        "Immediate Next Context:",
        "Most Recent Step Under Review:",
        "Candidate Step Under Review:",
        "Earlier Half:",
        "Task:",
        "Return JSON",
    ]
    for idx, line in enumerate(lines):
        if line.strip() in start_markers:
            selected: list[str] = []
            for candidate in lines[idx + 1 :]:
                if candidate.strip() in stop_markers:
                    break
                selected.append(candidate)
            return selected
    return lines


def current_step_under_review(prompt: str) -> int | None:
    patterns = [
        r"Most Recent Step Under Review:\s*Step\s+(-?\d+)",
        r"Candidate Step Under Review:\s*Step\s+(-?\d+)",
        r"The most recent step\s*\((-?\d+)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def current_step_from_prompt(prompt: str) -> int | None:
    patterns = [
        r"Current Step sk:\s*Step\s+(-?\d+)",
        r"Current Step:\s*Step\s+(-?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return current_step_under_review(prompt)


def step_has_suspicious_text(prompt: str, step: int) -> bool:
    suspicious = [
        "unrelated",
        "irrelevant",
        "wrong",
        "incorrect",
        "mistake",
        "error",
        "unverified",
        "unsupported",
        "contradict",
    ]
    lines = prompt.splitlines()
    in_step = False
    for line in lines:
        if line.strip() in {
            "Most Recent Step Under Review:",
            "Candidate Step Under Review:",
            "Task:",
            "Return JSON only with keys:",
        }:
            in_step = False
        if re.match(r"\s*Step\s+(-?\d+)(?:\s+-\s*.*)?\s*$", line, flags=re.IGNORECASE):
            match = re.match(r"\s*Step\s+(-?\d+)(?:\s+-\s*.*)?\s*$", line, flags=re.IGNORECASE)
            in_step = match is not None and int(match.group(1)) == step
            if in_step and any(word in line.lower() for word in suspicious):
                return True
            continue
        if in_step and any(word in line.lower() for word in suspicious):
            return True
    return False


def range_after_label(prompt: str, label: str) -> tuple[int, int] | None:
    idx = prompt.lower().find(label.lower())
    if idx < 0:
        return None
    match = re.search(r"Steps\s+(-?\d+)-(-?\d+)", prompt[idx:], flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def validation_log_step(prompt: str) -> int | None:
    match = re.search(r"['\"]step['\"]\s*:\s*(-?\d+)", prompt)
    if match:
        return int(match.group(1))
    return None


def validation_log_agent(prompt: str) -> str | None:
    match = re.search(r"['\"]agent['\"]\s*:\s*['\"]([^'\"]+)['\"]", prompt)
    if match:
        return match.group(1).strip()
    return None


def first_candidate_from_prompt(prompt: str) -> tuple[int | None, int | None, str | None]:
    pattern = re.compile(
        r"['\"]candidate_id['\"]\s*:\s*(?P<cid>\d+).*?"
        r"['\"]step['\"]\s*:\s*(?P<step>-?\d+).*?"
        r"['\"]agent['\"]\s*:\s*['\"](?P<agent>[^'\"]*)['\"]",
        flags=re.DOTALL,
    )
    matches = [
        (int(match.group("cid")), int(match.group("step")), match.group("agent").strip() or None)
        for match in pattern.finditer(prompt)
    ]
    if not matches:
        return None, None, None
    return sorted(matches, key=lambda item: item[1])[0]


def agent_for_step(prompt: str, step: int | None) -> str | None:
    if step is None:
        return None
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        inline_match = re.match(rf"\s*Step\s+{re.escape(str(step))}\s+-\s*([^:]+):", line, flags=re.IGNORECASE)
        if inline_match:
            return inline_match.group(1).strip()
        if re.match(rf"\s*Step\s+{re.escape(str(step))}\s*$", line, flags=re.IGNORECASE):
            for next_line in lines[i + 1 : i + 5]:
                match = re.match(r"\s*Agent:\s*(.+)\s*$", next_line, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()
    return None
