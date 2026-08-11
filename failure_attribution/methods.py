from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import Any

from .chunking import (
    estimate_steps_tokens,
    expand_region,
    make_adaptive_budgeted_chunks,
    make_budgeted_chunks,
    make_chunks,
    render_steps,
    sliding_windows,
    split_long_steps,
    steps_before_after,
    summarize_chunk,
)
from .json_utils import as_bool, as_float, extract_json
from .llm import LLM, get_usage_snapshot, usage_delta
from .prompts import (
    a2p_all_at_once_prompt,
    a2p_chunk_context_prompt,
    a2p_chunk_prompt,
    a2p_official_agent_step_prompt,
    a2p_official_agent_joint_prompt,
    a2p_official_beam_joint_prompt,
    a2p_official_chunk_prompt,
    a2p_official_global_chunk_router_prompt,
    a2p_official_local_confidence_chunk_prompt,
    a2p_official_global_router_reread_prompt,
    a2p_official_prompt,
    a2p_official_reread_prompt,
    a2p_official_rerank_prompt,
    a2p_repo_exact_prompt,
    a2p_rerank_prompt,
    a2p_scaffold_all_at_once_prompt,
    a2p_scaffold_chunk_prompt,
    a2p_scaffold_reread_prompt,
    a2p_scaffold_rerank_prompt,
    a2p_step_prompt,
    agentrx_constraint_prompt,
    agentrx_judge_prompt,
    agentrx_original_dynamic_constraints_prompt,
    agentrx_original_global_constraints_prompt,
    agentrx_original_judge_prompt,
    agentrx_original_step_validation_prompt,
    agentrx_official_judge_prompt,
    agentrx_official_wrapper_judge_prompt,
    agentrx_validation_prompt,
    all_at_once_prompt,
    task_only_all_at_once_prompt,
    binary_search_prompt,
    ccv_chunk_prompt,
    ccv_chunk_context_prompt,
    ccv_beam_rerank_prompt,
    ccv_beam_rerank_bool_prompt,
    ccv_constraint_prompt,
    ccv_checklist_equivalent_full_trace_prompt,
    ccv_checklist_equivalent_generation_prompt,
    ccv_causal_no_requirements_prompt,
    ccv_constraint_only_full_trace_prompt,
    ccv_full_trace_prompt,
    ccv_requirements_direct_prompt,
    ccv_trace_elephant_full_trace_prompt,
    direct_requirements_full_trace_prompt,
    tsr_direct_no_requirements_prompt,
    tsr_minimal_r0_prompt,
    tsr_minimal_r1_prompt,
    ccv_global_chunk_router_prompt,
    ccv_ordinal_chunk_prompt,
    ccv_scalar_chunk_prompt,
    ccv_selected_chunks_joint_prompt,
    ccv_step_context_prompt,
    ccv_step_bool_prompt,
    ccv_step_prompt,
    task_checklist_full_trace_prompt,
    task_checklist_generation_prompt,
    trace_elephant_official_all_at_once_prompt,
    reference_checklist_full_trace_prompt,
    reference_checklist_generation_prompt,
    cgv_final_judge_prompt,
    cgv_step_validation_prompt,
    echo_chunk_ranking_prompt,
    echo_appendix_conversation_summary,
    echo_appendix_strict_conversation_summary,
    echo_appendix_strict_objective_analysis_prompt,
    echo_objective_analysis_prompt,
    echo_original_conversation_summary,
    echo_original_global_chunk_router_prompt,
    echo_original_objective_analysis_prompt,
    mvbs_chunk_scoring_prompt,
    mvbs_pairwise_prompt,
    mvbs_window_prompt,
    chunk_all_at_once_prompt,
    chunk_all_at_once_context_prompt,
    chunk_bool_prompt,
    chunk_ordinal_prompt,
    chunk_ordinal_reread_prompt,
    paper_hybrid_step_bool_prompt,
    paper_hybrid_step_context_prompt,
    paper_hybrid_step_ordinal_prompt,
    paper_hybrid_step_prompt,
    paper_hybrid_global_chunk_router_prompt,
    paper_hybrid_target_agent_joint_prompt,
    render_steps_a2p_official,
    render_who_when_official_steps,
    step_by_step_prompt,
    who_when_beam_joint_prompt,
    who_when_pro_all_at_once_prompt,
    who_when_official_all_at_once_prompt,
    who_when_official_beam_joint_prompt,
    who_when_official_binary_search_prompt,
    who_when_official_chunk_prompt,
    who_when_official_global_chunk_router_prompt,
    who_when_official_global_router_joint_prompt,
    who_when_official_step_by_step_prompt,
)
from .schema import Case, LogStep, Prediction
from .shared_cache import case_fingerprint, constraint_fingerprint, load_ccv_constraint_cache


def run_method(name: str, case: Case, llm: LLM, config: dict[str, Any]) -> Prediction:
    name = name.lower()
    if name in {"all_at_once", "all-at-once"}:
        return run_all_at_once(case, llm)
    if name in {"task_only_all_at_once", "task-only-all-at-once", "direct_no_gt"}:
        return run_task_only_all_at_once(case, llm)
    if name in {
        "trace_elephant_official_all_at_once",
        "trace-elephant-official-all-at-once",
        "te_official_all_at_once",
    }:
        method_config = config.get("methods", {}).get("trace_elephant_official_all_at_once", {})
        return run_trace_elephant_official_all_at_once(case, llm, method_config)
    if name in {"step_by_step", "step-by-step"}:
        return run_step_by_step(case, llm)
    if name in {"binary_search", "binary-search"}:
        return run_binary_search(case, llm)
    if name in {"who_when_official_all_at_once", "ww_official_all_at_once", "ww-all-at-once-official"}:
        return run_who_when_official_all_at_once(case, llm)
    if name in {"who_when_pro_all_at_once", "ww_pro_all_at_once", "ww-pro-all-at-once"}:
        return run_who_when_pro_all_at_once(case, llm)
    if name in {"who_when_official_step_by_step", "ww_official_step_by_step", "ww-step-by-step-official"}:
        return run_who_when_official_step_by_step(case, llm)
    if name in {"who_when_official_binary_search", "ww_official_binary_search", "ww-binary-search-official"}:
        return run_who_when_official_binary_search(case, llm)
    if name in {"paper_hybrid", "hybrid"}:
        return run_paper_hybrid(case, llm)
    if name in {"chunk_vote10", "chunk-vote10"}:
        return run_chunk_vote10(case, llm, config.get("methods", {}).get("chunk_vote10", {}))
    if name in {"chunk_vote_simple10", "chunk-vote-simple10", "chunk_bool10", "chunk-bool10"}:
        return run_chunk_vote_simple10(case, llm, config.get("methods", {}).get("chunk_vote_simple10", {}))
    if name in {"chunk_vote_ordinal_reread10", "chunk-vote-ordinal-reread10", "chunk_vote_reread10", "chunk-reread10"}:
        return run_chunk_vote_ordinal_reread10(
            case,
            llm,
            config.get("methods", {}).get("chunk_vote_ordinal_reread10", {}),
        )
    if name in {
        "who_when_beam_joint",
        "who_and_when_beam_joint",
        "ww_beam_joint",
        "who-when-beam-joint",
        "who-and-when-beam-joint",
    }:
        return run_who_when_beam_joint(case, llm, config.get("methods", {}).get("who_when_beam_joint", {}))
    if name in {
        "who_when_official_beam_joint",
        "who_and_when_official_beam_joint",
        "ww_official_beam_joint",
        "who-when-official-beam-joint",
        "who-and-when-official-beam-joint",
    }:
        return run_who_when_official_beam_joint(
            case,
            llm,
            config.get("methods", {}).get("who_when_official_beam_joint", {}),
        )
    if name in {
        "who_when_official_global_router_beam_joint",
        "who_and_when_official_global_router_beam_joint",
        "ww_official_global_router_beam_joint",
        "who-when-official-global-router-beam-joint",
        "who-and-when-official-global-router-beam-joint",
    }:
        method_config = config.get("methods", {}).get("who_when_official_global_router_beam_joint")
        if method_config is None:
            method_config = config.get("methods", {}).get("who_when_official_beam_joint", {})
        return run_who_when_official_global_router_beam_joint(case, llm, method_config)
    if name in {"paper_hybrid10", "hybrid10", "paper-hybrid10"}:
        return run_paper_hybrid10(case, llm, config.get("methods", {}).get("paper_hybrid10", {}))
    if name in {"paper_hybrid_adaptive", "hybrid_adaptive", "paper-hybrid-adaptive"}:
        return run_paper_hybrid_adaptive(case, llm, config.get("methods", {}).get("paper_hybrid_adaptive", {}))
    if name in {"paper_hybrid_adaptive_context", "hybrid_adaptive_context", "paper-hybrid-adaptive-context"}:
        return run_paper_hybrid_adaptive_context(
            case,
            llm,
            config.get("methods", {}).get("paper_hybrid_adaptive_context", {}),
        )
    if name in {
        "paper_hybrid_global_router",
        "paper_hybrid_caw_global_router",
        "paper-hybrid-global-router",
        "paper_hybrid_router",
    }:
        return run_paper_hybrid_global_router(
            case,
            llm,
            config.get("methods", {}).get("paper_hybrid_global_router", {}),
        )
    if name in {"a2p_original", "original_a2p", "a2p", "a2p_all_at_once", "a2p-all-at-once"}:
        return run_a2p_original(case, llm)
    if name in {"a2p_official", "a2p_official_like", "a2p-reproduction", "a2p_reproduction"}:
        return run_a2p_official(case, llm)
    if name in {"a2p_repo_exact", "a2p-repo-exact"}:
        return run_a2p_repo_exact(case, llm)
    if name in {
        "a2p_scaffold_original",
        "a2p_original_scaffold",
        "a2p-scaffold-original",
        "a2p_scaffold_all_at_once",
    }:
        return run_a2p_scaffold_original(case, llm)
    if name in {"a2p_adaptive", "a2p-adaptive", "adaptive_a2p"}:
        return run_a2p_adaptive(case, llm, config.get("methods", {}).get("a2p_adaptive", {}))
    if name in {"a2p_adaptive_beam", "a2p_beam_adaptive", "adaptive_a2p_beam", "a2p-adaptive-beam"}:
        return run_a2p_adaptive_beam(case, llm, config.get("methods", {}).get("a2p_adaptive_beam", {}))
    if name in {"a2p_official_beam", "a2p_official_wrapper_beam", "a2p-official-beam"}:
        return run_a2p_official_beam(case, llm, config.get("methods", {}).get("a2p_official_beam", {}))
    if name in {"a2p_official_beam_joint", "a2p_official_joint_beam", "a2p-official-beam-joint"}:
        return run_a2p_official_beam_joint(
            case,
            llm,
            config.get("methods", {}).get("a2p_official_beam_joint", {}),
        )
    if name in {
        "a2p_official_global_router_beam_joint",
        "a2p_official_router_beam_joint",
        "a2p-global-router-beam-joint",
        "a2p_official_whole_trace_router",
    }:
        method_config = config.get("methods", {}).get("a2p_official_global_router_beam_joint")
        if method_config is None:
            method_config = config.get("methods", {}).get("a2p_official_beam_joint", {})
        return run_a2p_official_global_router_beam_joint(case, llm, method_config)
    if name in {
        "a2p_official_local_confidence_beam_joint",
        "a2p_official_chunk_confidence_beam_joint",
        "a2p-local-confidence-beam-joint",
    }:
        method_config = config.get("methods", {}).get("a2p_official_local_confidence_beam_joint")
        if method_config is None:
            method_config = config.get("methods", {}).get("a2p_official_global_router_beam_joint", {})
        return run_a2p_official_local_confidence_beam_joint(case, llm, method_config)
    if name in {
        "a2p_official_agent_hybrid",
        "a2p_official_hybrid",
        "a2p-official-agent-hybrid",
        "a2p-official-hybrid",
    }:
        return run_a2p_official_agent_hybrid(
            case,
            llm,
            config.get("methods", {}).get("a2p_official_agent_hybrid", {}),
        )
    if name in {
        "a2p_official_agent_hybrid_joint",
        "a2p_official_joint_agent_hybrid",
        "a2p-official-agent-hybrid-joint",
        "a2p-official-joint-agent-hybrid",
    }:
        return run_a2p_official_agent_hybrid_joint(
            case,
            llm,
            config.get("methods", {}).get("a2p_official_agent_hybrid_joint", {}),
        )
    if name in {
        "a2p_scaffold_beam",
        "a2p_original_beam",
        "a2p_wrapper_beam",
        "a2p-scaffold-beam",
    }:
        return run_a2p_scaffold_beam(case, llm, config.get("methods", {}).get("a2p_scaffold_beam", {}))
    if name in {
        "a2p_scaffold_agent_hybrid",
        "a2p_scaffold_hybrid",
        "a2p-original-agent-hybrid",
        "a2p-scaffold-agent-hybrid",
    }:
        return run_a2p_scaffold_agent_hybrid(
            case,
            llm,
            config.get("methods", {}).get("a2p_scaffold_agent_hybrid", {}),
        )
    if name in {
        "a2p_beam_agent_hybrid",
        "a2p_agent_hybrid",
        "a2p_top3_paper_hybrid",
        "a2p-beam-agent-hybrid",
    }:
        return run_a2p_beam_agent_hybrid(case, llm, config.get("methods", {}).get("a2p_beam_agent_hybrid", {}))
    if name in {
        "a2p_beam_agent_hybrid_context",
        "a2p_agent_hybrid_context",
        "a2p_top3_paper_hybrid_context",
        "a2p-beam-agent-hybrid-context",
    }:
        return run_a2p_beam_agent_hybrid_context(
            case,
            llm,
            config.get("methods", {}).get("a2p_beam_agent_hybrid_context", {}),
        )
    if name in {"paper_hybrid_simple10", "hybrid_simple10", "paper-hybrid-simple10"}:
        return run_paper_hybrid_simple10(case, llm, config.get("methods", {}).get("paper_hybrid_simple10", {}))
    if name in {"paper_hybrid_ordinal10", "hybrid_ordinal10", "paper-hybrid-ordinal10"}:
        return run_paper_hybrid_ordinal10(case, llm, config.get("methods", {}).get("paper_hybrid_ordinal10", {}))
    if name in {"agentrx_official", "agentrx_original", "agentrx-paper", "agentrx_paper", "agentrx"}:
        return run_agentrx_original(case, llm, config.get("methods", {}).get("agentrx_original", {}))
    if name in {"agentrx_baseline", "agentrx_g1_baseline", "agentrx_official_baseline"}:
        return run_agentrx_baseline(case, llm)
    if name in {"agentrx_wrapper", "agentrx_adaptive", "agentrx_beam", "agentrx-official-wrapper"}:
        return run_agentrx_wrapper(case, llm, config.get("methods", {}).get("agentrx_wrapper", {}))
    if name in {"echo_official", "echo", "echo-paper", "echo_paper"}:
        return run_echo_official(case, llm, config.get("methods", {}).get("echo_official", {}))
    if name in {"echo_wrapper", "echo_adaptive", "echo_beam", "echo-official-wrapper"}:
        return run_echo_wrapper(case, llm, config.get("methods", {}).get("echo_wrapper", {}))
    if name in {
        "echo_original",
        "echo_original_official",
        "echo-paper-original",
        "echo_paper_original",
        "echo_i4",
        "echo_i4_original",
        "echo_full_original",
    }:
        return run_echo_original(case, llm, config.get("methods", {}).get("echo_original", {}))
    if name in {
        "echo_appendix_original",
        "echo_appendix_i4",
        "echo-appendix-original",
        "echo_paper_appendix",
    }:
        method_config = config.get("methods", {}).get("echo_appendix_original")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original", {})
        return run_echo_appendix_original(case, llm, method_config)
    if name in {
        "echo_appendix_strict_original",
        "echo_appendix_strict_i4",
        "echo-strict-original",
        "echo_i4_strict",
    }:
        method_config = config.get("methods", {}).get("echo_appendix_strict_original")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_appendix_original")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original", {})
        return run_echo_appendix_strict_original(case, llm, method_config)
    if name in {"echo_i3_original", "echo_objective_original", "echo_i3"}:
        return run_echo_i3_original(case, llm, config.get("methods", {}).get("echo_original", {}))
    if name in {"echo_original_wrapper", "echo_original_beam", "echo-original-wrapper", "echo_i4_wrapper"}:
        return run_echo_original_wrapper(case, llm, config.get("methods", {}).get("echo_original_wrapper", {}))
    if name in {
        "echo_original_global_router_wrapper",
        "echo_global_router_wrapper",
        "echo-whole-trace-router",
        "echo_original_router_wrapper",
    }:
        method_config = config.get("methods", {}).get("echo_original_global_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original_wrapper", {})
        return run_echo_original_global_router_wrapper(case, llm, method_config)
    if name in {
        "echo_original_panel_router_wrapper",
        "echo_panel_router_wrapper",
        "echo-panel-router-wrapper",
        "echo_i4_panel_router_wrapper",
    }:
        method_config = config.get("methods", {}).get("echo_original_panel_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original_global_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original_wrapper", {})
        return run_echo_original_panel_router_wrapper(case, llm, method_config)
    if name in {
        "echo_appendix_panel_router_wrapper",
        "echo_appendix_wrapper",
        "echo-appendix-panel-router-wrapper",
    }:
        method_config = config.get("methods", {}).get("echo_appendix_panel_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original_panel_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original", {})
        return run_echo_appendix_panel_router_wrapper(case, llm, method_config)
    if name in {
        "echo_appendix_strict_panel_router_wrapper",
        "echo_appendix_strict_wrapper",
        "echo-strict-panel-router-wrapper",
    }:
        method_config = config.get("methods", {}).get("echo_appendix_strict_panel_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_appendix_panel_router_wrapper")
        if method_config is None:
            method_config = config.get("methods", {}).get("echo_original", {})
        return run_echo_appendix_strict_panel_router_wrapper(case, llm, method_config)
    if name in {"agentrx10", "agentrx_like", "agentrx-like"}:
        return run_agentrx10(case, llm, config.get("methods", {}).get("agentrx10", {}))
    if name == "mvbs10":
        return run_mvbs10(case, llm, config.get("methods", {}).get("mvbs10", {}))
    if name == "ccv10":
        return run_ccv10(case, llm, config.get("methods", {}).get("ccv10", {}))
    if name in {"ccv_adaptive", "adaptive_ccv"}:
        return run_ccv_adaptive(case, llm, config.get("methods", {}).get("ccv_adaptive", {}))
    if name in {"ccv_adaptive_context", "adaptive_ccv_context"}:
        return run_ccv_adaptive_context(case, llm, config.get("methods", {}).get("ccv_adaptive_context", {}))
    if name in {"ccv_full_trace", "ccv_fulltrace", "ccv_all_at_once", "ccv_all_trace"}:
        return run_ccv_full_trace(case, llm, config.get("methods", {}).get("ccv_full_trace", {}))
    if name in {
        "tsr_r0c1_causal_no_requirements",
        "ccv_causal_no_requirements",
        "r0c1_causal_no_requirements",
    }:
        return run_ccv_causal_no_requirements(case, llm)
    if name in {
        "tsr_r1c0_requirements_direct",
        "ccv_requirements_direct",
        "r1c0_requirements_direct",
    }:
        return run_ccv_requirements_direct(
            case,
            llm,
            config.get("methods", {}).get("tsr_r1c0_requirements_direct", {}),
        )
    if name in {
        "tsr_task_r1_direct",
        "task_only_r1_direct",
        "task-only-r1-direct",
    }:
        return run_tsr_task_r1_direct(
            case,
            llm,
            config.get("methods", {}).get("tsr_task_r1_direct", {}),
        )
    if name in {
        "tsr_direct_no_requirements",
        "tsr_r0_direct",
        "direct-no-requirements",
    }:
        return run_tsr_direct_no_requirements(case, llm)
    if name in {
        "tsr_minimal_r0",
        "tsr_agent_step_minimal_r0",
    }:
        return run_tsr_minimal_r0(case, llm)
    if name in {
        "tsr_minimal_r1",
        "tsr_agent_step_minimal_r1",
    }:
        return run_tsr_minimal_r1(
            case,
            llm,
            config.get("methods", {}).get("tsr_minimal_r1", {}),
        )
    if name in {
        "ccv_trace_elephant_full_trace",
        "ccv_elephant_full_trace",
        "ccv_elephant_aligned_full_trace",
    }:
        method_config = config.get("methods", {}).get("ccv_trace_elephant_full_trace")
        if method_config is None:
            method_config = config.get("methods", {}).get("ccv_full_trace", {})
        return run_ccv_trace_elephant_full_trace(case, llm, method_config)
    if name in {
        "tsr_trace_elephant_causal",
        "tsr_elephant_causal",
        "tsr-loc-elephant-causal",
    }:
        method_config = config.get("methods", {}).get("tsr_trace_elephant_causal")
        if method_config is None:
            method_config = config.get("methods", {}).get("ccv_trace_elephant_full_trace", {})
        return run_tsr_trace_elephant_causal(case, llm, method_config)
    if name in {
        "tsr_trace_elephant_original",
        "tsr_elephant_original",
        "tsr-loc-elephant-original",
    }:
        method_config = config.get("methods", {}).get("tsr_trace_elephant_original")
        if method_config is None:
            method_config = config.get("methods", {}).get("ccv_ablation_no_gt", {})
        return run_tsr_trace_elephant_original(case, llm, method_config)
    if name in {
        "tsr_trace_elephant_gt_assisted",
        "tsr_elephant_gt_assisted",
        "tsr-loc-elephant-gt-assisted",
    }:
        method_config = config.get("methods", {}).get("tsr_trace_elephant_gt_assisted")
        if method_config is None:
            method_config = config.get("methods", {}).get("tsr_trace_elephant_original", {})
        return run_tsr_trace_elephant_gt_assisted(case, llm, method_config)
    if name in {
        "tsr_mp_bench_task_only",
        "tsr-mp-bench-task-only",
    }:
        method_config = config.get("methods", {}).get("tsr_mp_bench_task_only", {})
        pred = run_tsr_trace_elephant_original(case, llm, method_config)
        pred.method = "tsr_mp_bench_task_only"
        pred.trace["benchmark_adapter"] = "mp_bench"
        pred.trace["selection_rule"] = (
            "MP-Bench task-only TSR-Loc: generate natural-language task-success requirements from the "
            "task alone, then use the canonical earliest-unrecovered-violation full-trace localizer. "
            "The reference answer and final system answer are hidden from both calls."
        )
        return pred
    if name in {
        "tsr_mp_bench_gt_assisted",
        "tsr-mp-bench-gt-assisted",
    }:
        method_config = config.get("methods", {}).get("tsr_mp_bench_gt_assisted", {})
        pred = run_tsr_trace_elephant_gt_assisted(case, llm, method_config)
        pred.method = "tsr_mp_bench_gt_assisted"
        pred.trace["benchmark_adapter"] = "mp_bench"
        pred.trace["selection_rule"] = (
            "MP-Bench GT-assisted TSR-Loc: generate natural-language task-success requirements with the "
            "reference answer, then use the same canonical earliest-unrecovered-violation full-trace "
            "localizer. The final system answer remains hidden."
        )
        return pred
    if name in {
        "ccv_constraint_only_full_trace",
        "ccv_constraints_only_full_trace",
        "ccv_neutral_constraint_judge",
    }:
        method_config = config.get("methods", {}).get("ccv_constraint_only_full_trace")
        if method_config is None:
            method_config = config.get("methods", {}).get("ccv_full_trace", {})
        return run_ccv_constraint_only_full_trace(case, llm, method_config)
    if name in {
        "ccv_ablation_no_gt",
        "ccv_no_gt_full_trace",
        "ccv_task_only_constraint_full_trace",
    }:
        return run_ccv_information_ablation(
            case,
            llm,
            config.get("methods", {}).get("ccv_ablation_no_gt", {}),
            method_name="ccv_ablation_no_gt",
            ground_truth_to_generator=False,
            ground_truth_to_judge=False,
        )
    if name in {
        "tsr_no_requirements_same_judge",
        "ccv_no_requirements_same_judge",
    }:
        return run_tsr_requirement_content_ablation(
            case,
            llm,
            config.get("methods", {}).get("tsr_no_requirements_same_judge", {}),
            method_name="tsr_no_requirements_same_judge",
            requirement_condition="none",
        )
    if name in {
        "tsr_shuffled_requirements_same_judge",
        "ccv_shuffled_requirements_same_judge",
    }:
        return run_tsr_requirement_content_ablation(
            case,
            llm,
            config.get("methods", {}).get("tsr_shuffled_requirements_same_judge", {}),
            method_name="tsr_shuffled_requirements_same_judge",
            requirement_condition="shuffled",
        )
    if name in {
        "direct_requirements_full_trace",
        "direct_with_requirements",
        "direct_requirements",
    }:
        return run_direct_requirements_full_trace(
            case,
            llm,
            config.get("methods", {}).get("direct_requirements_full_trace", {}),
        )
    if name in {
        "ccv_ablation_gt_generator_only",
        "ccv_gt_generator_only_full_trace",
    }:
        return run_ccv_information_ablation(
            case,
            llm,
            config.get("methods", {}).get("ccv_ablation_gt_generator_only", {}),
            method_name="ccv_ablation_gt_generator_only",
            ground_truth_to_generator=True,
            ground_truth_to_judge=False,
        )
    if name in {
        "ccv_ablation_gt_judge_only",
        "ccv_gt_judge_only_full_trace",
    }:
        return run_ccv_information_ablation(
            case,
            llm,
            config.get("methods", {}).get("ccv_ablation_gt_judge_only", {}),
            method_name="ccv_ablation_gt_judge_only",
            ground_truth_to_generator=False,
            ground_truth_to_judge=True,
        )
    if name in {
        "ccv_checklist_equivalent_full_trace",
        "ccv_checklist_equivalent",
        "ccv_checklist_clone",
    }:
        return run_ccv_checklist_equivalent_full_trace(
            case,
            llm,
            config.get("methods", {}).get("ccv_checklist_equivalent_full_trace", {}),
        )
    if name in {
        "task_checklist_full_trace",
        "task_checklist",
        "ccv_task_only_full_trace",
        "ccv_task_full_trace",
        "task_ccv",
        "task-ccv",
    }:
        return run_task_checklist_full_trace(
            case,
            llm,
            config.get("methods", {}).get("task_checklist_full_trace", {}),
        )
    if name in {
        "reference_checklist_full_trace",
        "reference_checklist",
        "reference_ccv",
        "reference-ccv",
    }:
        return run_reference_checklist_full_trace(
            case,
            llm,
            config.get("methods", {}).get("reference_checklist_full_trace", {}),
        )
    if name in {"ccv_scalar10", "scalar_ccv10", "scv10"}:
        return run_ccv_scalar10(case, llm, config.get("methods", {}).get("ccv_scalar10", {}))
    if name in {"ccv_simple10", "simple_ccv10"}:
        return run_ccv_scalar10(case, llm, config.get("methods", {}).get("ccv_simple10", {}), method_name="ccv_simple10")
    if name in {"ccv_ordinal10", "ordinal_ccv10"}:
        return run_ccv_ordinal10(case, llm, config.get("methods", {}).get("ccv_ordinal10", {}))
    if name in {"ccv_beam10", "ccv-beam10", "ccvbeam10"}:
        return run_ccv_beam10(case, llm, config.get("methods", {}).get("ccv_beam10", {}))
    if name in {"ccv_adaptive_beam", "ccv_beam_adaptive", "adaptive_ccv_beam"}:
        return run_ccv_adaptive_beam(case, llm, config.get("methods", {}).get("ccv_adaptive_beam", {}))
    if name in {
        "ccv_global_router_beam",
        "ccv_router_beam",
        "ccv_adaptive_beam_global_router",
        "ccv_global_router_adaptive_beam",
        "ccv_gr_beam",
        "cgv_global_router_beam",
        "cgv_router_beam",
    }:
        method_config = config.get("methods", {}).get("ccv_global_router_beam")
        if method_config is None:
            method_config = config.get("methods", {}).get("ccv_adaptive_beam", {})
        return run_ccv_global_router_beam(case, llm, method_config)
    if name in {"cgv_full_step_judge", "cgv_full_judge", "cgv_step_full_judge"}:
        return run_cgv_full_step_judge(case, llm, config.get("methods", {}).get("cgv_full_step_judge", {}))
    if name in {
        "cgv_global_router_step_judge",
        "cgv_gr_step_judge",
        "cgv_caw_global_router",
        "cgv_caw_gr",
    }:
        return run_cgv_global_router_step_judge(
            case,
            llm,
            config.get("methods", {}).get("cgv_global_router_step_judge", {}),
        )
    if name in {"ccv_adaptive_beam_context", "ccv_beam_adaptive_context", "adaptive_ccv_beam_context"}:
        return run_ccv_adaptive_beam_context(
            case,
            llm,
            config.get("methods", {}).get("ccv_adaptive_beam_context", {}),
        )
    if name in {"ccv_beam_simple10", "ccv-beam-simple10", "ccvbeam_simple10"}:
        return run_ccv_beam_simple10(case, llm, config.get("methods", {}).get("ccv_beam_simple10", {}))
    if name in {"ccv_beam_ordinal10", "ccv-beam-ordinal10", "ccvbeam_ordinal10"}:
        return run_ccv_beam_ordinal10(case, llm, config.get("methods", {}).get("ccv_beam_ordinal10", {}))
    if name in {
        "ccv_beam_ordinal_reread10",
        "ccv-beam-ordinal-reread10",
        "ccvbeam_ordinal_reread10",
        "ccv_beam_ordinal_loc10",
    }:
        return run_ccv_beam_ordinal_reread10(case, llm, config.get("methods", {}).get("ccv_beam_ordinal_reread10", {}))
    raise ValueError(f"Unknown method: {name}")


def run_all_at_once(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(all_at_once_prompt(case))
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="all_at_once",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
        trace={"raw_response": raw, "parsed": parsed},
    )


def run_task_only_all_at_once(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(task_only_all_at_once_prompt(case))
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="task_only_all_at_once",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "ground_truth_visible": False,
            "explicit_final_answer_visible": False,
            "selection_rule": "One-call direct attribution with task-only information visibility.",
        },
    )


def run_trace_elephant_official_all_at_once(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run TraceElephant's official no-GT All-at-Once prompt on the compact trace."""
    no_gt_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    max_full_trace_chars = int(method_config.get("max_full_trace_chars", 0))
    working_steps, context_compaction = compact_steps_to_total_char_budget(
        case.steps,
        max_full_trace_chars,
    )
    prompt = trace_elephant_official_all_at_once_prompt(no_gt_case, working_steps)
    raw = generate_with_system_prompt(
        llm,
        prompt,
        "You are a helpful assistant skilled in analyzing conversations.",
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed)
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="trace_elephant_official_all_at_once",
        agent=agent,
        step=step,
        confidence=None,
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "ground_truth_visible": False,
            "explicit_final_answer_visible": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "context_safety_compaction": context_compaction,
            "source": (
                "TraceElephant official _build_all_at_once_prompt and system message, rendered over the shared "
                "compact trace representation under the paper's without-ground-truth condition."
            ),
            "selection_rule": (
                "One static full-trace call asks for the directly responsible agent and the first step where "
                "that agent made a mistake. The reference answer and final system answer are hidden."
            ),
        },
    )


def run_who_when_official_all_at_once(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(who_when_official_all_at_once_prompt(case))
    parsed = parse_a2p_official_response(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="who_when_official_all_at_once",
        agent=agent,
        step=step,
        confidence=None,
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "source": "Who&When official all-at-once prompt structure from Agents_Failure_Attribution.",
            "selection_rule": (
                "Full trace in one prompt with problem and ground-truth answer; parse Agent Name, Step Number, "
                "and Reason for Mistake text fields."
            ),
        },
    )


def run_who_when_pro_all_at_once(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(who_when_pro_all_at_once_prompt(case))
    parsed = parse_a2p_official_response(raw)
    error_mode_match = re.search(r"(?im)^\s*(?:[-*]\s*)?\**\s*Error\s+Mode\s*\**\s*:?\s*\**\s*:?\s*(.+?)\s*$", raw)
    reason_match = re.search(r"(?im)^\s*(?:[-*]\s*)?\**\s*Reason\s*\**\s*:?\s*\**\s*:?\s*(.+?)\s*$", raw)
    error_mode = clean_a2p_field(error_mode_match.group(1)) if error_mode_match else None
    reason = clean_a2p_field(reason_match.group(1)) if reason_match else parsed.get("reason") or parsed.get("Reason for Mistake")
    parsed["error_mode"] = error_mode
    parsed["reason"] = reason
    return Prediction(
        case_id=case.case_id,
        method="who_when_pro_all_at_once",
        agent=parsed_agent(parsed),
        step=parsed_step(parsed),
        confidence=None,
        reason=normalize_optional_str(reason),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "error_mode": error_mode,
            "ground_truth_visible": False,
            "explicit_final_answer_visible": False,
            "source": "Who&When Pro Appendix J.1 closed-book All-at-Once evaluation template.",
        },
    )


def run_who_when_official_step_by_step(case: Case, llm: LLM) -> Prediction:
    step_results: list[dict[str, Any]] = []
    for idx, step in enumerate(case.steps):
        history = case.steps[: idx + 1]
        raw = llm.generate(who_when_official_step_by_step_prompt(case, history=history, current_step=step))
        parsed = parse_who_when_official_step_response(raw)
        parsed["current_step"] = step.step
        parsed["current_agent"] = step.agent
        parsed["raw_response"] = raw
        step_results.append(parsed)
        if parsed.get("contains_error") is True:
            return Prediction(
                case_id=case.case_id,
                method="who_when_official_step_by_step",
                agent=step.agent,
                step=step.step,
                confidence=None,
                reason=normalize_optional_str(parsed.get("reason")),
                trace={
                    "step_results": strip_raw_large(step_results),
                    "source": "Who&When official step-by-step prompt structure from Agents_Failure_Attribution.",
                    "selection_rule": (
                        "Evaluate growing prefixes one step at a time and stop at the first response whose "
                        "first field is Yes."
                    ),
                },
            )
    return Prediction(
        case_id=case.case_id,
        method="who_when_official_step_by_step",
        agent=None,
        step=None,
        confidence=None,
        reason="No decisive errors found by official step-by-step scan.",
        trace={
            "step_results": strip_raw_large(step_results),
            "source": "Who&When official step-by-step prompt structure from Agents_Failure_Attribution.",
            "selection_rule": "No Yes response was observed before the trace ended.",
        },
    )


def run_who_when_official_binary_search(case: Case, llm: LLM) -> Prediction:
    if not case.steps:
        return Prediction(
            case_id=case.case_id,
            method="who_when_official_binary_search",
            agent=None,
            step=None,
            confidence=None,
            reason="No steps available.",
            trace={"search_results": []},
        )

    start = 0
    end = len(case.steps) - 1
    search_results: list[dict[str, Any]] = []
    while start < end:
        mid = start + (end - start) // 2
        segment = case.steps[start : end + 1]
        upper_half = case.steps[start : mid + 1]
        lower_half = case.steps[mid + 1 : end + 1]
        raw = llm.generate(
            who_when_official_binary_search_prompt(
                case,
                segment=segment,
                upper_half=upper_half,
                lower_half=lower_half,
            )
        )
        decision = parse_who_when_official_binary_response(raw)
        search_results.append(
            {
                "start": start,
                "end": end,
                "mid": mid,
                "raw_response": raw,
                "decision": decision,
            }
        )
        if decision == "lower":
            start = mid + 1
        else:
            end = mid

    agent = agent_at_step(case.steps, start)
    return Prediction(
        case_id=case.case_id,
        method="who_when_official_binary_search",
        agent=agent,
        step=start,
        confidence=None,
        reason="Official binary-search prompt recursively selected halves until one step remained.",
        trace={
            "search_results": strip_raw_large(search_results),
            "source": "Who&When official binary-search prompt structure from Agents_Failure_Attribution.",
            "selection_rule": "Ask upper/lower half over the active segment; ambiguous responses default to upper for reproducibility.",
        },
    )


def run_a2p_original(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(a2p_all_at_once_prompt(case))
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="a2p_original",
        agent=agent,
        step=step,
        confidence=a2p_result_score(parsed),
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "selection_rule": "Original A2P-style all-at-once counterfactual attribution over the full rendered trace.",
        },
    )


def run_a2p_official(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(a2p_official_prompt(case))
    parsed = parse_a2p_official_response(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="a2p_official",
        agent=agent,
        step=step,
        confidence=None,
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "selection_rule": (
                "Official A2P-like all-at-once reproduction: construct_a2p_prompt(a2p=True), "
                "0-based contextual step numbering, text output parser, no chunking or beam."
            ),
        },
    )


def run_a2p_repo_exact(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(a2p_repo_exact_prompt(case))
    parsed = parse_a2p_official_response(raw)
    return Prediction(
        case_id=case.case_id,
        method="a2p_repo_exact",
        agent=parsed_agent(parsed),
        step=parsed_step(parsed),
        confidence=None,
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "selection_rule": (
                "A2P repository construct_a2p_prompt(a2p=True) at commit "
                "7953d780c85054721a7b4bf246bcf60a16bb28af; one full-trace call, "
                "0-based contextual step numbering, no chunking or beam."
            ),
            "source_repository": "https://github.com/ResearAI/A2P",
            "source_commit": "7953d780c85054721a7b4bf246bcf60a16bb28af",
        },
    )


def run_a2p_scaffold_original(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(a2p_scaffold_all_at_once_prompt(case))
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    agent = parsed_agent(parsed)
    return Prediction(
        case_id=case.case_id,
        method="a2p_scaffold_original",
        agent=agent,
        step=step,
        confidence=a2p_result_score(parsed),
        reason=normalize_optional_str(parsed.get("reason") or parsed.get("causal_mechanism")),
        trace={
            "raw_response": raw,
            "parsed": parsed,
            "selection_rule": "Original A2P scaffold all-at-once attribution over the full rendered trace.",
        },
    )


def run_step_by_step(case: Case, llm: LLM) -> Prediction:
    step_results: list[dict[str, Any]] = []
    for idx, step in enumerate(case.steps):
        if is_human_agent(step.agent):
            continue
        history = case.steps[: idx + 1]
        raw = llm.generate(step_by_step_prompt(case, history=history, current_step=step))
        parsed = safe_json(raw)
        parsed["current_step"] = step.step
        parsed["current_agent"] = step.agent
        parsed["raw_response"] = raw
        step_results.append(parsed)
        if as_bool(parsed.get("contains_error") or parsed.get("error") or parsed.get("yes")):
            pred_step = parsed_step(parsed)
            return Prediction(
                case_id=case.case_id,
                method="step_by_step",
                agent=parsed_agent(parsed) or step.agent,
                step=pred_step if pred_step is not None else step.step,
                confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
                reason=normalize_optional_str(parsed.get("reason")),
                trace={"step_results": strip_raw_large(step_results)},
            )

    fallback = first_system_step(case.steps)
    return Prediction(
        case_id=case.case_id,
        method="step_by_step",
        agent=fallback.agent if fallback else None,
        step=fallback.step if fallback else None,
        confidence=None,
        reason="No decisive error detected; fell back to first non-human system step.",
        trace={"step_results": strip_raw_large(step_results)},
    )


def run_binary_search(case: Case, llm: LLM) -> Prediction:
    low = 0
    high = len(case.steps) - 1
    search_results: list[dict[str, Any]] = []

    while low < high:
        mid = (low + high) // 2
        segment = case.steps[low : high + 1]
        earlier_half = case.steps[low : mid + 1]
        later_half = case.steps[mid + 1 : high + 1]
        if not later_half:
            break
        raw = llm.generate(binary_search_prompt(case, segment=segment, earlier_half=earlier_half, later_half=later_half))
        parsed = safe_json(raw)
        half = str(parsed.get("half") or parsed.get("choice") or "").strip().lower()
        parsed.update(
            {
                "low_step": case.steps[low].step,
                "high_step": case.steps[high].step,
                "mid_step": case.steps[mid].step,
                "raw_response": raw,
            }
        )
        search_results.append(parsed)
        if "later" in half or "upper" in half or "second" in half:
            low = mid + 1
        else:
            high = mid

    selected = case.steps[low]
    return Prediction(
        case_id=case.case_id,
        method="binary_search",
        agent=None if is_human_agent(selected.agent) else selected.agent,
        step=selected.step,
        confidence=as_float(search_results[-1].get("confidence"), default=None) if search_results else None,  # type: ignore[arg-type]
        reason=normalize_optional_str(search_results[-1].get("reason")) if search_results else None,
        trace={"search_results": strip_raw_large(search_results)},
    )


def paper_hybrid_target_agent_candidates(
    steps: list[LogStep],
    target_agent: str,
    context_steps: int = 2,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for step in steps:
        if is_human_agent(step.agent) or not agents_match(step.agent, target_agent):
            continue
        before_context, after_context = steps_before_after(steps, [step], context_steps)
        candidates.append(
            {
                "step": step,
                "before_context": before_context,
                "after_context": after_context,
            }
        )
    return candidates


def paper_hybrid_candidate_step_trace(candidate_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": item["step"].step,
            "agent": item["step"].agent,
            "before_start": item["before_context"][0].step if item.get("before_context") else None,
            "before_end": item["before_context"][-1].step if item.get("before_context") else None,
            "after_start": item["after_context"][0].step if item.get("after_context") else None,
            "after_end": item["after_context"][-1].step if item.get("after_context") else None,
        }
        for item in candidate_steps
    ]


def run_paper_hybrid_joint_step_selection(
    case: Case,
    target_agent: str,
    candidate_steps: list[dict[str, Any]],
    selection_source: str,
    llm: LLM,
) -> dict[str, Any]:
    if not candidate_steps:
        return {
            "agent": target_agent,
            "step": None,
            "reason": "No target-agent candidate steps were available.",
            "valid_target_agent_step": False,
        }
    raw = llm.generate(
        paper_hybrid_target_agent_joint_prompt(
            case=case,
            target_agent=target_agent,
            candidate_steps=candidate_steps,
            selection_source=selection_source,
        )
    )
    parsed = safe_json(raw)
    parsed["raw_response"] = raw
    candidate_step_numbers = {int(item["step"].step) for item in candidate_steps}
    step = parse_int_maybe(parsed.get("step"))
    agent = normalize_optional_str(parsed.get("agent")) or target_agent
    parsed["valid_target_agent_step"] = (
        step is not None
        and step in candidate_step_numbers
        and agents_match(agent, target_agent)
    )
    if step is not None:
        parsed["step"] = step
    return parsed


def run_paper_hybrid(case: Case, llm: LLM) -> Prediction:
    all_pred = run_all_at_once(case, llm)
    target_agent = all_pred.agent
    if not target_agent:
        all_pred.method = "paper_hybrid"
        all_pred.reason = "All-at-once did not return an agent; using all-at-once fallback."
        return all_pred

    candidate_steps = paper_hybrid_target_agent_candidates(case.steps, target_agent, context_steps=2)
    joint_result = run_paper_hybrid_joint_step_selection(
        case=case,
        target_agent=target_agent,
        candidate_steps=candidate_steps,
        selection_source="all_at_once",
        llm=llm,
    )
    if joint_result.get("valid_target_agent_step"):
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid",
            agent=normalize_optional_str(joint_result.get("agent")) or target_agent,
            step=int(joint_result["step"]) if joint_result.get("step") is not None else None,
            confidence=None,
            reason=normalize_optional_str(joint_result.get("reason")),
            trace={
                "all_at_once": all_pred.trace,
                "target_agent": target_agent,
                "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
                "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
                "selection_rule": "target agent from all-at-once; final step from joint reread of target-agent actions",
            },
        )

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid",
        agent=target_agent,
        step=all_pred.step,
        confidence=None,
        reason="Joint target-agent reread produced no usable target-agent step; using all-at-once fallback.",
        trace={
            "all_at_once": all_pred.trace,
            "target_agent": target_agent,
            "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
            "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "selection_rule": "target agent from all-at-once; fallback to all-at-once step",
        },
    )


def run_chunk_vote10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_vote_stage(case, chunks, summaries, llm)
    best = select_chunk_vote_candidate(chunk_results)

    return Prediction(
        case_id=case.case_id,
        method="chunk_vote10",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": best.get("chunk_id"),
        },
    )


def run_chunk_vote_simple10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_bool_stage(case, chunks, summaries, llm)
    best = select_chunk_bool_candidate(chunk_results)

    return Prediction(
        case_id=case.case_id,
        method="chunk_vote_simple10",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=None,
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": best.get("chunk_id"),
            "selection_rule": "earliest positive chunk with a step; fallback to earliest chunk candidate",
        },
    )


def run_chunk_vote_ordinal_reread10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_ordinal_stage(case, chunks, summaries, llm)
    top_chunks = select_top_ordinal_chunks(chunk_results)

    reread_results: list[dict[str, Any]] = []
    for selected in top_chunks:
        chunk_idx = int(selected["chunk_id"]) - 1
        chunk = chunks[chunk_idx]
        raw = llm.generate(
            chunk_ordinal_reread_prompt(
                case=case,
                chunk_id=int(selected["chunk_id"]),
                chunk_count=len(chunks),
                chunk=chunk,
                first_pass=strip_raw_large([selected])[0],
                prev_summary=summaries[chunk_idx - 1] if chunk_idx > 0 else "None",
                next_summary=summaries[chunk_idx + 1] if chunk_idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        normalize_ordinal_result(parsed)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = selected["chunk_id"]
        parsed["chunk_start"] = selected["chunk_start"]
        parsed["chunk_end"] = selected["chunk_end"]
        parsed["first_pass_blame_score"] = selected.get("blame_score")
        parsed["raw_response"] = raw
        reread_results.append(parsed)

    best = select_ordinal_candidate(reread_results or top_chunks)
    agent = normalize_optional_str(best.get("agent"))
    if agent and agent.upper() == "NONE":
        agent = None

    return Prediction(
        case_id=case.case_id,
        method="chunk_vote_ordinal_reread10",
        agent=agent,
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=None,
        reason="Selected by ordinal chunk blame_score with reread of all top-scoring chunks.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "top_chunk_ids": [item.get("chunk_id") for item in top_chunks],
            "reread_results": strip_raw_large(reread_results),
            "selected_chunk_id": best.get("chunk_id"),
            "selection_rule": "first pass: max blame_score chunks; reread all ties; final: highest reread blame_score, earliest step, earliest chunk",
        },
    )


def run_who_when_beam_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "selected_top_k_chunks_in_temporal_order"
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        base = run_all_at_once(case, llm)
        base.method = "who_when_beam_joint"
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to basic Who&When all-at-once prompt.",
        }
        return base

    chunk_results = run_chunk_vote_stage(case, chunks, summaries, llm)
    selected_chunks = select_who_when_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append({"chunk_id": chunk_id, "chunk": chunks[chunk_id - 1]})

    joint_raw = llm.generate(who_when_beam_joint_prompt(case=case, selected_chunks=selected_payload))
    joint = safe_json(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if as_bool(joint.get("contains_decisive_error"), default=True) and selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": as_float(joint.get("confidence"), default=None),  # type: ignore[arg-type]
            "score": as_float(joint.get("confidence"), default=0.0),
            "reason": normalize_optional_str(joint.get("reason")),
        }
        selection_rule = (
            "Basic Who&When chunk scorer selected top-k chunks; selected chunks were jointly reread in temporal "
            "order and the joint attribution was accepted if it named a step inside selected chunks."
        )
    else:
        best = select_chunk_vote_candidate(selected_chunks or chunk_results)
        selection_rule = (
            "Joint basic Who&When reread did not return a usable selected step; using best selected chunk fallback."
        )

    return Prediction(
        case_id=case.case_id,
        method="who_when_beam_joint",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "selection_rule": selection_rule,
        },
    )


def run_who_when_official_beam_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "selected_top_k_chunks_in_temporal_order"
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        base = run_who_when_official_all_at_once(case, llm)
        base.method = "who_when_official_beam_joint"
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": (
                "Single adaptive chunk; falling back to official Who&When all-at-once prompt reproduction."
            ),
        }
        return base

    chunk_results = run_who_when_official_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_who_when_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append({"chunk_id": chunk_id, "chunk": chunks[chunk_id - 1]})

    joint_raw = llm.generate(who_when_official_beam_joint_prompt(case=case, selected_chunks=selected_payload))
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": as_float(joint.get("confidence") or joint.get("score"), default=None),  # type: ignore[arg-type]
            "score": as_float(joint.get("score"), default=0.0),
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "Official Who&When-style chunk scorer selected top-k chunks; selected chunks were jointly reread in "
            "temporal order with the official text-output attribution format."
        )
    else:
        best = select_chunk_vote_candidate(selected_chunks or chunk_results)
        selection_rule = (
            "Official Who&When-style joint reread did not return a usable selected step; using best selected "
            "chunk fallback."
        )

    return Prediction(
        case_id=case.case_id,
        method="who_when_official_beam_joint",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "source": (
                "Wrapper adaptation: official Who&When all-at-once text-output attribution prompt applied to "
                "adaptive chunks, then a selected-chunk joint reread."
            ),
            "selection_rule": selection_rule,
        },
    )


def run_who_when_official_global_router_beam_joint(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "global_router_selected_chunks_in_temporal_order"

    if len(chunks) <= 2:
        base = run_who_when_official_all_at_once(case, llm)
        base.method = "who_when_official_global_router_beam_joint"
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": (
                "Official Who&When all-at-once fallback for traces that produce at most two adaptive chunks."
            ),
        }
        return base

    conversation = render_who_when_official_steps(working_steps, with_step_numbers=True)
    router_raw = llm.generate(
        who_when_official_global_chunk_router_prompt(
            case=case,
            conversation=conversation,
            chunk_ranges=chunking.get("chunk_ranges", []),
            beam_k=beam_k,
        )
    )
    router_result = safe_json(router_raw)
    router_result["raw_response"] = router_raw
    selected_token_budget = int(method_config.get("selected_token_budget", 0))
    selected_chunks, router_candidates = select_echo_global_router_chunks(
        router_result,
        chunks,
        chunking,
        beam_k,
        selected_token_budget=selected_token_budget if selected_token_budget > 0 else None,
    )
    if selected_token_budget > 0:
        chunking["selected_token_budget"] = selected_token_budget
        chunking["selected_token_budget_actual"] = sum(
            int(item.get("estimated_tokens") or 0) for item in selected_chunks
        )
        chunking["selected_token_budget_policy"] = "router_order_closest_prefix"

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append({"chunk_id": chunk_id, "chunk": chunks[chunk_id - 1]})

    joint_raw = llm.generate(
        who_when_official_global_router_joint_prompt(case=case, selected_chunks=selected_payload)
    )
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": as_float(joint.get("confidence") or joint.get("score"), default=None),  # type: ignore[arg-type]
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "The Who&When global router read the whole trace and selected chunk IDs without confidence scores; "
            "the selected chunks were jointly reread in temporal order with a Who&When-compatible final "
            "attribution prompt that repeats the router's earliest-responsible-error criteria."
        )
    else:
        best = {
            "agent": None,
            "step": None,
            "confidence": None,
            "reason": "Global-router reread produced no usable Who&When attribution inside selected chunks.",
        }
        selection_rule = "Who&When global-router reread did not return a usable step inside selected chunks."

    return Prediction(
        case_id=case.case_id,
        method="who_when_official_global_router_beam_joint",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "router_result": {k: v for k, v in router_result.items() if k != "raw_response"},
            "router_candidates": router_candidates,
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "source": (
                "Wrapper adaptation: the official Who&When attribution objective is used for whole-trace chunk "
                "routing, followed by a method-compatible selected-chunk joint reread with the same causal "
                "selection criteria."
            ),
            "selection_rule": selection_rule,
        },
    )


def run_paper_hybrid10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_vote_stage(case, chunks, summaries, llm)
    target_agent = select_target_agent_from_chunks(chunk_results)
    best_chunk_candidate = select_chunk_vote_candidate(chunk_results)

    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid10",
            agent=normalize_optional_str(best_chunk_candidate.get("agent")),
            step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
            confidence=as_float(best_chunk_candidate.get("confidence"), default=None),  # type: ignore[arg-type]
            reason="No target agent could be selected from chunk votes; using chunk vote fallback.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunk_results": strip_raw_large(chunk_results),
                "target_agent": None,
                "step_results": [],
            },
        )

    candidate_steps = paper_hybrid_target_agent_candidates(working_steps, target_agent, context_steps=2)
    joint_result = run_paper_hybrid_joint_step_selection(
        case=case,
        target_agent=target_agent,
        candidate_steps=candidate_steps,
        selection_source="chunk_vote10_agent_selection",
        llm=llm,
    )
    if joint_result.get("valid_target_agent_step"):
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid10",
            agent=normalize_optional_str(joint_result.get("agent")) or target_agent,
            step=int(joint_result["step"]) if joint_result.get("step") is not None else None,
            confidence=None,
            reason=normalize_optional_str(joint_result.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunk_results": strip_raw_large(chunk_results),
                "target_agent": target_agent,
                "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
                "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
                "selection_rule": "target agent from chunk vote; final step from joint reread of target-agent actions",
            },
        )

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid10",
        agent=target_agent,
        step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
        confidence=None,
        reason="Joint target-agent reread produced no usable target-agent step; using chunk vote fallback.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "target_agent": target_agent,
            "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
            "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "selection_rule": "target agent from chunk vote; fallback to best chunk-vote candidate",
        },
    )


def run_paper_hybrid_adaptive(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []
    all_at_once_trace: dict[str, Any] | None = None
    best_chunk_candidate: dict[str, Any] = {}

    if len(chunks) == 1:
        all_pred = run_all_at_once(case, llm)
        target_agent = all_pred.agent
        all_at_once_trace = all_pred.trace
        best_chunk_candidate = {
            "agent": all_pred.agent,
            "step": all_pred.step,
            "confidence": all_pred.confidence,
            "reason": all_pred.reason,
        }
        selection_source = "all_at_once_short_trace"
    else:
        chunk_results = run_chunk_vote_stage(case, chunks, summaries, llm)
        target_agent = select_target_agent_from_chunks(chunk_results)
        best_chunk_candidate = select_chunk_vote_candidate(chunk_results)
        selection_source = "adaptive_chunk_vote"

    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_adaptive",
            agent=normalize_optional_str(best_chunk_candidate.get("agent")),
            step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
            confidence=as_float(best_chunk_candidate.get("confidence"), default=None),  # type: ignore[arg-type]
            reason="No target agent could be selected; using adaptive chunk fallback.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "all_at_once": all_at_once_trace,
                "target_agent": None,
                "selection_source": selection_source,
                "step_results": [],
            },
        )

    candidate_steps = paper_hybrid_target_agent_candidates(working_steps, target_agent, context_steps=2)
    joint_result = run_paper_hybrid_joint_step_selection(
        case=case,
        target_agent=target_agent,
        candidate_steps=candidate_steps,
        selection_source=selection_source,
        llm=llm,
    )
    if joint_result.get("valid_target_agent_step"):
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_adaptive",
            agent=normalize_optional_str(joint_result.get("agent")) or target_agent,
            step=int(joint_result["step"]) if joint_result.get("step") is not None else None,
            confidence=None,
            reason=normalize_optional_str(joint_result.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "all_at_once": all_at_once_trace,
                "target_agent": target_agent,
                "selection_source": selection_source,
                "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
                "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
                "selection_rule": "target agent selected first; final step from joint reread of target-agent actions",
            },
        )

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid_adaptive",
        agent=target_agent,
        step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
        confidence=None,
        reason="Joint target-agent reread produced no usable target-agent step; using adaptive chunk fallback.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "all_at_once": all_at_once_trace,
            "target_agent": target_agent,
            "selection_source": selection_source,
            "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
            "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "selection_rule": "target agent selected first; fallback to adaptive chunk candidate",
        },
    )


def run_paper_hybrid_adaptive_context(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    context_steps = int(method_config.get("context_steps", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    chunking["context_steps"] = context_steps
    chunking["context_policy"] = "read_only_before_after_context"
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []
    all_at_once_trace: dict[str, Any] | None = None
    best_chunk_candidate: dict[str, Any] = {}

    if len(chunks) == 1:
        all_pred = run_all_at_once(case, llm)
        target_agent = all_pred.agent
        all_at_once_trace = all_pred.trace
        best_chunk_candidate = {
            "agent": all_pred.agent,
            "step": all_pred.step,
            "confidence": all_pred.confidence,
            "reason": all_pred.reason,
        }
        selection_source = "all_at_once_short_trace"
    else:
        chunk_results = run_chunk_vote_context_stage(case, working_steps, chunks, summaries, llm, context_steps)
        target_agent = select_target_agent_from_chunks(chunk_results)
        best_chunk_candidate = select_chunk_vote_candidate(chunk_results)
        selection_source = "adaptive_context_chunk_vote"

    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_adaptive_context",
            agent=normalize_optional_str(best_chunk_candidate.get("agent")),
            step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
            confidence=as_float(best_chunk_candidate.get("confidence"), default=None),  # type: ignore[arg-type]
            reason="No target agent could be selected; using adaptive context chunk fallback.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "all_at_once": all_at_once_trace,
                "target_agent": None,
                "selection_source": selection_source,
                "step_results": [],
            },
        )

    candidate_steps = paper_hybrid_target_agent_candidates(working_steps, target_agent, context_steps=context_steps)
    joint_result = run_paper_hybrid_joint_step_selection(
        case=case,
        target_agent=target_agent,
        candidate_steps=candidate_steps,
        selection_source=selection_source,
        llm=llm,
    )
    if joint_result.get("valid_target_agent_step"):
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_adaptive_context",
            agent=normalize_optional_str(joint_result.get("agent")) or target_agent,
            step=int(joint_result["step"]) if joint_result.get("step") is not None else None,
            confidence=None,
            reason=normalize_optional_str(joint_result.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "all_at_once": all_at_once_trace,
                "target_agent": target_agent,
                "selection_source": selection_source,
                "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
                "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
                "selection_rule": "target agent selected first; final step from joint reread of target-agent actions with local context",
            },
        )

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid_adaptive_context",
        agent=target_agent,
        step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
        confidence=None,
        reason="Joint target-agent reread produced no usable target-agent step; using adaptive context chunk fallback.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "all_at_once": all_at_once_trace,
            "target_agent": target_agent,
            "selection_source": selection_source,
            "target_agent_steps": paper_hybrid_candidate_step_trace(candidate_steps),
            "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "selection_rule": "target agent selected first; fallback to adaptive context chunk candidate",
        },
    )


def run_paper_hybrid_global_router(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    context_steps = int(method_config.get("context_steps", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["context_steps"] = context_steps
    chunking["router_policy"] = "whole_trace_target_agent_hybrid_chunk_router"
    chunking["localization_policy"] = "joint_reread_target_agent_steps_inside_selected_chunks"

    all_pred = run_all_at_once(case, llm)
    target_agent = all_pred.agent
    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_global_router",
            agent=all_pred.agent,
            step=all_pred.step,
            confidence=None,
            reason="All-at-once did not return a target agent; using all-at-once fallback.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": all_pred.trace,
                "target_agent": None,
                "selection_rule": "fallback_to_all_at_once_no_target_agent",
            },
        )

    if len(chunks) <= 2:
        selected_chunks = [
            {
                "chunk_id": idx + 1,
                "chunk_start": chunk[0].step,
                "chunk_end": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
                "agent": None,
                "step": None,
                "confidence": 0.0,
                "score": 0.0,
                "reason": "Short-trace fallback: reread all produced chunks.",
                "router_preference_rank": idx + 1,
                "selection_source": "short_trace_fallback",
            }
            for idx, chunk in enumerate(chunks)
        ]
        router_result: dict[str, Any] = {
            "skipped": True,
            "reason": "Trace produced at most two adaptive chunks; routing is unnecessary.",
        }
        router_candidates: list[dict[str, Any]] = []
    else:
        conversation = render_steps(working_steps)
        router_raw = llm.generate(
            paper_hybrid_global_chunk_router_prompt(
                case=case,
                target_agent=target_agent,
                conversation=conversation,
                chunk_ranges=chunking.get("chunk_ranges", []),
                beam_k=beam_k,
            )
        )
        router_result = safe_json(router_raw)
        router_result["raw_response"] = router_raw
        selected_token_budget = int(method_config.get("selected_token_budget", 0))
        selected_chunks, router_candidates = select_echo_global_router_chunks(
            router_result,
            chunks,
            chunking,
            beam_k,
            selected_token_budget=selected_token_budget if selected_token_budget > 0 else None,
        )
        if selected_token_budget > 0:
            chunking["selected_token_budget"] = selected_token_budget
            chunking["selected_token_budget_actual"] = sum(
                int(item.get("estimated_tokens") or 0) for item in selected_chunks
            )
            chunking["selected_token_budget_policy"] = "router_order_closest_prefix"

    selected_chunks = sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9)))
    selected_step_numbers: set[int] = set()
    for selected in selected_chunks:
        chunk_id = parse_int_maybe(selected.get("chunk_id"))
        if chunk_id is None or chunk_id < 1 or chunk_id > len(chunks):
            continue
        selected_step_numbers.update(step.step for step in chunks[chunk_id - 1])

    all_target_candidates = paper_hybrid_target_agent_candidates(
        working_steps,
        target_agent,
        context_steps=context_steps,
    )
    selected_target_candidates = [
        item for item in all_target_candidates if int(item["step"].step) in selected_step_numbers
    ]

    if selected_target_candidates:
        joint_result = run_paper_hybrid_joint_step_selection(
            case=case,
            target_agent=target_agent,
            candidate_steps=selected_target_candidates,
            selection_source="global_router_selected_chunks",
            llm=llm,
        )
    else:
        joint_result = {
            "agent": target_agent,
            "step": None,
            "reason": "No target-agent candidate steps were present inside the selected chunks.",
            "valid_target_agent_step": False,
        }

    if joint_result.get("valid_target_agent_step"):
        best_agent = normalize_optional_str(joint_result.get("agent")) or target_agent
        best_step = int(joint_result["step"]) if joint_result.get("step") is not None else None
        reason = normalize_optional_str(joint_result.get("reason"))
        selection_rule = (
            "Paper-hybrid CAW-GR: all-at-once selects the target agent; a whole-trace router selects chunks; "
            "the final step is selected by joint reread of target-agent actions inside selected chunks."
        )
    else:
        best_agent = target_agent
        best_step = all_pred.step
        reason = "Global-router target-agent reread produced no usable selected step; using all-at-once fallback."
        selection_rule = "target agent from all-at-once; fallback to all-at-once step after router/joint reread failure"

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid_global_router",
        agent=best_agent,
        step=best_step,
        confidence=None,
        reason=reason,
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "all_at_once": all_pred.trace,
            "target_agent": target_agent,
            "all_target_agent_steps": paper_hybrid_candidate_step_trace(all_target_candidates),
            "target_agent_steps": paper_hybrid_candidate_step_trace(selected_target_candidates),
            "router_result": {k: v for k, v in router_result.items() if k != "raw_response"},
            "router_candidates": router_candidates,
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_step_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "selection_rule": selection_rule,
        },
    )


def run_a2p_adaptive(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_adaptive",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "A2P all-at-once fallback for short adaptive trace",
            },
        )

    chunk_results = run_a2p_chunk_stage(case, chunks, summaries, llm)
    selected = select_a2p_chunk_candidate(chunk_results, threshold=threshold)
    selected_chunk = chunks[int(selected["chunk_id"]) - 1] if selected.get("chunk_id") else []

    step_result: dict[str, Any] = {}
    step_raw = ""
    if selected_chunk:
        step_raw = llm.generate(a2p_step_prompt(case=case, chunk=selected_chunk, chunk_candidate=strip_raw_large([selected])[0]))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        if step_result.get("step") is not None and not step_inside_chunk(step_result.get("step"), selected_chunk):
            step_result["ignored_out_of_focal_step"] = step_result.get("step")
            step_result["step"] = None

    step = step_result.get("step") or selected.get("step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method="a2p_adaptive",
        agent=normalize_optional_str(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=a2p_result_score(step_result) or a2p_result_score(selected),
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "step_result": step_result,
            "selection_rule": "Adaptive chunks scored by A2P counterfactual causal_score; earliest viable chunk above threshold, otherwise highest score.",
        },
    )


def run_a2p_adaptive_beam(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_adaptive_beam",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "A2P all-at-once fallback for short adaptive trace",
            },
        )

    chunk_results = run_a2p_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        step_raw = llm.generate(a2p_step_prompt(case=case, chunk=selected_chunk, chunk_candidate=strip_raw_large([selected])[0]))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        if step_result.get("step") is not None and not step_inside_chunk(step_result.get("step"), selected_chunk):
            step_result["ignored_out_of_focal_step"] = step_result.get("step")
            step_result["step"] = None
        step_results.append(step_result)

        step = step_result.get("step") or selected.get("step")
        agent = step_result.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "causal_score": a2p_result_score(step_result) or a2p_result_score(selected),
                "would_fix_failure": as_bool(step_result.get("would_fix_failure") or selected.get("would_fix_failure")),
                "abduction": step_result.get("abduction") or selected.get("abduction"),
                "action": step_result.get("action") or selected.get("action"),
                "prediction": step_result.get("prediction") or selected.get("prediction"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(a2p_rerank_prompt(case=case, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_a2p_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "causal_score": None, "reason": "No A2P candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="a2p_adaptive_beam",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=a2p_result_score(best),
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "Adaptive A2P beam: top-k counterfactual chunks, A2P step localization, A2P rerank.",
        },
    )


def run_a2p_scaffold_beam(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_scaffold_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_scaffold_beam",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("causal_mechanism")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Original A2P scaffold all-at-once fallback for short adaptive trace.",
            },
        )

    chunk_results = run_a2p_scaffold_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    reread_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        reread_raw = llm.generate(
            a2p_scaffold_reread_prompt(
                case=case,
                chunk=selected_chunk,
                chunk_candidate=strip_raw_large([selected])[0],
            )
        )
        reread = safe_json(reread_raw)
        reread["raw_response"] = reread_raw
        reread["source_chunk_id"] = selected.get("chunk_id")
        reread["source_chunk_start"] = selected.get("chunk_start")
        reread["source_chunk_end"] = selected.get("chunk_end")
        reread["source_chunk_score"] = selected.get("score")
        if reread.get("step") is not None and not step_inside_chunk(reread.get("step"), selected_chunk):
            reread["ignored_out_of_focal_step"] = reread.get("step")
            reread["step"] = None
            reread["would_fix_failure"] = False
        reread_results.append(reread)

        step = reread.get("step") or selected.get("step")
        agent = reread.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "causal_score": a2p_result_score(reread) or a2p_result_score(selected),
                "would_fix_failure": as_bool(reread.get("would_fix_failure") or selected.get("would_fix_failure")),
                "abduction": reread.get("abduction") or selected.get("abduction"),
                "action": reread.get("action") or selected.get("action"),
                "prediction": reread.get("prediction") or selected.get("prediction"),
                "causal_mechanism": reread.get("causal_mechanism") or selected.get("causal_mechanism"),
                "reason": normalize_optional_str(
                    reread.get("reason")
                    or reread.get("causal_mechanism")
                    or selected.get("reason")
                    or selected.get("causal_mechanism")
                ),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(a2p_scaffold_rerank_prompt(case=case, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_a2p_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "causal_score": None, "reason": "No A2P scaffold candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="a2p_scaffold_beam",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=a2p_result_score(best),
        reason=normalize_optional_str(best.get("reason") or best.get("causal_mechanism")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "reread_results": strip_raw_large(reread_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": (
                "Original A2P scaffold wrapped with adaptive chunking and beam retrieval: "
                "A2P scaffold chunk scoring, top-k selected chunk rereading, A2P scaffold rerank."
            ),
        },
    )


def run_a2p_scaffold_agent_hybrid(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    agent_k = int(method_config.get("agent_k", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_scaffold_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_scaffold_agent_hybrid",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("causal_mechanism")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Original A2P scaffold all-at-once fallback for short adaptive trace.",
            },
        )

    scaffold_chunk_results = run_a2p_scaffold_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(scaffold_chunk_results, beam_k=beam_k, threshold=threshold)
    hybrid_chunk_results = run_chunk_vote_stage_for_selected_chunks(case, chunks, summaries, selected_chunks, llm)
    target_agents = select_top_target_agents_from_chunks(hybrid_chunk_results, agent_k=agent_k)

    if not target_agents:
        fallback = select_a2p_chunk_candidate(selected_chunks or scaffold_chunk_results, threshold=threshold)
        return Prediction(
            case_id=case.case_id,
            method="a2p_scaffold_agent_hybrid",
            agent=normalize_optional_str(fallback.get("agent")),
            step=int(fallback["step"]) if fallback.get("step") is not None else None,
            confidence=a2p_result_score(fallback),
            reason=normalize_optional_str(fallback.get("reason") or fallback.get("causal_mechanism"))
            or "No target agent could be selected from scaffold top chunks.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "scaffold_chunk_results": strip_raw_large(scaffold_chunk_results),
                "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
                "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
                "target_agents": [],
                "step_results": [],
                "selection_rule": "Fallback to A2P scaffold selected chunk because paper-hybrid agent selection returned no agent.",
            },
        )

    selected_by_id = {int(item["chunk_id"]): item for item in selected_chunks if item.get("chunk_id") is not None}
    step_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for chunk_meta in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(chunk_meta["chunk_id"])
        chunk = chunks[chunk_id - 1]
        for idx, step in enumerate(chunk):
            if is_human_agent(step.agent):
                continue
            target_agent = first_matching_agent(step.agent, target_agents)
            if not target_agent:
                continue
            history = chunk[: idx + 1]
            raw = llm.generate(paper_hybrid_step_prompt(case, target_agent=target_agent, history=history, current_step=step))
            parsed = safe_json(raw)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["target_agent"] = target_agent
            parsed["target_agent_rank"] = target_agents.index(target_agent) + 1
            parsed["chunk_id"] = chunk_id
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["source_a2p_scaffold_chunk_score"] = selected_by_id.get(chunk_id, {}).get("score")
            parsed["raw_response"] = raw
            step_results.append(parsed)

            if not as_bool(parsed.get("contains_error") or parsed.get("error") or parsed.get("yes")):
                continue
            pred_step = parsed_step(parsed)
            candidate_step = pred_step if pred_step is not None else step.step
            if not step_inside_chunk(candidate_step, chunk):
                parsed["ignored_out_of_focal_step"] = candidate_step
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "step": int(candidate_step),
                    "agent": parsed_agent(parsed) or step.agent,
                    "target_agent": target_agent,
                    "target_agent_rank": target_agents.index(target_agent) + 1,
                    "chunk_id": chunk_id,
                    "chunk_start": chunk[0].step,
                    "chunk_end": chunk[-1].step,
                    "score": as_float(parsed.get("confidence"), default=0.0),
                    "confidence": as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
                    "reason": normalize_optional_str(parsed.get("reason")),
                    "context": context_around(working_steps, int(candidate_step)),
                }
            )

    candidates = dedupe_candidates(candidates)
    if candidates:
        best = sorted(
            candidates,
            key=lambda x: (
                int(x.get("step", 10**9)),
                int(x.get("target_agent_rank", 10**9)),
                -float(x.get("confidence") or 0.0),
            ),
        )[0]
        selection_rule = "A2P scaffold top chunks were rescored with paper-hybrid agent voting; final step is the earliest positive step among top-agent candidates."
    else:
        best = select_agent_hybrid_fallback_candidate(hybrid_chunk_results, selected_chunks, target_agents)
        selection_rule = "No top-agent step was flagged; using best top-chunk paper-hybrid fallback candidate."

    return Prediction(
        case_id=case.case_id,
        method="a2p_scaffold_agent_hybrid",
        agent=normalize_optional_str(best.get("agent") or best.get("target_agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "scaffold_chunk_results": strip_raw_large(scaffold_chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
            "target_agents": target_agents,
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "selection_rule": selection_rule,
        },
    )


def run_a2p_official_beam(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_beam",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for short adaptive trace.",
            },
        )

    chunk_results = run_a2p_official_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    reread_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        reread_raw = llm.generate(
            a2p_official_reread_prompt(
                case=case,
                chunk=selected_chunk,
                chunk_candidate=strip_raw_large([selected])[0],
            )
        )
        reread = parse_a2p_official_response(reread_raw)
        reread["raw_response"] = reread_raw
        reread["source_chunk_id"] = selected.get("chunk_id")
        reread["source_chunk_start"] = selected.get("chunk_start")
        reread["source_chunk_end"] = selected.get("chunk_end")
        reread["source_chunk_score"] = selected.get("score")
        if reread.get("step") is not None and not step_inside_chunk(reread.get("step"), selected_chunk):
            reread["ignored_out_of_focal_step"] = reread.get("step")
            reread["step"] = None
            reread["contains_counterfactual_error"] = False
            reread["would_fix_failure"] = False
        reread_results.append(reread)

        explicitly_rejected = (
            ("contains_counterfactual_error" in reread and not as_bool(reread.get("contains_counterfactual_error")))
            or ("would_fix_failure" in reread and not as_bool(reread.get("would_fix_failure")))
        )
        if explicitly_rejected:
            continue

        step = reread.get("step") if reread.get("step") is not None else selected.get("step")
        agent = reread.get("agent") or reread.get("Agent Name") or selected.get("agent") or selected.get("Agent Name")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "causal_score": a2p_result_score(reread) or a2p_result_score(selected),
                "would_fix_failure": as_bool(reread.get("would_fix_failure") or selected.get("would_fix_failure")),
                "reason": normalize_optional_str(
                    reread.get("reason")
                    or reread.get("Reason for Mistake")
                    or selected.get("reason")
                    or selected.get("Reason for Mistake")
                ),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(a2p_official_rerank_prompt(case=case, candidates=judge_candidates))
        rerank = parse_a2p_official_response(rerank_raw)
        best = select_a2p_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "causal_score": None, "reason": "No official A2P candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_beam",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=a2p_result_score(best),
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "reread_results": strip_raw_large(reread_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": (
                "Official A2P scaffold with context-allocation wrapper: adaptive chunks, top-k beam, "
                "official A2P reread, official A2P rerank."
            ),
        },
    )


def run_a2p_official_beam_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "selected_top_k_chunks_in_temporal_order"
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_beam_joint",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for short adaptive trace.",
            },
        )

    chunk_results = run_a2p_official_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append(
            {
                "chunk_id": chunk_id,
                "chunk": chunks[chunk_id - 1],
            }
        )

    joint_raw = llm.generate(a2p_official_beam_joint_prompt(case=case, selected_chunks=selected_payload))
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": a2p_result_score(joint),
            "score": a2p_result_score(joint),
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "Official A2P chunk scorer selected top-k chunks; selected chunks were jointly reread in temporal "
            "order and the joint A2P attribution was accepted if it named a step inside the selected chunks."
        )
    else:
        best = select_a2p_chunk_candidate(selected_chunks or chunk_results, threshold=threshold)
        selection_rule = (
            "Joint official-A2P reread did not return a usable step inside selected chunks; using best "
            "official-A2P top-chunk fallback candidate."
        )

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_beam_joint",
        agent=normalize_optional_str(best.get("agent") or best.get("Agent Name")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score") or best.get("causal_score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "selection_rule": selection_rule,
        },
    )


def run_a2p_official_global_router_beam_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "global_router_selected_chunks_in_temporal_order"

    if len(chunks) <= 2:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_global_router_beam_joint",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for traces that produce at most two adaptive chunks.",
            },
        )

    conversation = render_steps_a2p_official(working_steps, global_step_numbers=True)
    router_raw = llm.generate(
        a2p_official_global_chunk_router_prompt(
            case=case,
            conversation=conversation,
            chunk_ranges=chunking.get("chunk_ranges", []),
            beam_k=beam_k,
        )
    )
    router_result = safe_json(router_raw)
    router_result["raw_response"] = router_raw
    selected_token_budget = int(method_config.get("selected_token_budget", 0))
    selected_chunks, router_candidates = select_echo_global_router_chunks(
        router_result,
        chunks,
        chunking,
        beam_k,
        selected_token_budget=selected_token_budget if selected_token_budget > 0 else None,
    )
    if selected_token_budget > 0:
        chunking["selected_token_budget"] = selected_token_budget
        chunking["selected_token_budget_actual"] = sum(
            int(item.get("estimated_tokens") or 0) for item in selected_chunks
        )
        chunking["selected_token_budget_policy"] = "router_order_closest_prefix"

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append(
            {
                "chunk_id": chunk_id,
                "chunk": chunks[chunk_id - 1],
            }
        )

    joint_raw = llm.generate(a2p_official_beam_joint_prompt(case=case, selected_chunks=selected_payload))
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": a2p_result_score(joint),
            "score": a2p_result_score(joint),
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "Official A2P global router read the whole trace and selected chunk IDs without confidence scores; "
            "selected chunks were jointly reread with the unchanged official A2P beam-joint reread prompt and "
            "accepted if the final step was inside them."
        )
    else:
        best = {"agent": None, "step": None, "confidence": None, "reason": "Global-router reread produced no usable A2P attribution."}
        selection_rule = (
            "Official A2P global-router reread did not return a usable step inside selected chunks."
        )

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_global_router_beam_joint",
        agent=normalize_optional_str(best.get("agent") or best.get("Agent Name")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score") or best.get("causal_score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "router_result": {k: v for k, v in router_result.items() if k != "raw_response"},
            "router_candidates": router_candidates,
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "selection_rule": selection_rule,
            "threshold": threshold,
        },
    )


def run_a2p_official_local_confidence_beam_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "local_confidence_selected_chunks_in_temporal_order"

    if len(chunks) <= 2:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_local_confidence_beam_joint",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for traces that produce at most two adaptive chunks.",
            },
        )

    chunk_results = run_a2p_official_local_confidence_chunk_stage(case, chunks, llm)
    selected_chunks = select_a2p_local_confidence_chunks(chunk_results, chunks, beam_k=beam_k)

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        selected_payload.append(
            {
                "chunk_id": chunk_id,
                "chunk": chunks[chunk_id - 1],
            }
        )

    joint_raw = llm.generate(a2p_official_global_router_reread_prompt(case=case, selected_chunks=selected_payload))
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_step = parsed_step(joint)
    selected_step_in_payload = False
    if selected_step is not None:
        selected_step_in_payload = any(step_inside_chunk(selected_step, item["chunk"]) for item in selected_payload)

    if selected_step is not None and selected_step_in_payload:
        best = {
            "agent": parsed_agent(joint),
            "step": int(selected_step),
            "confidence": a2p_result_score(joint),
            "score": a2p_result_score(joint),
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "Each chunk was independently scored with a local confidence prompt; top-k chunks by confidence "
            "were jointly reread in temporal order and accepted if the final step was inside them."
        )
    else:
        best = select_a2p_local_confidence_fallback_candidate(selected_chunks or chunk_results)
        selection_rule = (
            "Local-confidence joint reread did not return a usable step inside selected chunks; using the "
            "highest-confidence chunk fallback candidate."
        )

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_local_confidence_beam_joint",
        agent=normalize_optional_str(best.get("agent") or best.get("Agent Name")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score") or best.get("causal_score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "selection_rule": selection_rule,
        },
    )


def run_a2p_official_agent_hybrid(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    agent_k = int(method_config.get("agent_k", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_agent_hybrid",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for short adaptive trace.",
            },
        )

    chunk_results = run_a2p_official_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    target_agents = select_top_agents_from_a2p_chunks(selected_chunks or chunk_results, agent_k=agent_k)

    if not target_agents:
        fallback = select_a2p_chunk_candidate(selected_chunks or chunk_results, threshold=threshold)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_agent_hybrid",
            agent=normalize_optional_str(fallback.get("agent") or fallback.get("Agent Name")),
            step=int(fallback["step"]) if fallback.get("step") is not None else None,
            confidence=a2p_result_score(fallback),
            reason=normalize_optional_str(fallback.get("reason") or fallback.get("Reason for Mistake"))
            or "No target agent could be selected from official A2P top chunks.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
                "target_agents": [],
                "step_results": [],
                "selection_rule": "Fallback to official A2P selected chunk because agent selection returned no agent.",
            },
        )

    step_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for chunk_meta in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(chunk_meta["chunk_id"])
        chunk = chunks[chunk_id - 1]
        for idx, step in enumerate(chunk):
            if is_human_agent(step.agent):
                continue
            target_agent = first_matching_agent(step.agent, target_agents)
            if not target_agent:
                continue
            history = chunk[: idx + 1]
            raw = llm.generate(
                a2p_official_agent_step_prompt(
                    case=case,
                    target_agent=target_agent,
                    history=history,
                    current_step=step,
                )
            )
            parsed = parse_a2p_official_response(raw)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["target_agent"] = target_agent
            parsed["target_agent_rank"] = target_agents.index(target_agent) + 1
            parsed["chunk_id"] = chunk_id
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["source_a2p_official_chunk_score"] = chunk_meta.get("score")
            parsed["raw_response"] = raw
            step_results.append(parsed)

            if not a2p_result_is_positive(parsed):
                continue
            pred_step = parsed_step(parsed)
            candidate_step = pred_step if pred_step is not None else step.step
            if not step_inside_chunk(candidate_step, chunk):
                parsed["ignored_out_of_focal_step"] = candidate_step
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "step": int(candidate_step),
                    "agent": parsed_agent(parsed) or step.agent,
                    "target_agent": target_agent,
                    "target_agent_rank": target_agents.index(target_agent) + 1,
                    "chunk_id": chunk_id,
                    "chunk_start": chunk[0].step,
                    "chunk_end": chunk[-1].step,
                    "score": a2p_result_score(parsed) or 0.0,
                    "confidence": a2p_result_score(parsed),
                    "reason": normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
                    "context": context_around(working_steps, int(candidate_step)),
                }
            )

    candidates = dedupe_candidates(candidates)
    if candidates:
        best = sorted(
            candidates,
            key=lambda x: (
                int(x.get("step", 10**9)),
                int(x.get("target_agent_rank", 10**9)),
                -float(x.get("confidence") or 0.0),
            ),
        )[0]
        selection_rule = (
            "Official A2P top chunks selected target agents; final step is the earliest positive official-A2P "
            "step scan among top-agent candidates."
        )
    else:
        best = select_a2p_chunk_candidate(selected_chunks or chunk_results, threshold=threshold)
        selection_rule = "No top-agent official-A2P step was flagged; using best official-A2P top-chunk fallback candidate."

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_agent_hybrid",
        agent=normalize_optional_str(best.get("agent") or best.get("target_agent") or best.get("Agent Name")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score") or best.get("causal_score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "target_agents": target_agents,
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "selection_rule": selection_rule,
        },
    )


def run_a2p_official_agent_hybrid_joint(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    agent_k = int(method_config.get("agent_k", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["joint_reread_policy"] = "selected_top_k_chunks_in_temporal_order"
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_official_prompt(case))
        parsed = parse_a2p_official_response(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_agent_hybrid_joint",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason") or parsed.get("Reason for Mistake")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "Official A2P all-at-once fallback for short adaptive trace.",
            },
        )

    chunk_results = run_a2p_official_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    target_agents = select_top_agents_from_a2p_chunks(selected_chunks or chunk_results, agent_k=agent_k)

    if not target_agents:
        fallback = select_a2p_chunk_candidate(selected_chunks or chunk_results, threshold=threshold)
        return Prediction(
            case_id=case.case_id,
            method="a2p_official_agent_hybrid_joint",
            agent=normalize_optional_str(fallback.get("agent") or fallback.get("Agent Name")),
            step=int(fallback["step"]) if fallback.get("step") is not None else None,
            confidence=a2p_result_score(fallback),
            reason=normalize_optional_str(fallback.get("reason") or fallback.get("Reason for Mistake"))
            or "No target agent could be selected from official A2P top chunks.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "chunk_results": strip_raw_large(chunk_results),
                "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
                "target_agents": [],
                "joint_result": None,
                "selection_rule": "Fallback to official A2P selected chunk because agent selection returned no agent.",
            },
        )

    selected_payload = []
    for item in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(item["chunk_id"])
        chunk = chunks[chunk_id - 1]
        selected_payload.append(
            {
                "chunk_id": chunk_id,
                "score": item.get("score"),
                "chunk": chunk,
            }
        )

    joint_raw = llm.generate(
        a2p_official_agent_joint_prompt(
            case=case,
            target_agents=target_agents,
            selected_chunks=selected_payload,
        )
    )
    joint = parse_a2p_official_response(joint_raw)
    joint["raw_response"] = joint_raw
    joint["target_agents"] = target_agents
    joint["selected_chunk_ids"] = [item.get("chunk_id") for item in selected_chunks]

    selected_chunk_lists = [item["chunk"] for item in selected_payload]
    selected_step = parsed_step(joint)
    selected_agent = parsed_agent(joint)
    selected_step_agent = None
    selected_step_in_payload = False
    if selected_step is not None:
        for chunk in selected_chunk_lists:
            if step_inside_chunk(selected_step, chunk):
                selected_step_in_payload = True
                for step in chunk:
                    if int(step.step) == int(selected_step):
                        selected_step_agent = step.agent
                        break
                break

    candidate_agent = selected_agent or selected_step_agent
    target_agent = first_matching_agent(candidate_agent or "", target_agents) if candidate_agent else None
    joint_is_usable = (
        a2p_result_is_positive(joint)
        and selected_step is not None
        and selected_step_in_payload
        and target_agent is not None
    )

    if joint_is_usable:
        best = {
            "agent": selected_agent or selected_step_agent or target_agent,
            "target_agent": target_agent,
            "step": int(selected_step),
            "confidence": a2p_result_score(joint),
            "score": a2p_result_score(joint),
            "reason": normalize_optional_str(joint.get("reason") or joint.get("Reason for Mistake")),
        }
        selection_rule = (
            "Official A2P top chunks selected target agents; selected top-k chunks were jointly reread in "
            "temporal order, and the joint A2P attribution was accepted if it named a target-agent step "
            "inside the selected chunks."
        )
    else:
        target_chunk_candidates = []
        for result in selected_chunks or chunk_results:
            agent = normalize_optional_str(result.get("agent") or result.get("Agent Name"))
            if not agent or result.get("step") is None:
                continue
            if first_matching_agent(agent, target_agents):
                target_chunk_candidates.append(result)
        best = select_a2p_chunk_candidate(target_chunk_candidates or selected_chunks or chunk_results, threshold=threshold)
        selection_rule = (
            "Joint official-A2P reread did not return a usable target-agent step inside selected chunks; "
            "using best target-agent top-chunk fallback candidate."
        )

    return Prediction(
        case_id=case.case_id,
        method="a2p_official_agent_hybrid_joint",
        agent=normalize_optional_str(best.get("agent") or best.get("target_agent") or best.get("Agent Name")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score") or best.get("causal_score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "target_agents": target_agents,
            "joint_result": {k: v for k, v in joint.items() if k != "raw_response"},
            "selection_rule": selection_rule,
        },
    )


def run_a2p_beam_agent_hybrid(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    agent_k = int(method_config.get("agent_k", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_beam_agent_hybrid",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "A2P all-at-once fallback for short adaptive trace.",
            },
        )

    a2p_chunk_results = run_a2p_chunk_stage(case, chunks, summaries, llm)
    selected_chunks = select_a2p_beam_chunks(a2p_chunk_results, beam_k=beam_k, threshold=threshold)
    hybrid_chunk_results = run_chunk_vote_stage_for_selected_chunks(case, chunks, summaries, selected_chunks, llm)
    target_agents = select_top_target_agents_from_chunks(hybrid_chunk_results, agent_k=agent_k)

    if not target_agents:
        fallback = select_a2p_chunk_candidate(selected_chunks or a2p_chunk_results, threshold=threshold)
        return Prediction(
            case_id=case.case_id,
            method="a2p_beam_agent_hybrid",
            agent=normalize_optional_str(fallback.get("agent")),
            step=int(fallback["step"]) if fallback.get("step") is not None else None,
            confidence=a2p_result_score(fallback),
            reason=normalize_optional_str(fallback.get("reason")) or "No target agent could be selected from top A2P chunks.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "a2p_chunk_results": strip_raw_large(a2p_chunk_results),
                "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
                "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
                "target_agents": [],
                "step_results": [],
                "selection_rule": "Fallback to A2P selected chunk because paper-hybrid agent selection returned no agent.",
            },
        )

    selected_by_id = {int(item["chunk_id"]): item for item in selected_chunks if item.get("chunk_id") is not None}
    step_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for chunk_meta in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(chunk_meta["chunk_id"])
        chunk = chunks[chunk_id - 1]
        for idx, step in enumerate(chunk):
            if is_human_agent(step.agent):
                continue
            target_agent = first_matching_agent(step.agent, target_agents)
            if not target_agent:
                continue
            history = chunk[: idx + 1]
            raw = llm.generate(paper_hybrid_step_prompt(case, target_agent=target_agent, history=history, current_step=step))
            parsed = safe_json(raw)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["target_agent"] = target_agent
            parsed["target_agent_rank"] = target_agents.index(target_agent) + 1
            parsed["chunk_id"] = chunk_id
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["source_a2p_chunk_score"] = selected_by_id.get(chunk_id, {}).get("score")
            parsed["raw_response"] = raw
            step_results.append(parsed)

            if not as_bool(parsed.get("contains_error") or parsed.get("error") or parsed.get("yes")):
                continue
            pred_step = parsed_step(parsed)
            candidate_step = pred_step if pred_step is not None else step.step
            if not step_inside_chunk(candidate_step, chunk):
                parsed["ignored_out_of_focal_step"] = candidate_step
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "step": int(candidate_step),
                    "agent": parsed_agent(parsed) or step.agent,
                    "target_agent": target_agent,
                    "target_agent_rank": target_agents.index(target_agent) + 1,
                    "chunk_id": chunk_id,
                    "chunk_start": chunk[0].step,
                    "chunk_end": chunk[-1].step,
                    "score": as_float(parsed.get("confidence"), default=0.0),
                    "confidence": as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
                    "reason": normalize_optional_str(parsed.get("reason")),
                    "context": context_around(working_steps, int(candidate_step)),
                }
            )

    candidates = dedupe_candidates(candidates)
    if candidates:
        best = sorted(
            candidates,
            key=lambda x: (
                int(x.get("step", 10**9)),
                int(x.get("target_agent_rank", 10**9)),
                -float(x.get("confidence") or 0.0),
            ),
        )[0]
        selection_rule = "Top A2P chunks were rescored with paper-hybrid agent voting; final step is the earliest positive step among top-agent candidates."
    else:
        best = select_agent_hybrid_fallback_candidate(hybrid_chunk_results, selected_chunks, target_agents)
        selection_rule = "No top-agent step was flagged; using best top-chunk paper-hybrid fallback candidate."

    return Prediction(
        case_id=case.case_id,
        method="a2p_beam_agent_hybrid",
        agent=normalize_optional_str(best.get("agent") or best.get("target_agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "a2p_chunk_results": strip_raw_large(a2p_chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
            "target_agents": target_agents,
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "selection_rule": selection_rule,
        },
    )


def run_a2p_beam_agent_hybrid_context(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.60))
    agent_k = int(method_config.get("agent_k", 2))
    context_steps = int(method_config.get("context_steps", 2))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["context_steps"] = context_steps
    chunking["context_policy"] = "read_only_before_after_context"
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        raw = llm.generate(a2p_all_at_once_prompt(case))
        parsed = safe_json(raw)
        step = parsed_step(parsed)
        return Prediction(
            case_id=case.case_id,
            method="a2p_beam_agent_hybrid_context",
            agent=parsed_agent(parsed),
            step=step,
            confidence=a2p_result_score(parsed),
            reason=normalize_optional_str(parsed.get("reason")),
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "all_at_once": {**parsed, "raw_response": raw},
                "selection_rule": "A2P all-at-once fallback for short adaptive trace.",
            },
        )

    a2p_chunk_results = run_a2p_chunk_context_stage(case, working_steps, chunks, summaries, llm, context_steps)
    selected_chunks = select_a2p_beam_chunks(a2p_chunk_results, beam_k=beam_k, threshold=threshold)
    hybrid_chunk_results = run_chunk_vote_context_stage_for_selected_chunks(
        case,
        working_steps,
        chunks,
        summaries,
        selected_chunks,
        llm,
        context_steps,
    )
    target_agents = select_top_target_agents_from_chunks(hybrid_chunk_results, agent_k=agent_k)

    if not target_agents:
        fallback = select_a2p_chunk_candidate(selected_chunks or a2p_chunk_results, threshold=threshold)
        return Prediction(
            case_id=case.case_id,
            method="a2p_beam_agent_hybrid_context",
            agent=normalize_optional_str(fallback.get("agent")),
            step=int(fallback["step"]) if fallback.get("step") is not None else None,
            confidence=a2p_result_score(fallback),
            reason=normalize_optional_str(fallback.get("reason")) or "No target agent could be selected from context top A2P chunks.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunking": chunking,
                "a2p_chunk_results": strip_raw_large(a2p_chunk_results),
                "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
                "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
                "target_agents": [],
                "step_results": [],
                "selection_rule": "Fallback to context A2P selected chunk because paper-hybrid agent selection returned no agent.",
            },
        )

    selected_by_id = {int(item["chunk_id"]): item for item in selected_chunks if item.get("chunk_id") is not None}
    step_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for chunk_meta in sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9))):
        chunk_id = int(chunk_meta["chunk_id"])
        chunk = chunks[chunk_id - 1]
        for step in chunk:
            if is_human_agent(step.agent):
                continue
            target_agent = first_matching_agent(step.agent, target_agents)
            if not target_agent:
                continue
            before_context, after_context = steps_before_after(working_steps, [step], context_steps)
            raw = llm.generate(
                paper_hybrid_step_context_prompt(
                    case,
                    target_agent=target_agent,
                    before_context=before_context,
                    current_step=step,
                    after_context=after_context,
                )
            )
            parsed = safe_json(raw)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["target_agent"] = target_agent
            parsed["target_agent_rank"] = target_agents.index(target_agent) + 1
            parsed["chunk_id"] = chunk_id
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["context_before_start"] = before_context[0].step if before_context else None
            parsed["context_before_end"] = before_context[-1].step if before_context else None
            parsed["context_after_start"] = after_context[0].step if after_context else None
            parsed["context_after_end"] = after_context[-1].step if after_context else None
            parsed["source_a2p_chunk_score"] = selected_by_id.get(chunk_id, {}).get("score")
            parsed["raw_response"] = raw

            if not parsed_step_matches(parsed, step.step):
                parsed["ignored_out_of_candidate_step"] = parsed_step(parsed)
                parsed["contains_error"] = False
            step_results.append(parsed)

            if not as_bool(parsed.get("contains_error") or parsed.get("error") or parsed.get("yes")):
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "step": step.step,
                    "agent": parsed_agent(parsed) or step.agent,
                    "target_agent": target_agent,
                    "target_agent_rank": target_agents.index(target_agent) + 1,
                    "chunk_id": chunk_id,
                    "chunk_start": chunk[0].step,
                    "chunk_end": chunk[-1].step,
                    "score": as_float(parsed.get("confidence"), default=0.0),
                    "confidence": as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
                    "reason": normalize_optional_str(parsed.get("reason")),
                    "context": context_around(working_steps, step.step),
                }
            )

    candidates = dedupe_candidates(candidates)
    if candidates:
        best = sorted(
            candidates,
            key=lambda x: (
                int(x.get("step", 10**9)),
                int(x.get("target_agent_rank", 10**9)),
                -float(x.get("confidence") or 0.0),
            ),
        )[0]
        selection_rule = "Context top A2P chunks were rescored with paper-hybrid agent voting; final step is the earliest positive candidate-step judgment."
    else:
        best = select_agent_hybrid_fallback_candidate(hybrid_chunk_results, selected_chunks, target_agents)
        selection_rule = "No context top-agent step was flagged; using best context top-chunk paper-hybrid fallback candidate."

    return Prediction(
        case_id=case.case_id,
        method="a2p_beam_agent_hybrid_context",
        agent=normalize_optional_str(best.get("agent") or best.get("target_agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "a2p_chunk_results": strip_raw_large(a2p_chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "hybrid_chunk_results": strip_raw_large(hybrid_chunk_results),
            "target_agents": target_agents,
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "selection_rule": selection_rule,
        },
    )


def run_paper_hybrid_simple10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_bool_stage(case, chunks, summaries, llm)
    target_agent = select_target_agent_from_bool_chunks(chunk_results)
    best_chunk_candidate = select_chunk_bool_candidate(chunk_results)

    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_simple10",
            agent=normalize_optional_str(best_chunk_candidate.get("agent")),
            step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
            confidence=None,
            reason="No target agent could be selected from unweighted chunk votes; using chunk fallback.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunk_results": strip_raw_large(chunk_results),
                "target_agent": None,
                "step_results": [],
                "selection_rule": "unweighted majority over positive chunk votes",
            },
        )

    step_results: list[dict[str, Any]] = []
    for chunk in chunks:
        for idx, step in enumerate(chunk):
            if is_human_agent(step.agent) or not agents_match(step.agent, target_agent):
                continue
            history = chunk[: idx + 1]
            raw = llm.generate(paper_hybrid_step_bool_prompt(case, target_agent=target_agent, history=history, current_step=step))
            parsed = safe_json(raw)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["raw_response"] = raw
            step_results.append(parsed)
            if as_bool(parsed.get("contains_error") or parsed.get("error") or parsed.get("yes")):
                pred_step = parsed_step(parsed)
                return Prediction(
                    case_id=case.case_id,
                    method="paper_hybrid_simple10",
                    agent=parsed_agent(parsed) or step.agent,
                    step=pred_step if pred_step is not None else step.step,
                    confidence=None,
                    reason=normalize_optional_str(parsed.get("reason")),
                    trace={
                        "working_step_count": len(working_steps),
                        "original_step_count": len(case.steps),
                        "chunk_results": strip_raw_large(chunk_results),
                        "target_agent": target_agent,
                        "step_results": strip_raw_large(step_results),
                        "selection_rule": "unweighted majority over positive chunk votes",
                    },
                )

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid_simple10",
        agent=target_agent,
        step=int(best_chunk_candidate["step"]) if best_chunk_candidate.get("step") is not None else None,
        confidence=None,
        reason="No target-agent step was flagged; using earliest chunk fallback.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "target_agent": target_agent,
            "step_results": strip_raw_large(step_results),
            "selection_rule": "unweighted majority over positive chunk votes",
        },
    )


def run_paper_hybrid_ordinal10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_chunk_ordinal_stage(case, chunks, summaries, llm)
    best_chunk_candidate = select_ordinal_candidate(chunk_results)
    target_agent = normalize_ordinal_agent(best_chunk_candidate.get("agent"))

    if not target_agent:
        return Prediction(
            case_id=case.case_id,
            method="paper_hybrid_ordinal10",
            agent=None,
            step=None,
            confidence=None,
            reason="No target agent could be selected from ordinal chunk scores.",
            trace={
                "working_step_count": len(working_steps),
                "original_step_count": len(case.steps),
                "chunk_results": strip_raw_large(chunk_results),
                "target_agent": None,
                "step_results": [],
                "selection_rule": "target agent from single highest ordinal chunk, no agent score accumulation",
            },
        )

    step_results: list[dict[str, Any]] = []
    for chunk in chunks:
        for idx, step in enumerate(chunk):
            if is_human_agent(step.agent) or not agents_match(step.agent, target_agent):
                continue
            history = chunk[: idx + 1]
            raw = llm.generate(
                paper_hybrid_step_ordinal_prompt(
                    case=case,
                    target_agent=target_agent,
                    history=history,
                    current_step=step,
                )
            )
            parsed = safe_json(raw)
            normalize_ordinal_result(parsed)
            parsed["current_step"] = step.step
            parsed["current_agent"] = step.agent
            parsed["chunk_start"] = chunk[0].step
            parsed["chunk_end"] = chunk[-1].step
            parsed["raw_response"] = raw
            step_results.append(parsed)

    best_step = select_ordinal_candidate(step_results) if step_results else best_chunk_candidate
    if best_step.get("step") is None:
        best_step = best_chunk_candidate

    agent = normalize_ordinal_agent(best_step.get("agent")) or target_agent
    step = best_step.get("step")

    return Prediction(
        case_id=case.case_id,
        method="paper_hybrid_ordinal10",
        agent=agent,
        step=int(step) if step is not None else None,
        confidence=None,
        reason="Target agent selected by highest ordinal chunk; final step selected by highest ordinal target-agent step.",
        trace={
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "target_agent": target_agent,
            "best_chunk_candidate": strip_raw_large([best_chunk_candidate])[0],
            "step_results": strip_raw_large(step_results),
            "selection_rule": "no agent score accumulation; target agent from highest chunk blame_score; final step from highest target-agent step blame_score",
        },
    )


def run_mvbs10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    beam_k = int(method_config.get("beam_k", 3))
    window_size = int(method_config.get("window_size", 5))
    stride = int(method_config.get("stride", 2))
    overlap_steps = int(method_config.get("overlap_steps", 2))
    max_candidates = int(method_config.get("max_candidates", 5))
    pairwise = bool(method_config.get("pairwise", True))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        prompt = mvbs_chunk_scoring_prompt(
            case=case,
            chunk_id=idx + 1,
            chunk_count=len(chunks),
            chunk=chunk,
            prev_summary=summaries[idx - 1] if idx > 0 else "None",
            next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
        )
        raw = llm.generate(prompt)
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = mvbs_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    ranked_chunks = sorted(chunk_results, key=lambda x: x.get("score", 0.0), reverse=True)
    selected_ids = {int(item["chunk_id"]) for item in ranked_chunks[:beam_k]}

    window_results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 not in selected_ids:
            continue
        region = expand_region(case.steps, chunk, overlap_steps=overlap_steps)
        for window in sliding_windows(region, window_size=window_size, stride=stride):
            before, after = steps_before_after(working_steps, window, context_steps=2)
            prompt = mvbs_window_prompt(case=case, region=region, window=window, before=before, after=after)
            raw = llm.generate(prompt)
            parsed = safe_json(raw)
            parsed["window_start"] = window[0].step
            parsed["window_end"] = window[-1].step
            parsed["region_start"] = region[0].step
            parsed["region_end"] = region[-1].step
            parsed["raw_response"] = raw
            window_results.append(parsed)
            if as_bool(parsed.get("contains_decisive_error")) and parsed.get("candidate_step") is not None:
                step = int(parsed["candidate_step"])
                candidates.append(
                    {
                        "step": step,
                        "agent": parsed.get("agent"),
                        "score": as_float(parsed.get("root_cause_score")),
                        "confidence": as_float(parsed.get("confidence")),
                        "reason": parsed.get("reason", ""),
                        "source": "window",
                        "content": step_content(window, step) or step_content(working_steps, step) or step_content(case.steps, step),
                    }
                )

    if not candidates:
        candidates.extend(chunk_level_candidates(chunk_results, working_steps))

    candidates = dedupe_candidates(candidates)
    candidates = sorted(candidates, key=lambda c: (c.get("score", 0.0), c.get("confidence", 0.0)), reverse=True)[:max_candidates]

    pairwise_results: list[dict[str, Any]] = []
    if pairwise and len(candidates) > 1:
        wins: Counter[int] = Counter()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                cand_a = candidates[i]
                cand_b = candidates[j]
                raw = llm.generate(
                    mvbs_pairwise_prompt(
                        case=case,
                        candidate_a=cand_a,
                        candidate_b=cand_b,
                        context_a=context_around(working_steps, int(cand_a["step"])),
                        context_b=context_around(working_steps, int(cand_b["step"])),
                    )
                )
                parsed = safe_json(raw)
                winner = str(parsed.get("winner", "")).strip().upper()
                if winner == "A":
                    wins[i] += 1
                elif winner == "B":
                    wins[j] += 1
                parsed.update({"candidate_a": cand_a, "candidate_b": cand_b, "raw_response": raw})
                pairwise_results.append(parsed)
        best_idx = wins.most_common(1)[0][0] if wins else 0
        best = candidates[best_idx]
    elif candidates:
        best = candidates[0]
    else:
        best = {"step": None, "agent": None, "confidence": None, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="mvbs10",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "chunk_results": strip_raw_large(chunk_results),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "selected_chunk_ids": sorted(selected_ids),
            "window_results": strip_raw_large(window_results),
            "candidates": candidates,
            "pairwise_results": strip_raw_large(pairwise_results),
        },
    )


def run_ccv10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    threshold = float(method_config.get("threshold", 0.65))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("contains_violation")) and float(result.get("score", 0.0)) >= threshold
    ]
    if viable:
        selected = sorted(viable, key=lambda x: (x["chunk_start"], -x["score"]))[0]
    else:
        selected = max(chunk_results, key=lambda x: x.get("score", 0.0))

    selected_chunk = chunks[int(selected["chunk_id"]) - 1]
    step_raw = llm.generate(ccv_step_prompt(case=case, constraints=constraints, chunk=selected_chunk))
    step_result = safe_json(step_raw)

    step = step_result.get("step") or selected.get("earliest_suspected_step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method="ccv10",
        agent=normalize_optional_str(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=as_float(step_result.get("confidence") or selected.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "step_result": {**step_result, "raw_response": step_raw},
        },
    )


def run_ccv_adaptive(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.65))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("contains_violation")) and float(result.get("score", 0.0)) >= threshold
    ]
    if viable:
        selected = sorted(viable, key=lambda x: (x["chunk_start"], -x["score"]))[0]
    else:
        selected = max(chunk_results, key=lambda x: x.get("score", 0.0))

    selected_chunk = chunks[int(selected["chunk_id"]) - 1]
    step_raw = llm.generate(ccv_step_prompt(case=case, constraints=constraints, chunk=selected_chunk))
    step_result = safe_json(step_raw)

    step = step_result.get("step") or selected.get("earliest_suspected_step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method="ccv_adaptive",
        agent=normalize_optional_str(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=as_float(step_result.get("confidence") or selected.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "step_result": {**step_result, "raw_response": step_raw},
            "selection_rule": "CCV constraint scoring with adaptive chunk granularity",
        },
    )


def run_ccv_adaptive_context(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.65))
    context_steps = int(method_config.get("context_steps", 2))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    chunking["context_steps"] = context_steps
    chunking["context_policy"] = "read_only_before_after_context"
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_ccv_context_stage(case, constraints, working_steps, chunks, summaries, llm, context_steps)

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("contains_violation")) and float(result.get("score", 0.0)) >= threshold
    ]
    if viable:
        selected = sorted(viable, key=lambda x: (x["chunk_start"], -x["score"]))[0]
    else:
        selected = max(chunk_results, key=lambda x: x.get("score", 0.0))

    selected_chunk = chunks[int(selected["chunk_id"]) - 1]
    before_context, after_context = steps_before_after(working_steps, selected_chunk, context_steps)
    step_raw = llm.generate(
        ccv_step_context_prompt(
            case=case,
            constraints=constraints,
            chunk=selected_chunk,
            before_context=before_context,
            after_context=after_context,
        )
    )
    step_result = safe_json(step_raw)
    if not step_inside_chunk(step_result.get("step"), selected_chunk):
        step_result["ignored_out_of_focal_step"] = step_result.get("step")
        step_result["step"] = None

    step = step_result.get("step") or selected.get("earliest_suspected_step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method="ccv_adaptive_context",
        agent=normalize_optional_str(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=as_float(step_result.get("confidence") or selected.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "step_result": {**step_result, "raw_response": step_raw},
            "selection_rule": "CCV constraint scoring with adaptive chunk granularity and read-only before/after context",
        },
    )


def run_ccv_full_trace(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(case, llm, method_config)

    working_steps = case.steps
    raw = llm.generate(ccv_full_trace_prompt(case=case, constraints=constraints, steps=working_steps))
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="ccv_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "CCV full-trace baseline: synthesize task-success constraints, then read the full trace once "
                "to directly select the earliest unrecovered constraint violation."
            ),
        },
    )


def run_ccv_causal_no_requirements(case: Case, llm: LLM) -> Prediction:
    working_steps = case.steps
    raw = llm.generate(
        ccv_causal_no_requirements_prompt(
            case=case,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_r0c1_causal_no_requirements",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "requirements": None,
            "ground_truth_visible_to_judge": bool(case.ground_truth),
            "explicit_final_answer_visible_to_judge": bool(case.final_answer),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "R0C1 causal-localization ablation: preserve the R1C1 hidden-cause, minimal-correction, "
                "recovery, and earliest-error procedure while removing explicit task-success requirements "
                "and requirement-specific terminology."
            ),
        },
    )


def run_ccv_requirements_direct(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        case,
        llm,
        method_config,
    )
    working_steps = case.steps
    raw = llm.generate(
        ccv_requirements_direct_prompt(
            case=case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_r1c0_requirements_direct",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "ground_truth_visible_to_judge": bool(case.ground_truth),
            "explicit_final_answer_visible_to_judge": bool(case.final_answer),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "R1C0 direct-attribution ablation: reuse the exact frozen R1C1 task-success requirements "
                "while removing hidden-cause analysis, minimal corrective intervention, outcome-change "
                "prediction, and explicit recovery analysis. Preserve the benchmark-level objective of "
                "selecting the earliest decisive error rather than a later downstream consequence."
            ),
        },
    )


def run_tsr_task_r1_direct(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run the canonical R1-Direct judge with frozen task-only requirements."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        judge_case,
        llm,
        method_config,
    )
    working_steps = case.steps
    raw = llm.generate(
        ccv_requirements_direct_prompt(
            case=judge_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_task_r1_direct",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "requirement_source": "frozen_task_only_generation",
            "ground_truth_visible_to_constraint_generator": False,
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Task-only R1-Direct: reuse the exact R1-Direct prompt and output schema with frozen "
                "requirements generated from task-only inputs. Ground truth and the final system answer "
                "are hidden from both requirement generation and direct localization. No hidden-cause, "
                "minimal-intervention, counterfactual-prediction, or explicit recovery scaffold is added."
            ),
        },
    )


def run_tsr_direct_no_requirements(
    case: Case,
    llm: LLM,
) -> Prediction:
    """Run task-only Direct localization after minimally deleting requirements."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    working_steps = case.steps
    raw = llm.generate(
        tsr_direct_no_requirements_prompt(
            case=judge_case,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_direct_no_requirements",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "requirements": None,
            "case_fingerprint": case_fingerprint(judge_case),
            "prompt_family": "tsr_direct_minimal_requirement_deletion_v1",
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Minimal-deletion R0 Direct ablation: start from task-only TSR-Loc Direct and remove "
                "the requirement-guided modifier, requirement block, requirement-selection clause, "
                "and requirement-specific output fields. Preserve the task-only problem, agent "
                "inventory, full trace, generic Direct attribution rules, model, and decoding settings."
            ),
        },
    )


def run_tsr_minimal_r0(
    case: Case,
    llm: LLM,
) -> Prediction:
    """Run the agent-and-step-only condition without requirements."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    working_steps = case.steps
    raw = llm.generate(
        tsr_minimal_r0_prompt(
            case=judge_case,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_minimal_r0",
        agent=agent,
        step=step,
        confidence=None,
        reason=None,
        trace={
            "requirements": None,
            "case_fingerprint": case_fingerprint(judge_case),
            "prompt_family": "tsr_agent_step_minimal_pair_v1",
            "condition": "R0_no_requirements",
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Matched minimal-pair R0: task, agent inventory, full trace, earliest-decisive-error "
                "objective, and agent/step-only output. No task-success requirement block or "
                "requirement-use instruction."
            ),
        },
    )


def run_tsr_minimal_r1(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run the matched agent-and-step-only condition with frozen requirements."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    requirements, requirements_raw, requirement_state = resolve_ccv_constraints(
        judge_case,
        llm,
        method_config,
    )
    working_steps = case.steps
    raw = llm.generate(
        tsr_minimal_r1_prompt(
            case=judge_case,
            requirements=requirements,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_minimal_r1",
        agent=agent,
        step=step,
        confidence=None,
        reason=None,
        trace={
            "requirements": requirements,
            "requirements_raw_response": requirements_raw,
            **requirement_state,
            "case_fingerprint": case_fingerprint(judge_case),
            "prompt_family": "tsr_agent_step_minimal_pair_v1",
            "condition": "R1_with_requirements",
            "requirement_source": "frozen_task_only_generation",
            "ground_truth_visible_to_constraint_generator": False,
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Matched minimal-pair R1: identical to R0 except for the frozen task-success "
                "requirements block and the instruction to use it as diagnostic references."
            ),
        },
    )


def run_ccv_trace_elephant_full_trace(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(case, llm, method_config)

    working_steps = case.steps
    raw = llm.generate(
        ccv_trace_elephant_full_trace_prompt(
            case=case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="ccv_trace_elephant_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "TraceElephant-aligned CCV: use task-success constraints and the full trace to select the earliest "
                "role-aware, recoverability-aware point where task failure becomes inevitable."
            ),
        },
    )


def run_tsr_trace_elephant_causal(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run task-only TSR-Loc under TraceElephant's recoverability-aware target."""
    task_only_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        task_only_case,
        llm,
        method_config,
    )

    max_full_trace_chars = int(method_config.get("max_full_trace_chars", 0))
    working_steps, context_compaction = compact_steps_to_total_char_budget(
        case.steps,
        max_full_trace_chars,
    )
    raw = llm.generate(
        ccv_trace_elephant_full_trace_prompt(
            case=task_only_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_trace_elephant_causal",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "requirement_source": "task_only_generation",
            "ground_truth_visible_to_constraint_generator": False,
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "context_safety_compaction": context_compaction,
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "TSR-Loc Causal external transfer: generate task-success requirements without the reference "
                "answer or final system answer, then apply TraceElephant's role- and recoverability-aware "
                "earliest-inevitable-failure criterion to the compact trace. A deterministic total-character "
                "safety cap preserves every step boundary and agent while shortening only oversized step bodies."
            ),
        },
    )


def run_tsr_trace_elephant_original(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Transfer the frozen task-only TSR-Loc prompts without target adaptation."""
    task_only_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        task_only_case,
        llm,
        method_config,
    )

    max_full_trace_chars = int(method_config.get("max_full_trace_chars", 0))
    working_steps, context_compaction = compact_steps_to_total_char_budget(
        case.steps,
        max_full_trace_chars,
    )
    raw = llm.generate(
        ccv_full_trace_prompt(
            case=task_only_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_trace_elephant_original",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "requirement_source": "task_only_generation",
            "ground_truth_visible_to_constraint_generator": False,
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "context_safety_compaction": context_compaction,
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Frozen task-only TSR-Loc transfer: reuse the canonical task-success requirement generation "
                "and earliest-unrecovered-violation full-trace localization prompts without TraceElephant-specific "
                "role, recoverability, inevitability, or component-target adaptation. A deterministic total-character "
                "safety cap preserves every step boundary and agent while shortening only oversized step bodies."
            ),
        },
    )


def run_tsr_trace_elephant_gt_assisted(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run canonical TSR-Loc with ground truth as the only added side information."""
    gt_case = case_with_side_information(
        case,
        include_ground_truth=True,
        include_final_answer=False,
    )
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        gt_case,
        llm,
        method_config,
    )

    max_full_trace_chars = int(method_config.get("max_full_trace_chars", 0))
    working_steps, context_compaction = compact_steps_to_total_char_budget(
        case.steps,
        max_full_trace_chars,
    )
    raw = llm.generate(
        ccv_full_trace_prompt(
            case=gt_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="tsr_trace_elephant_gt_assisted",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "requirement_source": "ground_truth_assisted_generation",
            "ground_truth_visible_to_constraint_generator": True,
            "ground_truth_visible_to_judge": True,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "context_safety_compaction": context_compaction,
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Ground-truth-assisted TSR-Loc control: reuse the canonical requirement-generation and "
                "earliest-unrecovered-violation prompts from the task-only condition. The reference answer is "
                "the only added side information; the explicit final system answer remains hidden. The same "
                "deterministic trace safety cap is applied."
            ),
        },
    )


def compact_steps_to_total_char_budget(
    steps: list[LogStep],
    max_total_chars: int,
) -> tuple[list[LogStep], dict[str, Any]]:
    """Shorten oversized step bodies while preserving every step and agent."""
    original_chars = sum(len(step.content) for step in steps)
    if max_total_chars <= 0 or original_chars <= max_total_chars or not steps:
        return list(steps), {
            "applied": False,
            "policy": "head_tail_per_step_waterfill",
            "max_total_chars": max_total_chars,
            "original_content_chars": original_chars,
            "compacted_content_chars": original_chars,
            "truncated_step_count": 0,
        }

    lengths = [len(step.content) for step in steps]
    low, high = 0, max(lengths)
    while low < high:
        midpoint = (low + high + 1) // 2
        if sum(min(length, midpoint) for length in lengths) <= max_total_chars:
            low = midpoint
        else:
            high = midpoint - 1
    per_step_cap = max(1, low)

    compacted: list[LogStep] = []
    truncated_step_count = 0
    marker = "\n...[context-safety truncation; middle omitted]...\n"
    for step in steps:
        content = step.content
        if len(content) <= per_step_cap:
            compacted.append(step)
            continue
        truncated_step_count += 1
        payload_chars = max(1, per_step_cap - len(marker))
        head_chars = max(1, int(payload_chars * 0.6))
        tail_chars = max(0, payload_chars - head_chars)
        shortened = content[:head_chars].rstrip() + marker
        if tail_chars:
            shortened += content[-tail_chars:].lstrip()
        compacted.append(LogStep(step=step.step, agent=step.agent, content=shortened))

    compacted_chars = sum(len(step.content) for step in compacted)
    return compacted, {
        "applied": True,
        "policy": "head_tail_per_step_waterfill",
        "max_total_chars": max_total_chars,
        "per_step_char_cap": per_step_cap,
        "original_content_chars": original_chars,
        "compacted_content_chars": compacted_chars,
        "truncated_step_count": truncated_step_count,
        "preserved_step_count": len(compacted),
    }


def run_ccv_constraint_only_full_trace(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(case, llm, method_config)

    working_steps = case.steps
    raw = llm.generate(
        ccv_constraint_only_full_trace_prompt(
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="ccv_constraint_only_full_trace",
        agent=agent,
        step=step,
        confidence=None,
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "judge_visible_information": ["frozen_constraints", "full_conversation"],
            "judge_hidden_information": ["problem_header", "ground_truth_header", "final_answer_header"],
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Neutral constraint-only judge: select one agent-step using only frozen constraints and explicit "
                "trace evidence, without temporal, root-cause, recoverability, or counterfactual guidance."
            ),
        },
    )


def run_ccv_information_ablation(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
    *,
    method_name: str,
    ground_truth_to_generator: bool,
    ground_truth_to_judge: bool,
) -> Prediction:
    """Run CCV unchanged while controlling GT visibility at its two LLM stages."""
    generator_case = case_with_side_information(
        case,
        include_ground_truth=ground_truth_to_generator,
        include_final_answer=False,
    )
    judge_case = case_with_side_information(
        case,
        include_ground_truth=ground_truth_to_judge,
        include_final_answer=False,
    )
    before_requirements = get_usage_snapshot(llm)
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(
        generator_case,
        llm,
        method_config,
    )
    after_requirements = get_usage_snapshot(llm)

    working_steps = case.steps
    raw = llm.generate(
        ccv_full_trace_prompt(
            case=judge_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    after_localization = get_usage_snapshot(llm)
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method=method_name,
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "ground_truth_visible_to_constraint_generator": ground_truth_to_generator,
            "ground_truth_visible_to_judge": ground_truth_to_judge,
            "explicit_final_answer_visible_to_constraint_generator": False,
            "explicit_final_answer_visible_to_judge": False,
            "source_case_has_explicit_final_answer": bool(case.final_answer),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "stage_usage": {
                "requirement_generation": usage_delta(
                    before_requirements,
                    after_requirements,
                ),
                "failure_localization": usage_delta(
                    after_requirements,
                    after_localization,
                ),
            },
            "selection_rule": (
                "CCV information ablation: preserve constraint generation and full-trace localization prompts "
                "while controlling ground-truth visibility independently at the generator and judge stages."
            ),
        },
    )


def run_tsr_requirement_content_ablation(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
    *,
    method_name: str,
    requirement_condition: str,
) -> Prediction:
    """Keep the task-only CCV localizer fixed while changing only requirement content."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    fingerprint = case_fingerprint(judge_case)

    if requirement_condition == "none":
        constraints: list[dict[str, Any]] = []
        constraints_raw = json_dumps_compact({"constraints": constraints})
        constraint_state: dict[str, Any] = {
            "constraint_source": "empty_requirement_ablation",
            "constraint_cache_path": None,
            "case_fingerprint": fingerprint,
            "constraint_fingerprint": constraint_fingerprint(constraints),
            "donor_case_id": None,
            "donor_case_fingerprint": None,
        }
    elif requirement_condition == "shuffled":
        cache_path = normalize_optional_str(method_config.get("constraint_cache_path"))
        if not cache_path:
            raise ValueError(
                f"{method_name} requires methods.{method_name}.constraint_cache_path"
            )
        cache = load_ccv_constraint_cache(cache_path)
        entry = cache.get(fingerprint)
        if entry is None:
            raise KeyError(
                "No shuffled requirements for this task-only trajectory fingerprint: "
                f"case_id={case.case_id}, fingerprint={fingerprint}, cache={cache_path}"
            )
        constraints_value = entry.get("constraints")
        if not isinstance(constraints_value, list) or not constraints_value or not all(
            isinstance(constraint, dict) for constraint in constraints_value
        ):
            raise ValueError(
                "Shuffled requirement cache contains an invalid requirement set: "
                f"case_id={case.case_id}, fingerprint={fingerprint}, cache={cache_path}"
            )
        constraints = constraints_value
        raw_value = entry.get("constraints_raw_response")
        constraints_raw = (
            raw_value
            if isinstance(raw_value, str) and raw_value.strip()
            else json_dumps_compact({"constraints": constraints})
        )
        constraint_state = {
            "constraint_source": "deterministic_shuffled_cache",
            "constraint_cache_path": cache_path,
            "case_fingerprint": fingerprint,
            "constraint_fingerprint": constraint_fingerprint(constraints),
            "donor_case_id": normalize_optional_str(entry.get("donor_case_id")),
            "donor_case_fingerprint": normalize_optional_str(
                entry.get("donor_case_fingerprint")
            ),
            "shuffle_group": normalize_optional_str(entry.get("shuffle_group")),
            "shuffle_policy": normalize_optional_str(entry.get("shuffle_policy")),
            "target_requirement_chars": entry.get("target_requirement_chars"),
            "donor_requirement_chars": entry.get("donor_requirement_chars"),
        }
        if constraint_state["donor_case_fingerprint"] == fingerprint:
            raise ValueError(
                f"Shuffled requirement donor equals target trajectory: {case.case_id}"
            )
    else:
        raise ValueError(f"Unknown requirement_condition: {requirement_condition}")

    working_steps = case.steps
    raw = llm.generate(
        ccv_full_trace_prompt(
            case=judge_case,
            constraints=constraints,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method=method_name,
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "requirement_ablation_condition": requirement_condition,
            "ground_truth_visible_to_judge": False,
            "explicit_final_answer_visible_to_judge": False,
            "source_case_has_explicit_final_answer": bool(case.final_answer),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Strict requirement-content ablation: reuse the task-only CCV full-trace localizer unchanged "
                "and vary only whether its requirement list is empty or deterministically shuffled."
            ),
        },
    )


def run_direct_requirements_full_trace(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    """Run one-pass direct attribution with frozen task-derived requirements."""
    judge_case = case_with_side_information(
        case,
        include_ground_truth=False,
        include_final_answer=False,
    )
    requirements, requirements_raw, requirement_state = resolve_ccv_constraints(
        judge_case,
        llm,
        method_config,
    )
    working_steps = case.steps
    raw = llm.generate(
        direct_requirements_full_trace_prompt(
            case=judge_case,
            requirements=requirements,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="direct_requirements_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "requirements": requirements,
            "requirements_raw_response": requirements_raw,
            **requirement_state,
            "ground_truth_visible": False,
            "explicit_final_answer_visible": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "One-call direct full-trace attribution using frozen task-derived requirements. "
                "No staged causal scan, hidden-cause analysis, intervention construction, or "
                "counterfactual prediction is requested."
            ),
        },
    )


def run_ccv_checklist_equivalent_full_trace(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> Prediction:
    del method_config

    checklist_raw = llm.generate(ccv_checklist_equivalent_generation_prompt(case))
    checklist_obj = safe_json(checklist_raw)
    checklist_items = extract_checklist_items(checklist_obj)
    used_default = not isinstance(checklist_items, list) or not checklist_items
    if used_default:
        checklist_items = default_checklist_items()

    working_steps = case.steps
    raw = llm.generate(
        ccv_checklist_equivalent_full_trace_prompt(
            case=case,
            checklist_items=checklist_items,
            steps=working_steps,
        )
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="ccv_checklist_equivalent_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "checklist_items": checklist_items,
            "checklist_raw_response": checklist_raw,
            "checklist_source": "ccv_terminology_equivalent",
            "checklist_used_default": used_default,
            "case_fingerprint": case_fingerprint(case),
            "checklist_fingerprint": constraint_fingerprint(checklist_items),
            "ground_truth_visible_to_checklist_generator": bool(case.ground_truth),
            "ground_truth_visible_to_judge": bool(case.ground_truth),
            "final_answer_field_visible_to_checklist_generator": bool(case.final_answer),
            "final_answer_field_visible_to_judge": bool(case.final_answer),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "CCV checklist-equivalent full-trace: preserve CCV inputs and decision procedure while "
                "replacing constraint-violation terminology with checklist-breakdown terminology."
            ),
        },
    )


def run_task_checklist_full_trace(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    del method_config

    checklist_raw = llm.generate(task_checklist_generation_prompt(case))
    checklist_obj = safe_json(checklist_raw)
    checklist_items = extract_checklist_items(checklist_obj)
    used_default = not isinstance(checklist_items, list) or not checklist_items
    if used_default:
        checklist_items = default_checklist_items()

    working_steps = case.steps
    raw = llm.generate(
        task_checklist_full_trace_prompt(case=case, checklist_items=checklist_items, steps=working_steps)
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="task_checklist_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "checklist_items": checklist_items,
            "checklist_raw_response": checklist_raw,
            "checklist_source": "generated_from_task_only",
            "checklist_used_default": used_default,
            "case_fingerprint": case_fingerprint(case),
            "checklist_fingerprint": constraint_fingerprint(checklist_items),
            "ground_truth_visible_to_checklist_generator": False,
            "ground_truth_visible_to_judge": False,
            "final_answer_field_visible_to_checklist_generator": False,
            "final_answer_field_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Task-checklist full-trace: synthesize a success checklist from the problem only, then read "
                "the full trace without a ground-truth field to select the earliest unrecovered checklist breakdown."
            ),
        },
    )


def run_reference_checklist_full_trace(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    del method_config
    if not case.ground_truth:
        raise ValueError(
            f"reference_checklist_full_trace requires a ground-truth answer: case_id={case.case_id}"
        )

    checklist_raw = llm.generate(reference_checklist_generation_prompt(case))
    checklist_obj = safe_json(checklist_raw)
    checklist_items = extract_checklist_items(checklist_obj)
    used_default = not isinstance(checklist_items, list) or not checklist_items
    if used_default:
        checklist_items = default_checklist_items()

    working_steps = case.steps
    raw = llm.generate(
        reference_checklist_full_trace_prompt(case=case, checklist_items=checklist_items, steps=working_steps)
    )
    parsed = safe_json(raw)
    step = parsed_step(parsed)
    if step is not None and not step_inside_chunk(step, working_steps):
        parsed["ignored_out_of_trace_step"] = step
        step = None
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="reference_checklist_full_trace",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason")),
        trace={
            "checklist_items": checklist_items,
            "checklist_raw_response": checklist_raw,
            "checklist_source": "generated_from_task_and_reference",
            "checklist_used_default": used_default,
            "case_fingerprint": case_fingerprint(case),
            "checklist_fingerprint": constraint_fingerprint(checklist_items),
            "ground_truth_visible_to_checklist_generator": True,
            "ground_truth_visible_to_judge": True,
            "final_answer_field_visible_to_checklist_generator": False,
            "final_answer_field_visible_to_judge": False,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "judge_result": {**parsed, "raw_response": raw},
            "selection_rule": (
                "Reference-checklist full-trace: compile the ground-truth answer into diagnostic success "
                "checkpoints, then select the earliest unrecovered checklist breakdown from the full trace."
            ),
        },
    )


def run_ccv_scalar10(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
    method_name: str = "ccv_scalar10",
) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    threshold = float(method_config.get("threshold", 0.65))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_scalar_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_scalar_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("contains_violation")) and float(result.get("score", 0.0)) >= threshold
    ]
    if viable:
        selected = sorted(viable, key=lambda x: (x["chunk_start"], -x["score"]))[0]
    else:
        selected = max(chunk_results, key=lambda x: x.get("score", 0.0))

    selected_chunk = chunks[int(selected["chunk_id"]) - 1]
    step_raw = llm.generate(ccv_step_bool_prompt(case=case, constraints=constraints, chunk=selected_chunk))
    step_result = safe_json(step_raw)

    step = step_result.get("step") or selected.get("earliest_suspected_step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method=method_name,
        agent=normalize_optional_str(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=None,
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "step_result": {**step_result, "raw_response": step_raw},
            "selection_rule": "single LLM root_cause_score per chunk",
        },
    )


def run_ccv_ordinal10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_ccv_ordinal_stage(case, constraints, chunks, summaries, llm)
    selected = select_ordinal_candidate(chunk_results)

    selected_chunk = chunks[int(selected["chunk_id"]) - 1] if selected.get("chunk_id") is not None else chunks[0]
    step_raw = llm.generate(ccv_step_bool_prompt(case=case, constraints=constraints, chunk=selected_chunk))
    step_result = safe_json(step_raw)

    step = step_result.get("step") or selected.get("step")
    agent = step_result.get("agent") or selected.get("agent")

    return Prediction(
        case_id=case.case_id,
        method="ccv_ordinal10",
        agent=normalize_ordinal_agent(agent),
        step=int(step) if step is not None and str(step) != "" else None,
        confidence=None,
        reason=normalize_optional_str(step_result.get("reason") or selected.get("reason"))
        or "Chunk selected by ordinal blame_score, then localized inside selected chunk.",
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_id": selected.get("chunk_id"),
            "selected_chunk_candidate": strip_raw_large([selected])[0],
            "step_result": {**step_result, "raw_response": step_raw},
            "selection_rule": "chunk selected by highest ordinal blame_score; final step localized inside selected chunk without confidence",
        },
    )


def run_ccv_beam10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    threshold = float(method_config.get("threshold", 0.65))
    beam_k = int(method_config.get("beam_k", 3))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    selected_chunks = select_ccv_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        step_raw = llm.generate(ccv_step_prompt(case=case, constraints=constraints, chunk=selected_chunk))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        step_results.append(step_result)

        step = step_result.get("step") or selected.get("earliest_suspected_step")
        agent = step_result.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "step_confidence": as_float(step_result.get("confidence") or selected.get("confidence")),
                "score": ccv_beam_candidate_score(selected, step_result),
                "violated_constraint": step_result.get("violated_constraint"),
                "violation_type": step_result.get("violation_type"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "step_confidence": None, "reason": "No candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_beam10",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("step_confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
        },
    )


def run_ccv_adaptive_beam(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.65))
    beam_k = int(method_config.get("beam_k", 3))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    selected_chunks = select_ccv_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        step_raw = llm.generate(ccv_step_prompt(case=case, constraints=constraints, chunk=selected_chunk))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        step_results.append(step_result)

        step = step_result.get("step") or selected.get("earliest_suspected_step")
        agent = step_result.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "step_confidence": as_float(step_result.get("confidence") or selected.get("confidence")),
                "score": ccv_beam_candidate_score(selected, step_result),
                "violated_constraint": step_result.get("violated_constraint"),
                "violation_type": step_result.get("violation_type"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "step_confidence": None, "reason": "No candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_adaptive_beam",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("step_confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "CCV beam with adaptive chunk granularity",
        },
    )


def run_ccv_global_router_beam(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    constraints, constraints_raw, constraint_state = resolve_ccv_constraints(case, llm, method_config)

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["router_policy"] = "whole_trace_constraint_guided_chunk_router"
    chunking["reread_policy"] = "selected_chunks_joint_reread_localization"

    if len(chunks) <= 2:
        selected_chunks = [
            {
                "chunk_id": idx + 1,
                "chunk_start": chunk[0].step,
                "chunk_end": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
                "score": 0.0,
                "confidence": 0.0,
                "reason": "Short-trace fallback: localize over all produced chunks.",
                "selection_source": "short_trace_fallback",
            }
            for idx, chunk in enumerate(chunks)
        ]
        router_result: dict[str, Any] = {
            "skipped": True,
            "reason": "Trace produced at most two adaptive chunks; routing is unnecessary.",
        }
        router_candidates: list[dict[str, Any]] = []
    else:
        conversation = render_steps(working_steps)
        router_raw = llm.generate(
            ccv_global_chunk_router_prompt(
                case=case,
                constraints=constraints,
                conversation=conversation,
                chunk_ranges=chunking.get("chunk_ranges", []),
                beam_k=beam_k,
            )
        )
        router_result = safe_json(router_raw)
        router_result["raw_response"] = router_raw
        selected_token_budget = int(method_config.get("selected_token_budget", 0))
        selected_chunks, router_candidates = select_echo_global_router_chunks(
            router_result,
            chunks,
            chunking,
            beam_k,
            selected_token_budget=selected_token_budget if selected_token_budget > 0 else None,
        )
        if selected_token_budget > 0:
            chunking["selected_token_budget"] = selected_token_budget
            chunking["selected_token_budget_actual"] = sum(
                int(item.get("estimated_tokens") or 0) for item in selected_chunks
            )
            chunking["selected_token_budget_policy"] = "router_order_closest_prefix"

    selected_chunk_payload: list[dict[str, Any]] = []
    for selected in selected_chunks:
        chunk_id = parse_int_maybe(selected.get("chunk_id"))
        if chunk_id is None or chunk_id < 1 or chunk_id > len(chunks):
            continue
        selected_chunk_payload.append(
            {
                **selected,
                "chunk": chunks[chunk_id - 1],
            }
        )

    if selected_chunk_payload:
        joint_raw = llm.generate(
            ccv_selected_chunks_joint_prompt(
                case=case,
                constraints=constraints,
                selected_chunks=selected_chunk_payload,
            )
        )
        joint_result = safe_json(joint_raw)
        joint_result["raw_response"] = joint_raw
        selected_ranges = [
            (int(item["chunk"][0].step), int(item["chunk"][-1].step))
            for item in selected_chunk_payload
            if item.get("chunk")
        ]
        step = parse_int_maybe(joint_result.get("step"))
        if step is not None and any(start <= step <= end for start, end in selected_ranges):
            best = {
                "step": step,
                "agent": normalize_optional_str(joint_result.get("agent")),
                "violated_constraint": joint_result.get("violated_constraint"),
                "violation_type": joint_result.get("violation_type"),
                "reason": normalize_optional_str(joint_result.get("reason")),
            }
            joint_result["valid_selected_step"] = True
        else:
            best = {
                "step": None,
                "agent": None,
                "reason": "Joint selected-chunk reread produced no usable step inside selected chunks.",
            }
            joint_result["valid_selected_step"] = False
    else:
        best = {"step": None, "agent": None, "reason": "No selected chunk available for joint reread."}
        joint_result = {"skipped": True, "reason": "No selected chunk available for joint reread."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_global_router_beam",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            **constraint_state,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "router_result": {k: v for k, v in router_result.items() if k != "raw_response"},
            "router_candidates": router_candidates,
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "joint_reread_result": {k: v for k, v in joint_result.items() if k != "raw_response"},
            "step_results": [],
            "candidates": [],
            "rerank_result": {"skipped": True, "reason": "Selected chunks were reread jointly; no candidate rerank stage was used."},
            "selection_rule": (
                "CCV global router reads the whole trace with synthesized constraints, selects chunk IDs, "
                "then rereads the selected chunks together in temporal order to localize one final step."
            ),
        },
    )


def run_cgv_full_step_judge(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    max_judge_violations = int(method_config.get("max_judge_violations", 0))
    max_validation_steps = int(method_config.get("max_validation_steps", 0))
    validation_context_steps = int(method_config.get("validation_context_steps", 2))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = case.steps
    validation_steps = working_steps[:max_validation_steps] if max_validation_steps > 0 else working_steps
    step_records, validation_log = run_cgv_step_validation_stage(
        case=case,
        constraints=constraints,
        validation_steps=validation_steps,
        evidence_steps=working_steps,
        evidence_label="Local validation window around the current step",
        llm=llm,
        context_steps=validation_context_steps,
    )
    judge_log = select_validation_log_for_judge(validation_log, max_judge_violations)
    judge_raw = llm.generate(
        cgv_final_judge_prompt(
            case=case,
            constraints=constraints,
            evidence_steps=working_steps,
            validation_log=judge_log,
            evidence_label="Full conversation",
        )
    )
    judge = safe_json(judge_raw)
    step = parsed_step(judge)
    if step is not None and not step_inside_any(step, [working_steps]):
        judge["ignored_out_of_evidence_step"] = step
        step = None
    if step is None:
        fallback = select_cgv_fallback(validation_log, working_steps)
        step = parse_int_maybe(fallback.get("step")) if fallback else None
        if fallback:
            judge["fallback_candidate"] = fallback
    agent = parsed_agent(judge) or normalize_optional_str(judge.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="cgv_full_step_judge",
        agent=agent,
        step=step,
        confidence=as_float(judge.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(judge.get("reason") or judge.get("reason_for_failure")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "validation_scope": "local_window_per_step_then_full_trace_judge",
            "validation_context_steps": validation_context_steps,
            "step_records": strip_raw_large(step_records),
            "validation_log": validation_log,
            "judge_validation_log": judge_log,
            "judge_result": {**judge, "raw_response": judge_raw},
            "selection_rule": (
                "CGV full-step judge: synthesize task-success constraints, validate every step against a "
                "local evidence window, then let a final judge localize the earliest unrecovered root cause "
                "from the full trace and validation log."
            ),
        },
    )


def run_cgv_global_router_step_judge(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    max_judge_violations = int(method_config.get("max_judge_violations", 0))
    validation_context_steps = int(method_config.get("validation_context_steps", 2))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    chunking["router_policy"] = "whole_trace_constraint_guided_step_validation_router"
    chunking["validation_policy"] = "stepwise_validation_only_inside_selected_chunks"
    chunking["judge_policy"] = "selected_chunks_plus_selected_validation_log"

    if len(chunks) <= 2:
        selected_chunks = [
            {
                "chunk_id": idx + 1,
                "chunk_start": chunk[0].step,
                "chunk_end": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
                "score": 0.0,
                "confidence": 0.0,
                "reason": "Short-trace fallback: validate all produced chunks.",
                "selection_source": "short_trace_fallback",
            }
            for idx, chunk in enumerate(chunks)
        ]
        router_result: dict[str, Any] = {
            "skipped": True,
            "reason": "Trace produced at most two adaptive chunks; routing is unnecessary.",
        }
        router_candidates: list[dict[str, Any]] = []
    else:
        conversation = render_steps(working_steps)
        router_raw = llm.generate(
            ccv_global_chunk_router_prompt(
                case=case,
                constraints=constraints,
                conversation=conversation,
                chunk_ranges=chunking.get("chunk_ranges", []),
                beam_k=beam_k,
            )
        )
        router_result = safe_json(router_raw)
        router_result["raw_response"] = router_raw
        selected_token_budget = int(method_config.get("selected_token_budget", 0))
        selected_chunks, router_candidates = select_echo_global_router_chunks(
            router_result,
            chunks,
            chunking,
            beam_k,
            selected_token_budget=selected_token_budget if selected_token_budget > 0 else None,
        )
        if selected_token_budget > 0:
            chunking["selected_token_budget"] = selected_token_budget
            chunking["selected_token_budget_actual"] = sum(
                int(item.get("estimated_tokens") or 0) for item in selected_chunks
            )
            chunking["selected_token_budget_policy"] = "router_order_closest_prefix"

    selected_chunks = sorted(selected_chunks, key=lambda x: int(x.get("chunk_start", 10**9)))
    selected_step_blocks: list[list[LogStep]] = []
    selected_step_records: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []

    for selected in selected_chunks:
        chunk_id = parse_int_maybe(selected.get("chunk_id"))
        if chunk_id is None or chunk_id < 1 or chunk_id > len(chunks):
            continue
        chunk = chunks[chunk_id - 1]
        selected_step_blocks.append(chunk)
        step_records, chunk_validation_log = run_cgv_step_validation_stage(
            case=case,
            constraints=constraints,
            validation_steps=chunk,
            evidence_steps=chunk,
            evidence_label=f"Selected Chunk {chunk_id}, steps {chunk[0].step}-{chunk[-1].step}",
            llm=llm,
            source_chunk_id=chunk_id,
            context_steps=validation_context_steps,
        )
        selected_step_records.extend(step_records)
        validation_log.extend(chunk_validation_log)

    evidence_steps = flatten_step_blocks(selected_step_blocks)
    judge_log = select_validation_log_for_judge(validation_log, max_judge_violations)
    judge_raw = llm.generate(
        cgv_final_judge_prompt(
            case=case,
            constraints=constraints,
            evidence_steps=evidence_steps,
            validation_log=judge_log,
            evidence_label="Selected chunks in temporal order",
        )
    )
    judge = safe_json(judge_raw)
    step = parsed_step(judge)
    if step is not None and not step_inside_any(step, selected_step_blocks):
        judge["ignored_out_of_selected_chunk_step"] = step
        step = None
    if step is None:
        fallback = select_cgv_fallback(validation_log, evidence_steps)
        step = parse_int_maybe(fallback.get("step")) if fallback else None
        if fallback:
            judge["fallback_candidate"] = fallback
    agent = parsed_agent(judge) or normalize_optional_str(judge.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(working_steps, step)

    return Prediction(
        case_id=case.case_id,
        method="cgv_global_router_step_judge",
        agent=agent,
        step=step,
        confidence=as_float(judge.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(judge.get("reason") or judge.get("reason_for_failure")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "validation_context_steps": validation_context_steps,
            "router_result": {k: v for k, v in router_result.items() if k != "raw_response"},
            "router_candidates": router_candidates,
            "selected_chunks": selected_chunks,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_records": strip_raw_large(selected_step_records),
            "validation_log": validation_log,
            "judge_validation_log": judge_log,
            "judge_result": {**judge, "raw_response": judge_raw},
            "selection_rule": (
                "CGV CAW-GR: synthesize task-success constraints, use a global router to select chunks for "
                "step validation, validate selected-chunk steps against local evidence windows, then judge "
                "from selected raw chunks and the selected validation log."
            ),
        },
    )


def run_ccv_adaptive_beam_context(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.65))
    beam_k = int(method_config.get("beam_k", 3))
    context_steps = int(method_config.get("context_steps", 2))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    chunking["context_steps"] = context_steps
    chunking["context_policy"] = "read_only_before_after_context"
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_ccv_context_stage(case, constraints, working_steps, chunks, summaries, llm, context_steps)

    selected_chunks = select_ccv_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        before_context, after_context = steps_before_after(working_steps, selected_chunk, context_steps)
        step_raw = llm.generate(
            ccv_step_context_prompt(
                case=case,
                constraints=constraints,
                chunk=selected_chunk,
                before_context=before_context,
                after_context=after_context,
            )
        )
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        if not step_inside_chunk(step_result.get("step"), selected_chunk):
            step_result["ignored_out_of_focal_step"] = step_result.get("step")
            step_result["step"] = None
        step_results.append(step_result)

        step = step_result.get("step") or selected.get("earliest_suspected_step")
        agent = step_result.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "step_confidence": as_float(step_result.get("confidence") or selected.get("confidence")),
                "score": ccv_beam_candidate_score(selected, step_result),
                "violated_constraint": step_result.get("violated_constraint"),
                "violation_type": step_result.get("violation_type"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "step_confidence": None, "reason": "No candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_adaptive_beam_context",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=as_float(best.get("confidence") or best.get("step_confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "CCV beam with adaptive chunk granularity and read-only before/after context",
        },
    )


def run_ccv_beam_simple10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    threshold = float(method_config.get("threshold", 0.65))
    beam_k = int(method_config.get("beam_k", 3))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_scalar_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = ccv_scalar_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    selected_chunks = select_ccv_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    candidates: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []

    for selected in selected_chunks:
        selected_chunk = chunks[int(selected["chunk_id"]) - 1]
        step_raw = llm.generate(ccv_step_bool_prompt(case=case, constraints=constraints, chunk=selected_chunk))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_score"] = selected.get("score")
        step_results.append(step_result)

        step = step_result.get("step") or selected.get("earliest_suspected_step")
        agent = step_result.get("agent") or selected.get("agent")
        if step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": normalize_optional_str(agent),
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "chunk_score": selected.get("score"),
                "score": selected.get("score"),
                "violated_constraint": step_result.get("violated_constraint"),
                "violation_type": step_result.get("violation_type"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_bool_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate_simple(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        best = {"step": None, "agent": None, "reason": "No candidate found."}
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_beam_simple10",
        agent=normalize_optional_str(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=None,
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "step_results": strip_raw_large(step_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "top-k chunks by a single LLM root_cause_score; rerank chooses one candidate without numeric confidence",
        },
    )


def run_ccv_beam_ordinal10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    beam_k = int(method_config.get("beam_k", 3))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_ccv_ordinal_stage(case, constraints, chunks, summaries, llm)
    selected_chunks = select_ordinal_beam_chunks(chunk_results, beam_k=beam_k)

    candidates: list[dict[str, Any]] = []
    for selected in selected_chunks:
        agent = normalize_ordinal_agent(selected.get("agent"))
        step = selected.get("step")
        if not agent or step is None:
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": agent,
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "blame_score": ordinal_score(selected),
                "chunk_score": ordinal_score(selected),
                "score": ordinal_score(selected),
                "reason": normalize_optional_str(selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_bool_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate_simple(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        fallback = select_ordinal_candidate(selected_chunks)
        best = {
            "step": fallback.get("step"),
            "agent": normalize_ordinal_agent(fallback.get("agent")),
            "reason": "No beam candidate with agent/step; using selected ordinal fallback.",
            "chunk_id": fallback.get("chunk_id"),
        }
        rerank_result = {"skipped": True, "reason": "No candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_beam_ordinal10",
        agent=normalize_ordinal_agent(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=None,
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "top-k chunks by ordinal blame_score; chunk agent/step become candidates; rerank chooses candidate without confidence",
        },
    )


def run_ccv_beam_ordinal_reread10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    beam_k = int(method_config.get("beam_k", 3))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    chunk_results = run_ccv_ordinal_stage(case, constraints, chunks, summaries, llm)
    selected_chunks = select_ordinal_beam_chunks(chunk_results, beam_k=beam_k)

    candidates: list[dict[str, Any]] = []
    reread_results: list[dict[str, Any]] = []
    for selected in selected_chunks:
        chunk_id = selected.get("chunk_id")
        if chunk_id is None:
            continue
        selected_chunk = chunks[int(chunk_id) - 1]
        step_raw = llm.generate(ccv_step_bool_prompt(case=case, constraints=constraints, chunk=selected_chunk))
        step_result = safe_json(step_raw)
        step_result["raw_response"] = step_raw
        step_result["source_chunk_id"] = selected.get("chunk_id")
        step_result["source_chunk_start"] = selected.get("chunk_start")
        step_result["source_chunk_end"] = selected.get("chunk_end")
        step_result["source_chunk_blame_score"] = ordinal_score(selected)
        step_result["first_pass_candidate"] = {
            "agent": selected.get("agent"),
            "step": selected.get("step"),
            "blame_score": ordinal_score(selected),
        }
        reread_results.append(step_result)

        step = step_result.get("step") or selected.get("step")
        agent = step_result.get("agent") or selected.get("agent")
        agent = normalize_ordinal_agent(agent)
        if not agent or step is None or str(step) == "":
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "step": step_int,
                "agent": agent,
                "chunk_id": selected.get("chunk_id"),
                "chunk_start": selected.get("chunk_start"),
                "chunk_end": selected.get("chunk_end"),
                "blame_score": ordinal_score(selected),
                "chunk_score": ordinal_score(selected),
                "score": ordinal_score(selected),
                "violated_constraint": step_result.get("violated_constraint"),
                "violation_type": step_result.get("violation_type"),
                "reason": normalize_optional_str(step_result.get("reason") or selected.get("reason")),
                "context": context_around(working_steps, step_int),
            }
        )

    candidates = dedupe_candidates(candidates)
    if len(candidates) > 1:
        judge_candidates = [
            {
                **candidate,
                "context": render_steps(candidate.get("context") or []),
            }
            for candidate in candidates
        ]
        rerank_raw = llm.generate(ccv_beam_rerank_bool_prompt(case=case, constraints=constraints, candidates=judge_candidates))
        rerank = safe_json(rerank_raw)
        best = select_ccv_beam_best_candidate_simple(candidates, rerank)
        rerank_result = {**rerank, "raw_response": rerank_raw}
    elif candidates:
        best = candidates[0]
        rerank_result = {"skipped": True, "reason": "Only one candidate."}
    else:
        fallback = select_ordinal_candidate(selected_chunks)
        best = {
            "step": fallback.get("step"),
            "agent": normalize_ordinal_agent(fallback.get("agent")),
            "reason": "No localized beam candidate; using selected ordinal fallback.",
            "chunk_id": fallback.get("chunk_id"),
        }
        rerank_result = {"skipped": True, "reason": "No localized candidate found."}

    return Prediction(
        case_id=case.case_id,
        method="ccv_beam_ordinal_reread10",
        agent=normalize_ordinal_agent(best.get("agent")),
        step=int(best["step"]) if best.get("step") is not None else None,
        confidence=None,
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "reread_results": strip_raw_large(reread_results),
            "candidates": strip_candidate_context(candidates),
            "rerank_result": rerank_result,
            "selection_rule": "top-k chunks by ordinal blame_score; reread/localize each selected chunk; rerank localized candidates without confidence",
        },
    )


def run_agentrx_baseline(case: Case, llm: LLM) -> Prediction:
    raw = llm.generate(agentrx_official_judge_prompt(case))
    parsed = safe_json(raw)
    step = parsed_step(parsed) or parse_int_maybe(parsed.get("index"))
    agent = parsed_agent(parsed) or normalize_optional_str(parsed.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="agentrx_baseline",
        agent=agent,
        step=step,
        confidence=as_float(parsed.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(parsed.get("reason_for_failure") or parsed.get("reason_for_index") or parsed.get("reason")),
        trace={
            "paper_basis": "AGENTRX Appendix G.1 baseline judge prompt; agent inferred from selected step if omitted.",
            "judge_result": {**parsed, "raw_response": raw},
        },
    )


def run_agentrx_original(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    semantic_window_steps = int(method_config.get("semantic_window_steps", 4))
    max_judge_violations = int(method_config.get("max_judge_violations", 0))

    global_raw = llm.generate(agentrx_original_global_constraints_prompt(case))
    global_obj = safe_json(global_raw)
    global_constraints = extract_constraints(global_obj)
    if not isinstance(global_constraints, list):
        global_constraints = []

    validation_log: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    constraint_counts: list[dict[str, Any]] = []

    for idx, step_obj in enumerate(case.steps):
        prefix = case.steps[: idx + 1]
        dynamic_raw = llm.generate(
            agentrx_original_dynamic_constraints_prompt(
                case=case,
                global_constraints=global_constraints,
                prefix=prefix,
                current_step=step_obj,
            )
        )
        dynamic_obj = safe_json(dynamic_raw)
        dynamic_constraints = extract_constraints(dynamic_obj)
        if not isinstance(dynamic_constraints, list):
            dynamic_constraints = []

        constraints = merge_agentrx_constraints(global_constraints, dynamic_constraints)
        window = case.steps[max(0, idx - semantic_window_steps) : min(len(case.steps), idx + semantic_window_steps + 1)]
        validation_raw = llm.generate(
            agentrx_original_step_validation_prompt(
                case=case,
                constraints=constraints,
                prefix=prefix,
                current_step=step_obj,
                window=window,
            )
        )
        validation_obj = safe_json(validation_raw)
        violations = extract_agentrx_original_violations(validation_obj, step_obj)
        validation_log.extend(violations)
        constraint_counts.append(
            {
                "step": step_obj.step,
                "global_constraint_count": len(global_constraints),
                "dynamic_constraint_count": len(dynamic_constraints),
                "merged_constraint_count": len(constraints),
                "violation_count": len(violations),
            }
        )
        step_records.append(
            {
                "step": step_obj.step,
                "agent": step_obj.agent,
                "dynamic_constraints": dynamic_constraints,
                "dynamic_raw_response": dynamic_raw,
                "validation": validation_obj,
                "validation_raw_response": validation_raw,
                "violations": violations,
            }
        )

    judge_log = select_agentrx_judge_violations(validation_log, max_judge_violations)
    judge_raw = llm.generate(agentrx_original_judge_prompt(case=case, validation_log=judge_log))
    judge = safe_json(judge_raw)
    step = parsed_step(judge) or parse_int_maybe(judge.get("index"))
    agent = parsed_agent(judge) or normalize_optional_str(judge.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)

    return Prediction(
        case_id=case.case_id,
        method="agentrx_original",
        agent=agent,
        step=step,
        confidence=as_float(judge.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(judge.get("reason_for_failure") or judge.get("reason_for_index") or judge.get("reason")),
        trace={
            "paper_basis": (
                "AGENTRX reproduction: global constraints, per-step dynamic constraints, guarded semantic "
                "constraint evaluation, validation log, and Appendix G.3-style final judge."
            ),
            "implementation_note": (
                "Who&When does not provide AGENTRX tool schemas or domain policies, so global constraints are "
                "inferred from the task and observed agent/tool inventory; executable checks fall back to "
                "semantic checks when deterministic tool schemas are unavailable."
            ),
            "semantic_window_steps": semantic_window_steps,
            "max_judge_violations": max_judge_violations,
            "global_constraints": global_constraints,
            "global_raw_response": global_raw,
            "constraint_counts": constraint_counts,
            "validation_log": validation_log,
            "judge_validation_log": judge_log,
            "step_records": strip_agentrx_step_records(step_records),
            "judge_result": {**judge, "raw_response": judge_raw},
        },
    )


def run_agentrx_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.50))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))

    if len(chunks) == 1:
        base = run_agentrx_original(case, llm, method_config)
        base.method = "agentrx_wrapper"
        base.trace = {
            **(base.trace or {}),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to AGENTRX original full-trace pipeline.",
        }
        return base

    validation_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_pred = run_agentrx_original(case_with_steps(case, chunk), llm, method_config)
        validation_log = (chunk_pred.trace or {}).get("validation_log")
        violation_count = len(validation_log) if isinstance(validation_log, list) else 0
        validation_results.append(
            {
                "chunk_id": idx + 1,
                "chunk_start": chunk[0].step,
                "chunk_end": chunk[-1].step,
                "agent": chunk_pred.agent,
                "step": chunk_pred.step,
                "confidence": chunk_pred.confidence,
                "score": agentrx_original_prediction_score(chunk_pred),
                "violation_count": violation_count,
                "reason": chunk_pred.reason,
                "prediction": compact_prediction_for_trace(chunk_pred),
            }
        )

    selected_chunks = select_agentrx_beam_chunks(validation_results, beam_k=beam_k, threshold=threshold)
    selected_ids = {int(item["chunk_id"]) for item in selected_chunks}
    selected_steps: list[LogStep] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 in selected_ids:
            selected_steps.extend(chunk)

    reread = run_agentrx_original(case_with_steps(case, selected_steps), llm, method_config)
    step = reread.step
    selected_step_ok = step is not None and any(step_inside_chunk(step, chunks[int(item["chunk_id"]) - 1]) for item in selected_chunks)
    if selected_step_ok:
        agent = reread.agent or agent_at_step(working_steps, step)
        confidence = reread.confidence
        reason = reread.reason
        selection_rule = (
            "AGENTRX original was applied to each adaptive chunk for ranking; top-k selected chunks were reread "
            "together by the same AGENTRX original pipeline."
        )
    else:
        fallback = select_echo_fallback_chunk(selected_chunks or validation_results)
        step = parse_int_maybe(fallback.get("step"))
        agent = normalize_optional_str(fallback.get("agent")) or (agent_at_step(working_steps, step) if step is not None else None)
        confidence = as_float(fallback.get("confidence") or fallback.get("score"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(fallback.get("reason"))
        selection_rule = "Selected-chunk AGENTRX reread did not return a selected step; using best selected chunk fallback."

    return Prediction(
        case_id=case.case_id,
        method="agentrx_wrapper",
        agent=normalize_optional_str(agent),
        step=step,
        confidence=confidence,
        reason=reason,
        trace={
            "paper_basis": (
                "AGENTRX original with context-allocation wrapper: adaptive chunk ranking, top-k beam, "
                "selected-chunk reread, and unchanged AGENTRX original pipeline on selected evidence."
            ),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": validation_results,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "selected_reread": compact_prediction_for_trace(reread),
            "selection_rule": selection_rule,
        },
    )


def run_echo_official(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    focuses = resolve_echo_focuses(method_config)
    conversation = echo_conversation_summary(case.steps)
    agent_analyses = run_echo_analyst_panel(case, llm, conversation, focuses, phase="agent")
    target_agents = select_echo_target_agents(agent_analyses, agent_k=int(method_config.get("agent_k", 2)))
    step_analyses = run_echo_analyst_panel(case, llm, conversation, focuses, phase="step", target_agents=target_agents)
    best = select_echo_consensus(step_analyses or agent_analyses)
    step = parse_int_maybe(best.get("step"))
    agent = normalize_optional_str(best.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="echo_official",
        agent=agent,
        step=step,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "paper_basis": "ECHO I4 decoupled objective-analysis panel with confidence-weighted consensus.",
            "focuses": focuses,
            "target_agents": target_agents,
            "agent_analyses": strip_raw_large(agent_analyses),
            "step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
        },
    )


def run_echo_i3_original(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    focuses = resolve_echo_focuses(method_config, case, phase="full")
    temperatures = resolve_echo_temperatures(method_config, len(focuses))
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    step_index_map = echo_step_index_map(case.steps)
    conversation = echo_original_conversation_summary(case.steps)
    analyses = run_echo_original_analyst_panel(
        case,
        llm,
        conversation,
        focuses,
        phase="full",
        temperatures=temperatures,
    )
    best = select_echo_consensus(
        analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, step_index_map)
    step = parse_int_maybe(best.get("step"))
    agent = normalize_optional_str(best.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="echo_i3_original",
        agent=agent,
        step=step,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "paper_basis": "ECHO I3-style objective analysis: Appendix A.5 ObjectiveAnalysisAgent prompt with confidence-weighted consensus.",
            "focuses": focuses,
            "temperatures": temperatures,
            "confidence_threshold": confidence_threshold,
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "step_index_map": step_index_map,
            "analyses": strip_raw_large(analyses),
            "consensus": best,
        },
    )


def run_echo_original(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    step_index_map = echo_step_index_map(case.steps)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    agent_conversation = echo_original_conversation_summary(case.steps)
    agent_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = echo_original_conversation_summary(case.steps, target_agents=target_agents or None)
    step_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus

    step = parse_int_maybe(best.get("step"))
    agent = normalize_optional_str(best.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="echo_original",
        agent=agent,
        step=step,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "paper_basis": (
                "ECHO I4 full reproduction: hierarchical context representation, 3 randomly sampled objective "
                "analysis agents from the 6-specialist pool, confidence threshold delta=0.30, confidence-weighted "
                "consensus, and decoupled agent-level then step-level attribution."
            ),
            "implementation_note": (
                "Hierarchical context is reconstructed locally from the paper appendix using deterministic "
                "L1/L2/L3 context extraction because no public ECHO code repository was available."
            ),
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "target_agent_rule": "Use ECHO agent-level consensus attribution list; fallback to analyst-evaluation ranking only if consensus is empty.",
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "step_index_map": step_index_map,
            "agent_analyses": strip_raw_large(agent_analyses),
            "agent_consensus": agent_consensus,
            "step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
        },
    )


def run_echo_appendix_original(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    step_index_map = echo_step_index_map(case.steps)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    agent_conversation = echo_appendix_conversation_summary(case.steps)
    agent_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = echo_appendix_conversation_summary(case.steps, target_agents=target_agents or None)
    step_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus

    step = parse_int_maybe(best.get("step"))
    agent = normalize_optional_str(best.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="echo_appendix_original",
        agent=agent,
        step=step,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "paper_basis": (
                "Appendix-based ECHO I4 reproduction: Appendix A.1 decoupled agent/step analysis, "
                "Appendix A.3-style hierarchical context extraction, Appendix A.5 ObjectiveAnalysisAgent "
                "schema, and Appendix A.7 confidence-weighted consensus."
            ),
            "implementation_note": (
                "The method preserves the appendix output schema and consensus logic while adapting the "
                "paper's conversation-history dictionaries to the Who&When local LogStep schema."
            ),
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "target_agent_rule": "Use ECHO agent-level consensus attribution list; fallback to analyst-evaluation ranking only if consensus is empty.",
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "step_index_map": step_index_map,
            "agent_analyses": strip_raw_large(agent_analyses),
            "agent_consensus": agent_consensus,
            "step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
        },
    )


def run_echo_appendix_strict_original(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    step_index_map = echo_step_index_map(case.steps)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    agent_conversation = echo_appendix_strict_conversation_summary(case.steps)
    agent_analyses = run_echo_appendix_strict_analyst_panel(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = echo_appendix_strict_conversation_summary(case.steps)
    step_analyses = run_echo_appendix_strict_analyst_panel(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus

    step = parse_int_maybe(best.get("step"))
    agent = normalize_optional_str(best.get("agent"))
    if not agent and step is not None:
        agent = agent_at_step(case.steps, step)
    return Prediction(
        case_id=case.case_id,
        method="echo_appendix_strict_original",
        agent=agent,
        step=step,
        confidence=as_float(best.get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(best.get("reason")),
        trace={
            "paper_basis": (
                "Strict Appendix ECHO I4 reproduction from echo.txt: A.1 decoupled agent/step calls, "
                "A.3 hierarchical context extraction, A.5 ObjectiveAnalysisAgent prompt without extra phase notes, "
                "and A.7 confidence-weighted consensus."
            ),
            "implementation_note": (
                "Prompt wording intentionally omits non-appendix helper text such as 'ECHO I4 phase'. "
                "Conversation turns are adapted from Who&When LogStep records into the appendix conversation-history schema."
            ),
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "step_indexing": "Appendix-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "step_index_map": step_index_map,
            "agent_analyses": strip_raw_large(agent_analyses),
            "agent_consensus": agent_consensus,
            "step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
        },
    )


def run_echo_original_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.30))
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))

    if len(chunks) == 1:
        base = run_echo_original(case, llm, method_config)
        base.method = "echo_original_wrapper"
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to ECHO original full-trace prompt.",
        }
        return base

    chunk_focuses = resolve_echo_chunk_focuses(method_config)
    chunk_temperatures = resolve_echo_temperatures(method_config, len(chunk_focuses))
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_index_map = echo_step_index_map(chunk)
        conversation = echo_original_conversation_summary(chunk)
        analyses = run_echo_original_analyst_panel(
            case,
            llm,
            conversation,
            chunk_focuses,
            phase="chunk",
            chunk_id=idx + 1,
            temperatures=chunk_temperatures,
        )
        best_chunk = select_echo_consensus(
            analyses,
            confidence_threshold=confidence_threshold,
            valid_steps=set(chunk_index_map),
        )
        best_chunk = map_echo_consensus_to_dataset_step(best_chunk, chunk_index_map)
        step = parse_int_maybe(best_chunk.get("step"))
        if step is not None and not step_inside_chunk(step, chunk):
            best_chunk["ignored_out_of_focal_step"] = step
            best_chunk["step"] = None
            best_chunk["agent"] = None
            best_chunk["confidence"] = 0.0
        chunk_results.append(
            {
                "chunk_id": idx + 1,
                "chunk_start": chunk[0].step,
                "chunk_end": chunk[-1].step,
                "agent": normalize_optional_str(best_chunk.get("agent")),
                "step": parse_int_maybe(best_chunk.get("step")),
                "confidence": as_float(best_chunk.get("confidence"), default=0.0),
                "score": echo_original_chunk_score(best_chunk),
                "reason": normalize_optional_str(best_chunk.get("reason")),
                "step_index": parse_int_maybe(best_chunk.get("echo_step_index")),
                "step_index_map": chunk_index_map,
                "analyses": strip_raw_large(analyses),
            }
        )

    selected_chunks = select_echo_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    selected_ids = {int(item["chunk_id"]) for item in selected_chunks}
    selected_steps: list[LogStep] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 in selected_ids:
            selected_steps.extend(chunk)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    selected_step_index_map = echo_step_index_map(selected_steps)
    agent_conversation = echo_original_conversation_summary(selected_steps)
    agent_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, selected_step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = echo_original_conversation_summary(selected_steps, target_agents=target_agents or None)
    step_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, selected_step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus
    step = parse_int_maybe(best.get("step"))
    selected_step_ok = step is not None and any(step_inside_chunk(step, chunks[int(item["chunk_id"]) - 1]) for item in selected_chunks)
    if selected_step_ok:
        agent = normalize_optional_str(best.get("agent")) or agent_at_step(working_steps, step)
        confidence = as_float(best.get("confidence"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(best.get("reason"))
        selection_rule = (
            "ECHO I4 original ObjectiveAnalysisAgent prompt was applied to adaptive chunks; top-k selected chunks "
            "were reread together with decoupled agent-level then step-level attribution."
        )
    else:
        fallback = select_echo_fallback_chunk(selected_chunks or chunk_results)
        step = parse_int_maybe(fallback.get("step"))
        agent = normalize_optional_str(fallback.get("agent")) or (agent_at_step(working_steps, step) if step is not None else None)
        confidence = as_float(fallback.get("confidence") or fallback.get("score"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(fallback.get("reason"))
        selection_rule = "Selected-chunk reread did not return a valid selected step; using best selected chunk fallback."

    return Prediction(
        case_id=case.case_id,
        method="echo_original_wrapper",
        agent=normalize_optional_str(agent),
        step=step,
        confidence=confidence,
        reason=reason,
        trace={
            "paper_basis": (
                "ECHO I4 full reproduction with context-allocation wrapper: adaptive chunk ranking, top-k beam, "
                "selected-chunk reread, then decoupled agent-level and step-level objective analysis."
            ),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_focuses": chunk_focuses,
            "chunk_temperatures": chunk_temperatures,
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "target_agent_rule": "Use ECHO agent-level consensus attribution list; fallback to analyst-evaluation ranking only if consensus is empty.",
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "selected_step_index_map": selected_step_index_map,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "selected_reread_agent_analyses": strip_raw_large(agent_analyses),
            "selected_reread_agent_consensus": agent_consensus,
            "selected_reread_step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
            "selection_rule": selection_rule,
        },
    )


def run_echo_original_global_router_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))

    if len(chunks) == 1:
        base = run_echo_original(case, llm, method_config)
        base.method = "echo_original_global_router_wrapper"
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to ECHO original full-trace prompt.",
        }
        return base

    router_conversation = echo_original_conversation_summary(working_steps)
    router_raw = llm.generate(
        echo_original_global_chunk_router_prompt(
            case=case,
            conversation=router_conversation,
            chunk_ranges=chunking.get("chunk_ranges", []),
            beam_k=beam_k,
        )
    )
    router_result = safe_json(router_raw)
    selected_chunks, router_candidates = select_echo_global_router_chunks(
        router_result=router_result,
        chunks=chunks,
        chunking=chunking,
        beam_k=beam_k,
    )
    selected_ids = {int(item["chunk_id"]) for item in selected_chunks}
    selected_steps: list[LogStep] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 in selected_ids:
            selected_steps.extend(chunk)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    selected_step_index_map = echo_step_index_map(selected_steps)
    agent_conversation = echo_original_conversation_summary(selected_steps)
    agent_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, selected_step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = echo_original_conversation_summary(selected_steps, target_agents=target_agents or None)
    step_analyses = run_echo_original_analyst_panel(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, selected_step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus

    step = parse_int_maybe(best.get("step"))
    selected_step_ok = step is not None and any(
        step_inside_chunk(step, chunks[int(item["chunk_id"]) - 1]) for item in selected_chunks
    )
    if selected_step_ok:
        agent = normalize_optional_str(best.get("agent")) or agent_at_step(working_steps, step)
        confidence = as_float(best.get("confidence"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(best.get("reason"))
        selection_rule = (
            "Global chunk router read the full trace and selected top-k chunk IDs; selected chunks were reread "
            "together with the unchanged ECHO decoupled agent-level then step-level attribution pipeline."
        )
    else:
        fallback = select_echo_fallback_chunk(selected_chunks)
        step = parse_int_maybe(fallback.get("step"))
        agent = normalize_optional_str(fallback.get("agent")) or (
            agent_at_step(working_steps, step) if step is not None else None
        )
        confidence = as_float(fallback.get("confidence") or fallback.get("score"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(fallback.get("reason"))
        selection_rule = "Selected-chunk reread did not return a valid selected step; using global-router fallback."

    return Prediction(
        case_id=case.case_id,
        method="echo_original_global_router_wrapper",
        agent=normalize_optional_str(agent),
        step=step,
        confidence=confidence,
        reason=reason,
        trace={
            "paper_basis": (
                "ECHO I4 selected-reread wrapper with only the chunk-selection stage changed: instead of scoring "
                "each chunk independently, a global router reads the full trace and chooses the top-k chunk IDs."
            ),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "router_result": router_result,
            "router_raw_response": router_raw,
            "router_candidates": router_candidates,
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "target_agent_rule": "Use ECHO agent-level consensus attribution list; fallback to analyst-evaluation ranking only if consensus is empty.",
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "selected_step_index_map": selected_step_index_map,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "selected_chunks": selected_chunks,
            "selected_reread_agent_analyses": strip_raw_large(agent_analyses),
            "selected_reread_agent_consensus": agent_consensus,
            "selected_reread_step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
            "selection_rule": selection_rule,
        },
    )


def run_echo_original_panel_router_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    confidence_threshold = float(method_config.get("confidence_threshold", 0.30))
    appendix_context = as_bool(method_config.get("appendix_context"), default=False)
    strict_appendix = as_bool(method_config.get("strict_appendix"), default=False)
    if strict_appendix:
        summary_fn = echo_appendix_strict_conversation_summary
        base_runner = run_echo_appendix_strict_original
        analyst_panel_fn = run_echo_appendix_strict_analyst_panel
    elif appendix_context:
        summary_fn = echo_appendix_conversation_summary
        base_runner = run_echo_appendix_original
        analyst_panel_fn = run_echo_original_analyst_panel
    else:
        summary_fn = echo_original_conversation_summary
        base_runner = run_echo_original
        analyst_panel_fn = run_echo_original_analyst_panel
    method_name = str(method_config.get("_method_name") or "echo_original_panel_router_wrapper")
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))

    if len(chunks) == 1:
        base = base_runner(case, llm, method_config)
        base.method = method_name
        base.trace = {
            **(base.trace or {}),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to the corresponding ECHO full-trace prompt.",
        }
        return base

    router_focuses = resolve_echo_focuses(method_config, case, phase="chunk_router")
    router_temperatures = resolve_echo_temperatures(method_config, len(router_focuses))
    router_conversation = summary_fn(working_steps)
    router_analyses: list[dict[str, Any]] = []
    for idx, focus in enumerate(router_focuses):
        prompt = echo_original_global_chunk_router_prompt(
            case=case,
            conversation=router_conversation,
            chunk_ranges=chunking.get("chunk_ranges", []),
            beam_k=beam_k,
            analyst_focus=focus,
        )
        temperature = router_temperatures[idx] if idx < len(router_temperatures) else None
        raw = generate_with_temperature(llm, prompt, temperature)
        parsed = safe_json(raw)
        parsed["analyst_focus"] = focus
        parsed["phase"] = "chunk_router"
        if temperature is not None:
            parsed["temperature"] = temperature
        parsed["raw_response"] = raw
        router_analyses.append(parsed)

    selected_chunks, router_vote_details = select_echo_panel_router_chunks(
        router_analyses=router_analyses,
        chunks=chunks,
        chunking=chunking,
        beam_k=beam_k,
    )
    selected_ids = {int(item["chunk_id"]) for item in selected_chunks}
    selected_steps: list[LogStep] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 in selected_ids:
            selected_steps.extend(chunk)

    agent_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_agent")
    agent_temperatures = resolve_echo_temperatures(method_config, len(agent_focuses))
    selected_step_index_map = echo_step_index_map(selected_steps)
    agent_conversation = summary_fn(selected_steps)
    agent_analyses = analyst_panel_fn(
        case,
        llm,
        agent_conversation,
        agent_focuses,
        phase="agent",
        temperatures=agent_temperatures,
    )
    agent_consensus = select_echo_consensus(
        agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    agent_consensus = map_echo_consensus_to_dataset_step(agent_consensus, selected_step_index_map)

    target_agents = echo_consensus_attribution_agents(agent_consensus)
    if not target_agents:
        target_agents = select_echo_target_agents(agent_analyses, agent_k=len(agent_analyses) or 1)

    step_focuses = resolve_echo_focuses(method_config, case, phase="selected_reread_step")
    step_temperatures = resolve_echo_temperatures(method_config, len(step_focuses))
    step_conversation = summary_fn(selected_steps, target_agents=target_agents or None)
    step_analyses = analyst_panel_fn(
        case,
        llm,
        step_conversation,
        step_focuses,
        phase="step",
        target_agents=target_agents,
        temperatures=step_temperatures,
    )
    best = select_echo_consensus(
        step_analyses or agent_analyses,
        confidence_threshold=confidence_threshold,
        valid_steps=set(selected_step_index_map),
    )
    best = map_echo_consensus_to_dataset_step(best, selected_step_index_map)
    if parse_int_maybe(best.get("step")) is None:
        best = agent_consensus

    step = parse_int_maybe(best.get("step"))
    selected_step_ok = step is not None and any(
        step_inside_chunk(step, chunks[int(item["chunk_id"]) - 1]) for item in selected_chunks
    )
    if selected_step_ok:
        agent = normalize_optional_str(best.get("agent")) or agent_at_step(working_steps, step)
        confidence = as_float(best.get("confidence"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(best.get("reason"))
        selection_rule = (
            "ECHO-style multi-analyst chunk router read the full trace and selected top-k chunk IDs by approval vote; "
            "selected chunks were reread with the unchanged ECHO objective-analysis consensus pipeline."
        )
    else:
        fallback = select_echo_fallback_chunk(selected_chunks)
        step = parse_int_maybe(fallback.get("step"))
        agent = normalize_optional_str(fallback.get("agent")) or (
            agent_at_step(working_steps, step) if step is not None else None
        )
        confidence = as_float(fallback.get("confidence") or fallback.get("score"), default=None)  # type: ignore[arg-type]
        reason = normalize_optional_str(fallback.get("reason"))
        selection_rule = "Selected-chunk reread did not return a valid selected step; using panel-router fallback."

    return Prediction(
        case_id=case.case_id,
        method=method_name,
        agent=normalize_optional_str(agent),
        step=step,
        confidence=confidence,
        reason=reason,
        trace={
            "paper_basis": (
                (
                    "Strict Appendix ECHO I4 context-allocation wrapper: chunk routing is added before the "
                    "A.5 ObjectiveAnalysisAgent prompt, and selected evidence is analyzed with the strict appendix "
                    "prompt and A.7 consensus."
                )
                if strict_appendix
                else (
                    "Appendix-based ECHO I4 context-allocation wrapper: Appendix A.3/A.5/A.7-style analysis is "
                    "preserved after the wrapper selects evidence chunks by panel approval vote."
                )
                if appendix_context
                else (
                    "ECHO I4-compatible context-allocation wrapper: the ECHO multi-analyst consensus idea is preserved "
                    "by making analysts approve chunk IDs instead of trusting single-call chunk confidence scores or fine-grained ranks."
                )
            ),
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "router_focuses": router_focuses,
            "router_temperatures": router_temperatures,
            "router_analyses": strip_raw_large(router_analyses),
            "router_vote_details": router_vote_details,
            "agent_focuses": agent_focuses,
            "agent_temperatures": agent_temperatures,
            "step_focuses": step_focuses,
            "step_temperatures": step_temperatures,
            "confidence_threshold": confidence_threshold,
            "target_agents": target_agents,
            "target_agent_rule": "Use ECHO agent-level consensus attribution list; fallback to analyst-evaluation ranking only if consensus is empty.",
            "step_indexing": "ECHO-style 0-based conversation index internally; mapped back to dataset step for evaluation.",
            "selected_step_index_map": selected_step_index_map,
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "selected_chunks": selected_chunks,
            "selected_reread_agent_analyses": strip_raw_large(agent_analyses),
            "selected_reread_agent_consensus": agent_consensus,
            "selected_reread_step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
            "selection_rule": selection_rule,
        },
    )


def run_echo_appendix_panel_router_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    appendix_config = dict(method_config)
    appendix_config["appendix_context"] = True
    appendix_config["_method_name"] = "echo_appendix_panel_router_wrapper"
    return run_echo_original_panel_router_wrapper(case, llm, appendix_config)


def run_echo_appendix_strict_panel_router_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    strict_config = dict(method_config)
    strict_config["appendix_context"] = True
    strict_config["strict_appendix"] = True
    strict_config["_method_name"] = "echo_appendix_strict_panel_router_wrapper"
    return run_echo_original_panel_router_wrapper(case, llm, strict_config)


def run_echo_wrapper(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    threshold = float(method_config.get("threshold", 0.30))
    working_steps, chunks, chunking = prepare_adaptive_chunks(case, method_config)
    beam_k = resolve_beam_k(method_config, len(chunks))
    record_beam_policy(chunking, method_config, beam_k, len(chunks))
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    if len(chunks) == 1:
        base = run_echo_official(case, llm, method_config)
        base.method = "echo_wrapper"
        base.trace = {
            **(base.trace or {}),
            "chunking": chunking,
            "selection_rule": "Single adaptive chunk; falling back to ECHO official full-trace panel.",
        }
        return base

    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            echo_chunk_ranking_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = echo_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)

    selected_chunks = select_echo_beam_chunks(chunk_results, beam_k=beam_k, threshold=threshold)
    selected_ids = {int(item["chunk_id"]) for item in selected_chunks}
    selected_steps: list[LogStep] = []
    for idx, chunk in enumerate(chunks):
        if idx + 1 in selected_ids:
            selected_steps.extend(chunk)
    conversation = echo_conversation_summary(selected_steps)
    focuses = resolve_echo_focuses(method_config)
    agent_analyses = run_echo_analyst_panel(case, llm, conversation, focuses, phase="agent")
    target_agents = select_echo_target_agents(agent_analyses, agent_k=int(method_config.get("agent_k", 2)))
    step_analyses = run_echo_analyst_panel(case, llm, conversation, focuses, phase="step", target_agents=target_agents)
    best = select_echo_consensus(step_analyses or agent_analyses)

    step = parse_int_maybe(best.get("step"))
    selected_step_ok = step is not None and any(step_inside_chunk(step, chunks[int(item["chunk_id"]) - 1]) for item in selected_chunks)
    if not selected_step_ok:
        fallback = select_echo_fallback_chunk(selected_chunks or chunk_results)
        step = parse_int_maybe(fallback.get("step"))
        agent = normalize_optional_str(fallback.get("agent")) or (agent_at_step(working_steps, step) if step is not None else None)
        reason = normalize_optional_str(fallback.get("reasoning") or fallback.get("reason"))
        confidence = echo_chunk_score(fallback)
        selection_rule = "ECHO selected-chunk panel did not return a valid selected step; using best selected chunk fallback."
    else:
        agent = normalize_optional_str(best.get("agent")) or agent_at_step(working_steps, step)
        reason = normalize_optional_str(best.get("reason"))
        confidence = as_float(best.get("confidence"), default=None)  # type: ignore[arg-type]
        selection_rule = "Adaptive chunk ranking selected top-k chunks, then ECHO I4 decoupled panel localized within selected chunks."

    return Prediction(
        case_id=case.case_id,
        method="echo_wrapper",
        agent=normalize_optional_str(agent),
        step=step,
        confidence=confidence,
        reason=reason,
        trace={
            "paper_basis": "ECHO objective-analysis/consensus with context-allocation wrapper.",
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "chunking": chunking,
            "chunk_results": strip_raw_large(chunk_results),
            "selected_chunk_ids": [item.get("chunk_id") for item in selected_chunks],
            "focuses": focuses,
            "target_agents": target_agents,
            "agent_analyses": strip_raw_large(agent_analyses),
            "step_analyses": strip_raw_large(step_analyses),
            "consensus": best,
            "selection_rule": selection_rule,
        },
    )


def run_agentrx10(case: Case, llm: LLM, method_config: dict[str, Any]) -> Prediction:
    chunk_count = int(method_config.get("chunks", 10))
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))
    max_violations = int(method_config.get("max_violations", 30))

    constraints_raw = llm.generate(agentrx_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    if not isinstance(constraints, list) or not constraints:
        constraints = default_constraints()

    working_steps = split_long_steps(case.steps, max_step_chars)
    chunks = make_budgeted_chunks(working_steps, chunk_count, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    validation_results: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            agentrx_validation_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["raw_response"] = raw
        validation_results.append(parsed)
        for violation in extract_violations(parsed):
            violation["chunk_id"] = idx + 1
            violation["chunk_start"] = chunk[0].step
            violation["chunk_end"] = chunk[-1].step
            violation["score"] = agentrx_violation_score(violation)
            violations.append(violation)

    compact_violations = select_validation_log_for_judge(violations, max_violations=max_violations)
    judge_raw = llm.generate(
        agentrx_judge_prompt(
            case=case,
            constraints=constraints,
            validation_log=compact_violations,
            chunk_summaries=summaries,
        )
    )
    judge = safe_json(judge_raw)
    fallback = select_agentrx_fallback(violations)

    step = parsed_step(judge)
    agent = parsed_agent(judge)
    if step is None and fallback:
        step = int(fallback["step"]) if fallback.get("step") is not None else None
    if not agent and fallback:
        agent = normalize_optional_str(fallback.get("agent"))

    return Prediction(
        case_id=case.case_id,
        method="agentrx10",
        agent=agent,
        step=step,
        confidence=as_float(judge.get("confidence") or (fallback or {}).get("confidence"), default=None),  # type: ignore[arg-type]
        reason=normalize_optional_str(judge.get("reason") or (fallback or {}).get("reason")),
        trace={
            "constraints": constraints,
            "constraints_raw_response": constraints_raw,
            "working_step_count": len(working_steps),
            "original_step_count": len(case.steps),
            "validation_results": strip_raw_large(validation_results),
            "validation_log": compact_violations,
            "judge_result": {**judge, "raw_response": judge_raw},
        },
    )


def prepare_adaptive_chunks(case: Case, method_config: dict[str, Any]) -> tuple[list[LogStep], list[list[LogStep]], dict[str, Any]]:
    max_step_chars = int(method_config.get("max_step_chars", 10000))
    max_chunk_chars = int(method_config.get("max_chunk_chars", 12000))
    target_chunk_tokens = int(method_config.get("target_chunk_tokens", 6000))
    target_chunk_steps = int(method_config.get("target_chunk_steps", 12))
    short_step_threshold = int(method_config.get("short_step_threshold", 20))
    short_token_threshold = int(method_config.get("short_token_threshold", target_chunk_tokens))
    max_chunks = int(method_config.get("max_chunks", 16))
    chunk_count_basis = str(method_config.get("chunk_count_basis", "tokens"))
    preserve_step_boundaries = bool(method_config.get("preserve_step_boundaries", True))
    enforce_max_chunks = bool(method_config.get("enforce_max_chunks", False))

    working_steps = case.steps if preserve_step_boundaries else split_long_steps(case.steps, max_step_chars)
    estimated_tokens = estimate_steps_tokens(working_steps)
    effective_max_chunks = max_chunks if enforce_max_chunks else len(working_steps)
    chunks = make_adaptive_budgeted_chunks(
        working_steps,
        target_chunk_tokens=target_chunk_tokens,
        target_chunk_steps=target_chunk_steps,
        short_step_threshold=short_step_threshold,
        short_token_threshold=short_token_threshold,
        max_chunks=effective_max_chunks,
        max_chunk_chars=max_chunk_chars,
        chunk_count_basis=chunk_count_basis,
    )
    normalized_basis = chunk_count_basis.strip().lower()
    if normalized_basis not in {"tokens", "steps"} and not preserve_step_boundaries and enforce_max_chunks:
        strategy = "adaptive_token_and_step_budget"
    else:
        strategy = "adaptive_token_budget_step_boundary"
    metadata = {
        "strategy": strategy,
        "chunk_count_basis": chunk_count_basis,
        "preserve_step_boundaries": preserve_step_boundaries,
        "enforce_max_chunks": enforce_max_chunks,
        "estimated_trace_tokens": estimated_tokens,
        "original_step_count": len(case.steps),
        "working_step_count": len(working_steps),
        "chunk_count": len(chunks),
        "short_step_threshold": short_step_threshold,
        "short_token_threshold": short_token_threshold,
        "target_chunk_tokens": target_chunk_tokens,
        "target_chunk_steps": target_chunk_steps,
        "max_chunks": max_chunks,
        "effective_max_chunks": effective_max_chunks,
        "max_step_chars": max_step_chars,
        "max_chunk_chars": max_chunk_chars,
        "chunk_ranges": [
            {
                "chunk_id": idx + 1,
                "start_step": chunk[0].step,
                "end_step": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
            }
            for idx, chunk in enumerate(chunks)
        ],
    }
    return working_steps, chunks, metadata


def mvbs_chunk_score(result: dict[str, Any]) -> float:
    return (
        0.35 * as_float(result.get("onset_score"))
        + 0.25 * as_float(result.get("causal_impact_score"))
        + 0.20 * as_float(result.get("answer_contrast_score"))
        + 0.10 * as_float(result.get("agent_specificity_score"))
        + 0.10 * as_float(result.get("confidence"))
        - 0.20 * as_float(result.get("symptom_penalty"))
    )


def ccv_chunk_score(result: dict[str, Any]) -> float:
    return (
        0.40 * as_float(result.get("severity"))
        + 0.30 * as_float(result.get("irreversibility"))
        + 0.20 * as_float(result.get("evidence_strength"))
        + 0.10 * as_float(result.get("confidence"))
        - 0.25 * as_float(result.get("downstream_symptom_penalty"))
    )


def ccv_scalar_chunk_score(result: dict[str, Any]) -> float:
    return as_float(result.get("root_cause_score") or result.get("score"))


def select_ccv_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int, threshold: float) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    beam_k = max(1, beam_k)
    ranked = sorted(chunk_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("contains_violation")) and float(result.get("score", 0.0)) >= threshold
    ]
    if viable:
        earliest = sorted(viable, key=lambda x: (int(x["chunk_start"]), -float(x.get("score", 0.0))))[0]
        selected.append(earliest)
        selected_ids.add(int(earliest["chunk_id"]))

    for item in ranked:
        chunk_id = int(item["chunk_id"])
        if chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(chunk_id)
        if len(selected) >= beam_k:
            break

    return sorted(selected, key=lambda x: int(x["chunk_start"]))


def select_ccv_beam_best_candidate(candidates: list[dict[str, Any]], rerank: dict[str, Any]) -> dict[str, Any]:
    candidate_id = rerank.get("candidate_id")
    if candidate_id is not None and str(candidate_id) != "":
        try:
            candidate_id_int = int(candidate_id)
            for candidate in candidates:
                if int(candidate.get("candidate_id", -1)) == candidate_id_int:
                    return {
                        **candidate,
                        "confidence": as_float(rerank.get("confidence") or candidate.get("step_confidence")),
                        "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                    }
        except (TypeError, ValueError):
            pass

    step = parsed_step(rerank)
    if step is not None:
        for candidate in candidates:
            if int(candidate.get("step", -1)) == step:
                return {
                    **candidate,
                    "confidence": as_float(rerank.get("confidence") or candidate.get("step_confidence")),
                    "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                }

    return sorted(
        candidates,
        key=lambda x: (int(x.get("step", 10**9)), -float(x.get("chunk_score", 0.0)), -float(x.get("step_confidence", 0.0))),
    )[0]


def select_ccv_beam_best_candidate_simple(candidates: list[dict[str, Any]], rerank: dict[str, Any]) -> dict[str, Any]:
    candidate_id = rerank.get("candidate_id")
    if candidate_id is not None and str(candidate_id) != "":
        try:
            candidate_id_int = int(candidate_id)
            for candidate in candidates:
                if int(candidate.get("candidate_id", -1)) == candidate_id_int:
                    return {
                        **candidate,
                        "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                    }
        except (TypeError, ValueError):
            pass

    step = parsed_step(rerank)
    if step is not None:
        for candidate in candidates:
            if int(candidate.get("step", -1)) == step:
                return {
                    **candidate,
                    "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                }

    return sorted(
        candidates,
        key=lambda x: (int(x.get("step", 10**9)), -float(x.get("chunk_score", 0.0))),
    )[0]


def ccv_beam_candidate_score(chunk_result: dict[str, Any], step_result: dict[str, Any]) -> float:
    return 0.65 * as_float(chunk_result.get("score")) + 0.35 * as_float(
        step_result.get("confidence") or chunk_result.get("confidence")
    )


def run_cgv_step_validation_stage(
    case: Case,
    constraints: list[dict],
    validation_steps: list[LogStep],
    evidence_steps: list[LogStep],
    evidence_label: str,
    llm: LLM,
    source_chunk_id: int | None = None,
    context_steps: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_records: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    for step_obj in validation_steps:
        current_evidence_steps = (
            context_around(evidence_steps, step_obj.step, radius=max(0, context_steps))
            if context_steps is not None
            else evidence_steps
        )
        current_evidence_label = (
            f"{evidence_label} (step radius = {max(0, context_steps)})"
            if context_steps is not None
            else evidence_label
        )
        raw = llm.generate(
            cgv_step_validation_prompt(
                case=case,
                constraints=constraints,
                current_step=step_obj,
                evidence_steps=current_evidence_steps,
                evidence_label=current_evidence_label,
            )
        )
        parsed = safe_json(raw)
        parsed["raw_response"] = raw
        parsed["current_step"] = step_obj.step
        parsed["current_agent"] = step_obj.agent
        if source_chunk_id is not None:
            parsed["source_chunk_id"] = source_chunk_id
        step_records.append(parsed)
        entry = cgv_validation_entry(parsed, step_obj, source_chunk_id=source_chunk_id)
        if entry is not None:
            validation_log.append(entry)
    return step_records, validation_log


def cgv_validation_entry(
    result: dict[str, Any],
    current_step: LogStep,
    source_chunk_id: int | None = None,
) -> dict[str, Any] | None:
    verdict = normalize_optional_str(result.get("verdict") or result.get("status") or result.get("judgment"))
    verdict_norm = verdict.strip().upper() if verdict else ""
    is_violation = verdict_norm == "VIOL" or as_bool(result.get("contains_violation"))
    candidate_step_match: dict[str, Any] | None = None
    if not is_violation:
        for item in result.get("candidate_steps") or []:
            if not isinstance(item, dict):
                continue
            if parse_int_maybe(item.get("step")) == current_step.step:
                candidate_step_match = item
                is_violation = True
                break
    if not is_violation:
        return None

    recoverability = normalize_optional_str(result.get("recoverability")) or "unclear"
    recoverable = recoverability.strip().lower() == "recovered"
    severity = as_float(result.get("severity") or result.get("onset_score"), default=0.0)
    causal_relevance = as_float(
        result.get("causal_relevance") or result.get("causal_score") or result.get("causal_impact_score"),
        default=0.0,
    )
    confidence = as_float(result.get("confidence"), default=0.0)
    score = 0.40 * severity + 0.35 * causal_relevance + 0.25 * confidence
    if recoverable:
        score -= 0.25
    step = parsed_step(result)
    if step is None or step != current_step.step:
        step = current_step.step
    agent = (
        parsed_agent(result)
        or normalize_optional_str(candidate_step_match.get("agent") if candidate_step_match else None)
        or current_step.agent
    )
    entry = {
        "step": step,
        "agent": agent,
        "verdict": "VIOL",
        "violated_constraints": normalize_constraint_names(result.get("violated_constraints")),
        "violated_constraint": normalize_optional_str(result.get("violated_constraint")),
        "violation_type": normalize_optional_str(result.get("violation_type")),
        "severity": severity,
        "causal_relevance": causal_relevance,
        "confidence": confidence,
        "recoverability": recoverability,
        "recoverable": recoverable,
        "score": max(0.0, score),
        "evidence": normalize_optional_str(result.get("evidence")),
        "reason": normalize_optional_str(
            result.get("reason") or (candidate_step_match.get("reason") if candidate_step_match else None)
        ),
    }
    if source_chunk_id is not None:
        entry["source_chunk_id"] = source_chunk_id
    return entry


def normalize_constraint_names(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = normalize_optional_str(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def select_cgv_fallback(validation_log: list[dict[str, Any]], evidence_steps: list[LogStep]) -> dict[str, Any] | None:
    valid_steps = {step.step for step in evidence_steps}
    pool = [item for item in validation_log if parse_int_maybe(item.get("step")) in valid_steps]
    if not pool:
        return None
    unrecovered = [item for item in pool if not as_bool(item.get("recoverable"))]
    candidates = unrecovered or pool
    return sorted(candidates, key=lambda x: (int(x.get("step", 10**9)), -float(x.get("score", 0.0))))[0]


def flatten_step_blocks(blocks: list[list[LogStep]]) -> list[LogStep]:
    out: list[LogStep] = []
    seen: set[int] = set()
    for block in blocks:
        for step in block:
            if int(step.step) in seen:
                continue
            out.append(step)
            seen.add(int(step.step))
    return sorted(out, key=lambda step: int(step.step))


def step_inside_any(value: Any, chunks: list[list[LogStep]]) -> bool:
    return any(step_inside_chunk(value, chunk) for chunk in chunks if chunk)


def a2p_result_score(result: dict[str, Any]) -> float | None:
    if not result:
        return None
    for key in ("causal_score", "Causal Score", "counterfactual_score", "score", "confidence"):
        if result.get(key) is not None:
            return as_float(result.get(key), default=0.0)  # type: ignore[arg-type]
    return None


def a2p_result_is_positive(result: dict[str, Any]) -> bool:
    return (
        as_bool(result.get("contains_counterfactual_error"))
        or as_bool(result.get("contains_decisive_error"))
        or as_bool(result.get("would_fix_failure"))
    )


def run_a2p_chunk_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            a2p_chunk_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["step"] = None
            parsed["contains_counterfactual_error"] = False
            parsed["would_fix_failure"] = False
        parsed["score"] = a2p_result_score(parsed) or 0.0
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_a2p_scaffold_chunk_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            a2p_scaffold_chunk_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["step"] = None
            parsed["contains_counterfactual_error"] = False
            parsed["would_fix_failure"] = False
        parsed["score"] = a2p_result_score(parsed) or 0.0
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_a2p_official_chunk_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            a2p_official_chunk_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = parse_a2p_official_response(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["step"] = None
            parsed["contains_counterfactual_error"] = False
            parsed["would_fix_failure"] = False
        parsed["score"] = a2p_result_score(parsed) or 0.0
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_a2p_official_local_confidence_chunk_stage(
    case: Case,
    chunks: list[list[LogStep]],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            a2p_official_local_confidence_chunk_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        step = parsed_step(parsed)
        if step is not None and not step_inside_chunk(step, chunk):
            parsed["ignored_out_of_focal_step"] = step
            parsed["step"] = None
            parsed["Step Number"] = None
            parsed["contains_root_cause"] = False
        confidence = as_float(
            parsed.get("confidence")
            or parsed.get("Confidence")
            or parsed.get("causal_score")
            or parsed.get("score"),
            default=0.0,
        )
        parsed["confidence"] = confidence
        parsed["score"] = confidence
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_a2p_chunk_context_stage(
    case: Case,
    working_steps: list[LogStep],
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
    context_steps: int,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        before_context, after_context = steps_before_after(working_steps, chunk, context_steps)
        raw = llm.generate(
            a2p_chunk_context_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                before_context=before_context,
                after_context=after_context,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["context_before_start"] = before_context[0].step if before_context else None
        parsed["context_before_end"] = before_context[-1].step if before_context else None
        parsed["context_after_start"] = after_context[0].step if after_context else None
        parsed["context_after_end"] = after_context[-1].step if after_context else None
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["step"] = None
            parsed["contains_counterfactual_error"] = False
            parsed["would_fix_failure"] = False
        parsed["score"] = a2p_result_score(parsed) or 0.0
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_chunk_vote_stage_for_selected_chunks(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    selected_chunks: list[dict[str, Any]],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for selected in selected_chunks:
        if selected.get("chunk_id") is None:
            continue
        idx = int(selected["chunk_id"]) - 1
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        raw = llm.generate(
            chunk_all_at_once_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["source_a2p_score"] = selected.get("score")
        parsed["source_a2p_step"] = selected.get("step")
        parsed["source_a2p_agent"] = selected.get("agent")
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["likely_contains_decisive_error"] = False
            parsed["agent"] = None
            parsed["step"] = None
        parsed["score"] = chunk_vote_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_chunk_vote_context_stage_for_selected_chunks(
    case: Case,
    working_steps: list[LogStep],
    chunks: list[list[LogStep]],
    summaries: list[str],
    selected_chunks: list[dict[str, Any]],
    llm: LLM,
    context_steps: int,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for selected in selected_chunks:
        if selected.get("chunk_id") is None:
            continue
        idx = int(selected["chunk_id"]) - 1
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        before_context, after_context = steps_before_after(working_steps, chunk, context_steps)
        raw = llm.generate(
            chunk_all_at_once_context_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                before_context=before_context,
                after_context=after_context,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["context_before_start"] = before_context[0].step if before_context else None
        parsed["context_before_end"] = before_context[-1].step if before_context else None
        parsed["context_after_start"] = after_context[0].step if after_context else None
        parsed["context_after_end"] = after_context[-1].step if after_context else None
        parsed["source_a2p_score"] = selected.get("score")
        parsed["source_a2p_step"] = selected.get("step")
        parsed["source_a2p_agent"] = selected.get("agent")
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["likely_contains_decisive_error"] = False
            parsed["agent"] = None
            parsed["step"] = None
        parsed["score"] = chunk_vote_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def select_top_target_agents_from_chunks(chunk_results: list[dict[str, Any]], agent_k: int) -> list[str]:
    weights: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for result in chunk_results:
        agent = normalize_optional_str(result.get("agent"))
        if not agent or result.get("step") is None:
            continue
        weight = chunk_vote_score(result)
        weights[agent] = weights.get(agent, 0.0) + weight
        counts[agent] += 1
        first_seen[agent] = min(first_seen.get(agent, 10**9), int(result.get("chunk_start", 10**9)))
    ranked = sorted(
        weights,
        key=lambda agent: (-weights[agent], first_seen.get(agent, 10**9), -counts[agent], agent),
    )
    return ranked[: max(1, agent_k)]


def select_top_agents_from_a2p_chunks(chunk_results: list[dict[str, Any]], agent_k: int) -> list[str]:
    weights: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for result in chunk_results:
        agent = normalize_optional_str(result.get("agent") or result.get("Agent Name"))
        if not agent or agent.upper() == "NONE" or result.get("step") is None:
            continue
        weight = a2p_result_score(result) or 0.0
        if weight <= 0 and not a2p_result_is_positive(result):
            continue
        weights[agent] = weights.get(agent, 0.0) + weight
        counts[agent] += 1
        first_seen[agent] = min(first_seen.get(agent, 10**9), int(result.get("chunk_start", 10**9)))
    ranked = sorted(
        weights,
        key=lambda agent: (-weights[agent], first_seen.get(agent, 10**9), -counts[agent], agent),
    )
    return ranked[: max(1, agent_k)]


def first_matching_agent(agent: str, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if agents_match(agent, candidate):
            return candidate
    return None


def select_agent_hybrid_fallback_candidate(
    hybrid_chunk_results: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
    target_agents: list[str],
) -> dict[str, Any]:
    target_rank = {agent: idx + 1 for idx, agent in enumerate(target_agents)}
    paper_candidates = []
    for result in hybrid_chunk_results:
        agent = normalize_optional_str(result.get("agent"))
        if not agent or result.get("step") is None:
            continue
        matched = first_matching_agent(agent, target_agents)
        if not matched:
            continue
        paper_candidates.append(
            {
                **result,
                "target_agent": matched,
                "target_agent_rank": target_rank.get(matched, 10**9),
                "confidence": as_float(result.get("confidence"), default=None),  # type: ignore[arg-type]
            }
        )
    if paper_candidates:
        return sorted(
            paper_candidates,
            key=lambda x: (
                int(x.get("target_agent_rank", 10**9)),
                int(x.get("chunk_start", 10**9)),
                ordinal_step_for_sort(x),
                -float(x.get("score", 0.0)),
            ),
        )[0]

    a2p_candidates = [item for item in selected_chunks if item.get("step") is not None]
    if a2p_candidates:
        return sorted(a2p_candidates, key=lambda x: (int(x.get("chunk_start", 10**9)), ordinal_step_for_sort(x)))[0]
    return {"agent": target_agents[0] if target_agents else None, "step": None, "confidence": None, "reason": "No fallback candidate."}


def select_a2p_chunk_candidate(chunk_results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not chunk_results:
        return {"agent": None, "step": None, "score": None, "chunk_id": None, "reason": "No A2P chunk results."}
    viable = [
        result
        for result in chunk_results
        if a2p_result_is_positive(result) and float(result.get("score", 0.0)) >= threshold and result.get("step") is not None
    ]
    if viable:
        return sorted(viable, key=lambda x: (int(x.get("chunk_start", 10**9)), -float(x.get("score", 0.0))))[0]
    candidates = [result for result in chunk_results if result.get("step") is not None]
    pool = candidates or chunk_results
    return sorted(
        pool,
        key=lambda x: (-float(x.get("score", 0.0)), int(x.get("chunk_start", 10**9)), ordinal_step_for_sort(x)),
    )[0]


def resolve_beam_k(method_config: dict[str, Any], chunk_count: int) -> int:
    chunk_count = max(1, int(chunk_count))
    if method_config.get("beam_fraction") is None:
        return min(chunk_count, max(1, int(method_config.get("beam_k", 3))))

    fraction = float(method_config.get("beam_fraction", 0.25))
    if fraction <= 0:
        raise ValueError("beam_fraction must be positive.")
    beam_k = math.ceil(chunk_count * fraction)
    beam_k_min = int(method_config.get("beam_k_min", 3))
    beam_k_max = int(method_config.get("beam_k_max", 8))
    if beam_k_max < beam_k_min:
        raise ValueError("beam_k_max must be greater than or equal to beam_k_min.")
    if beam_k_max <= 3:
        if chunk_count <= 1:
            return 1
        if chunk_count <= 2:
            return chunk_count
        if chunk_count == 3:
            return 2
        return min(chunk_count, beam_k_max)
    beam_k = min(beam_k_max, max(beam_k_min, beam_k))
    return min(chunk_count, max(1, beam_k))


def record_beam_policy(
    chunking: dict[str, Any],
    method_config: dict[str, Any],
    beam_k: int,
    chunk_count: int,
) -> None:
    chunking["beam_k_effective"] = beam_k
    if method_config.get("beam_fraction") is not None:
        if int(method_config.get("beam_k_max", 8)) <= 3:
            chunking["beam_policy"] = "caw_piecewise_cap3"
        else:
            chunking["beam_policy"] = "fraction_of_chunk_count"
        chunking["beam_fraction"] = float(method_config.get("beam_fraction", 0.25))
        chunking["beam_k_min"] = int(method_config.get("beam_k_min", 3))
        chunking["beam_k_max"] = int(method_config.get("beam_k_max", 8))
    else:
        chunking["beam_policy"] = "fixed"
        chunking["beam_k_configured"] = int(method_config.get("beam_k", 3))
    chunking["beam_chunk_count"] = chunk_count


def select_a2p_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int, threshold: float) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    beam_k = max(1, beam_k)
    ranked = sorted(chunk_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    viable = [
        result
        for result in chunk_results
        if a2p_result_is_positive(result) and float(result.get("score", 0.0)) >= threshold and result.get("step") is not None
    ]
    if viable:
        earliest = sorted(viable, key=lambda x: (int(x.get("chunk_start", 10**9)), -float(x.get("score", 0.0))))[0]
        selected.append(earliest)
        selected_ids.add(int(earliest["chunk_id"]))

    for item in ranked:
        chunk_id = int(item["chunk_id"])
        if chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(chunk_id)
        if len(selected) >= beam_k:
            break

    return sorted(selected, key=lambda x: int(x.get("chunk_start", 10**9)))


def select_a2p_local_confidence_chunks(
    chunk_results: list[dict[str, Any]],
    chunks: list[list[LogStep]],
    beam_k: int,
) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    beam_k = max(1, min(beam_k, len(chunks)))
    ranked = sorted(
        chunk_results,
        key=lambda x: (-as_float(x.get("confidence") or x.get("score"), default=0.0), int(x.get("chunk_start", 10**9))),
    )
    selected = ranked[:beam_k]
    return sorted(selected, key=lambda x: int(x.get("chunk_start", 10**9)))


def select_a2p_local_confidence_fallback_candidate(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunk_results:
        return {"agent": None, "step": None, "confidence": None, "reason": "No local-confidence chunk candidates."}
    pool = sorted(
        chunk_results,
        key=lambda x: (-as_float(x.get("confidence") or x.get("score"), default=0.0), int(x.get("chunk_start", 10**9))),
    )
    best = pool[0]
    return {
        "agent": parsed_agent(best),
        "step": parsed_step(best),
        "confidence": as_float(best.get("confidence") or best.get("score"), default=None),  # type: ignore[arg-type]
        "reason": normalize_optional_str(best.get("reason") or best.get("Reason for Mistake")),
    }


def select_a2p_best_candidate(candidates: list[dict[str, Any]], rerank: dict[str, Any]) -> dict[str, Any]:
    candidate_id = rerank.get("candidate_id")
    if candidate_id is not None and str(candidate_id) != "":
        try:
            candidate_id_int = int(candidate_id)
            for candidate in candidates:
                if int(candidate.get("candidate_id", -1)) == candidate_id_int:
                    return {
                        **candidate,
                        "causal_score": a2p_result_score(rerank) or candidate.get("causal_score"),
                        "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                    }
        except (TypeError, ValueError):
            pass

    step = parsed_step(rerank)
    if step is not None:
        for candidate in candidates:
            if int(candidate.get("step", -1)) == step:
                return {
                    **candidate,
                    "causal_score": a2p_result_score(rerank) or candidate.get("causal_score"),
                    "reason": normalize_optional_str(rerank.get("reason") or candidate.get("reason")),
                }

    return sorted(
        candidates,
        key=lambda x: (int(x.get("step", 10**9)), -float(x.get("causal_score", 0.0)), -float(x.get("chunk_score", 0.0))),
    )[0]


def strip_candidate_context(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for candidate in candidates:
        copy = dict(candidate)
        context = copy.get("context")
        if isinstance(context, list):
            copy["context"] = render_steps(context)
        stripped.append(copy)
    return stripped


def run_chunk_vote_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            chunk_all_at_once_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["score"] = chunk_vote_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_who_when_official_chunk_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            who_when_official_chunk_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = parse_a2p_official_response(raw)
        contains = as_bool(parsed.get("contains_counterfactual_error"), default=None)
        if contains is None:
            contains = parsed_agent(parsed) is not None and parsed_step(parsed) is not None
        parsed["likely_contains_decisive_error"] = contains
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["confidence"] = as_float(parsed.get("score") or parsed.get("causal_score"), default=0.0)
        parsed["score"] = as_float(parsed.get("score") or parsed.get("causal_score"), default=0.0)
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["likely_contains_decisive_error"] = False
            parsed["agent"] = None
            parsed["step"] = None
            parsed["score"] = 0.0
            parsed["confidence"] = 0.0
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_chunk_vote_context_stage(
    case: Case,
    working_steps: list[LogStep],
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
    context_steps: int,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        before_context, after_context = steps_before_after(working_steps, chunk, context_steps)
        raw = llm.generate(
            chunk_all_at_once_context_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                before_context=before_context,
                after_context=after_context,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["context_before_start"] = before_context[0].step if before_context else None
        parsed["context_before_end"] = before_context[-1].step if before_context else None
        parsed["context_after_start"] = after_context[0].step if after_context else None
        parsed["context_after_end"] = after_context[-1].step if after_context else None
        if parsed.get("step") is not None and not step_inside_chunk(parsed.get("step"), chunk):
            parsed["ignored_out_of_focal_step"] = parsed.get("step")
            parsed["likely_contains_decisive_error"] = False
            parsed["agent"] = None
            parsed["step"] = None
        parsed["score"] = chunk_vote_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_chunk_bool_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            chunk_bool_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_chunk_ordinal_stage(
    case: Case,
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            chunk_ordinal_prompt(
                case=case,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        normalize_ordinal_result(parsed)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_ccv_ordinal_stage(
    case: Case,
    constraints: list[dict],
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        raw = llm.generate(
            ccv_ordinal_chunk_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        normalize_ordinal_result(parsed)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def run_ccv_context_stage(
    case: Case,
    constraints: list[dict],
    working_steps: list[LogStep],
    chunks: list[list[LogStep]],
    summaries: list[str],
    llm: LLM,
    context_steps: int,
) -> list[dict[str, Any]]:
    chunk_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        before_context, after_context = steps_before_after(working_steps, chunk, context_steps)
        raw = llm.generate(
            ccv_chunk_context_prompt(
                case=case,
                constraints=constraints,
                chunk_id=idx + 1,
                chunk_count=len(chunks),
                chunk=chunk,
                before_context=before_context,
                after_context=after_context,
                prev_summary=summaries[idx - 1] if idx > 0 else "None",
                next_summary=summaries[idx + 1] if idx + 1 < len(chunks) else "None",
            )
        )
        parsed = safe_json(raw)
        parsed["model_chunk_id"] = parsed.get("chunk_id")
        parsed["chunk_id"] = idx + 1
        parsed["chunk_start"] = chunk[0].step
        parsed["chunk_end"] = chunk[-1].step
        parsed["context_before_start"] = before_context[0].step if before_context else None
        parsed["context_before_end"] = before_context[-1].step if before_context else None
        parsed["context_after_start"] = after_context[0].step if after_context else None
        parsed["context_after_end"] = after_context[-1].step if after_context else None
        suspected_step = parsed.get("earliest_suspected_step") or parsed.get("step")
        if suspected_step is not None and not step_inside_chunk(suspected_step, chunk):
            parsed["ignored_out_of_focal_step"] = suspected_step
            parsed["contains_violation"] = False
            parsed["earliest_suspected_step"] = None
            parsed["step"] = None
            parsed["agent"] = None
            parsed["score"] = 0.0
        else:
            parsed["score"] = ccv_chunk_score(parsed)
        parsed["raw_response"] = raw
        chunk_results.append(parsed)
    return chunk_results


def select_chunk_vote_candidate(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [result for result in chunk_results if result.get("step") is not None]
    if not candidates:
        return {"agent": None, "step": None, "confidence": None, "reason": "No chunk candidate found.", "chunk_id": None}
    positive = [result for result in candidates if as_bool(result.get("likely_contains_decisive_error"))]
    pool = positive or candidates
    return sorted(pool, key=lambda x: (x.get("score", 0.0), -int(x.get("chunk_start", 10**9))), reverse=True)[0]


def select_who_when_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int, threshold: float) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    beam_k = max(1, beam_k)
    ranked = sorted(chunk_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    viable = [
        result
        for result in chunk_results
        if as_bool(result.get("likely_contains_decisive_error"))
        and float(result.get("score", 0.0)) >= threshold
        and result.get("step") is not None
    ]
    if viable:
        earliest = sorted(viable, key=lambda x: (int(x.get("chunk_start", 10**9)), -float(x.get("score", 0.0))))[0]
        selected.append(earliest)
        selected_ids.add(int(earliest["chunk_id"]))

    for item in ranked:
        chunk_id = int(item["chunk_id"])
        if chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(chunk_id)
        if len(selected) >= beam_k:
            break

    return sorted(selected, key=lambda x: int(x.get("chunk_start", 10**9)))


def select_chunk_bool_candidate(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [result for result in chunk_results if result.get("step") is not None]
    if not candidates:
        return {"agent": None, "step": None, "confidence": None, "reason": "No chunk candidate found.", "chunk_id": None}
    positive = [result for result in candidates if as_bool(result.get("likely_contains_decisive_error"))]
    pool = positive or candidates
    return sorted(pool, key=lambda x: (int(x.get("chunk_start", 10**9)), int(x.get("step", 10**9))))[0]


def select_top_ordinal_chunks(chunk_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    max_score = max(ordinal_score(result) for result in chunk_results)
    top = [result for result in chunk_results if ordinal_score(result) == max_score]
    return sorted(top, key=lambda x: (ordinal_step_for_sort(x), int(x.get("chunk_start", 10**9))))


def select_ordinal_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    beam_k = max(1, beam_k)
    return sorted(
        chunk_results,
        key=lambda x: (-ordinal_score(x), ordinal_step_for_sort(x), int(x.get("chunk_start", 10**9))),
    )[:beam_k]


def select_ordinal_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"agent": None, "step": None, "blame_score": 0, "chunk_id": None}
    viable = [
        candidate
        for candidate in candidates
        if ordinal_score(candidate) >= 2 and candidate.get("step") is not None and normalize_optional_str(candidate.get("agent"))
    ]
    pool = viable or candidates
    return sorted(
        pool,
        key=lambda x: (-ordinal_score(x), ordinal_step_for_sort(x), int(x.get("chunk_start", 10**9))),
    )[0]


def normalize_ordinal_result(result: dict[str, Any]) -> None:
    result["blame_score"] = ordinal_score(result)
    if result["blame_score"] <= 1:
        result["agent"] = "NONE"
        result["step"] = None
    elif result.get("step") is not None:
        try:
            result["step"] = int(result["step"])
        except (TypeError, ValueError):
            result["step"] = None


def normalize_ordinal_agent(value: Any) -> str | None:
    agent = normalize_optional_str(value)
    if not agent or agent.upper() == "NONE":
        return None
    return agent


def ordinal_score(result: dict[str, Any]) -> int:
    value = result.get("blame_score")
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(4, score))


def ordinal_step_for_sort(result: dict[str, Any]) -> int:
    value = result.get("step")
    if value is None or str(value).strip() == "":
        return 10**9
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def select_target_agent_from_chunks(chunk_results: list[dict[str, Any]]) -> str | None:
    weights: dict[str, float] = {}
    for result in chunk_results:
        agent = normalize_optional_str(result.get("agent"))
        if not agent or result.get("step") is None:
            continue
        weight = chunk_vote_score(result)
        weights[agent] = weights.get(agent, 0.0) + weight
    if not weights:
        return None
    return max(weights.items(), key=lambda item: item[1])[0]


def select_target_agent_from_bool_chunks(chunk_results: list[dict[str, Any]]) -> str | None:
    candidates = [
        result
        for result in chunk_results
        if normalize_optional_str(result.get("agent")) and result.get("step") is not None
    ]
    if not candidates:
        return None

    positive = [result for result in candidates if as_bool(result.get("likely_contains_decisive_error"))]
    pool = positive or candidates
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}

    for result in pool:
        agent = normalize_optional_str(result.get("agent"))
        if not agent:
            continue
        counts[agent] += 1
        first_seen[agent] = min(first_seen.get(agent, 10**9), int(result.get("chunk_start", 10**9)))

    if not counts:
        return None
    return sorted(counts, key=lambda agent: (-counts[agent], first_seen.get(agent, 10**9), agent))[0]


def chunk_vote_score(result: dict[str, Any]) -> float:
    contains = 1.0 if as_bool(result.get("likely_contains_decisive_error")) else 0.0
    confidence = as_float(result.get("confidence"))
    has_step = 1.0 if result.get("step") is not None else 0.0
    return 0.60 * contains + 0.30 * confidence + 0.10 * has_step


def extract_violations(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("violations")
    if not isinstance(raw, list):
        return []
    violations: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("step") is None:
            continue
        copy = dict(item)
        try:
            copy["step"] = int(copy["step"])
        except (TypeError, ValueError):
            continue
        violations.append(copy)
    return violations


def merge_agentrx_constraints(global_constraints: list[dict[str, Any]], dynamic_constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, constraints in (("global", global_constraints), ("dynamic", dynamic_constraints)):
        for idx, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                continue
            copy = dict(constraint)
            copy.setdefault("source", source)
            constraint_id = normalize_optional_str(copy.get("id")) or f"{source}_{idx + 1}"
            copy["id"] = constraint_id
            key = f"{source}:{constraint_id}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy)
    return merged


def extract_agentrx_original_violations(result: dict[str, Any], current_step: LogStep) -> list[dict[str, Any]]:
    raw_violations = result.get("violations")
    violations: list[dict[str, Any]] = []
    if isinstance(raw_violations, list):
        for item in raw_violations:
            if isinstance(item, dict):
                normalized = normalize_agentrx_violation(item, current_step)
                if normalized:
                    violations.append(normalized)
    raw_checks = result.get("checks")
    if isinstance(raw_checks, list):
        for check in raw_checks:
            if not isinstance(check, dict):
                continue
            verdict = normalize_optional_str(check.get("verdict"))
            if verdict and verdict.strip().upper() == "VIOL":
                normalized = normalize_agentrx_violation(check, current_step)
                if normalized:
                    violations.append(normalized)
    deduped: dict[tuple[int, str, str], dict[str, Any]] = {}
    for violation in violations:
        key = (
            int(violation.get("step", current_step.step)),
            normalize_optional_str(violation.get("agent")) or current_step.agent,
            normalize_optional_str(violation.get("assertion_name") or violation.get("constraint_id")) or "",
        )
        deduped[key] = violation
    return list(deduped.values())


def normalize_agentrx_violation(item: dict[str, Any], current_step: LogStep) -> dict[str, Any] | None:
    step = parse_int_maybe(item.get("step") or item.get("index") or item.get("step_index")) or current_step.step
    agent = normalize_optional_str(item.get("agent")) or current_step.agent
    assertion_name = normalize_optional_str(item.get("assertion_name") or item.get("constraint_id") or item.get("violated_constraint"))
    reason = normalize_optional_str(item.get("reason") or item.get("reasoning"))
    evidence = normalize_optional_str(item.get("evidence")) or ""
    if not assertion_name and not reason and not evidence:
        return None
    return {
        "step": step,
        "agent": agent,
        "assertion_name": assertion_name or "unnamed_constraint",
        "constraint_type": normalize_optional_str(item.get("constraint_type")) or "ANY",
        "check_type": normalize_optional_str(item.get("check_type")) or "nl_check",
        "severity": as_float(item.get("severity"), default=0.5),
        "evidence": evidence,
        "taxonomy_targets": item.get("taxonomy_targets") if isinstance(item.get("taxonomy_targets"), list) else [],
        "reason": reason or evidence,
    }


def select_agentrx_judge_violations(violations: list[dict[str, Any]], max_violations: int) -> list[dict[str, Any]]:
    if max_violations <= 0 or len(violations) <= max_violations:
        return violations
    ranked = sorted(
        violations,
        key=lambda item: (int(item.get("step", 10**9)), -as_float(item.get("severity"), default=0.0)),
    )
    return ranked[:max_violations]


def strip_agentrx_step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for record in records:
        copy = dict(record)
        for key in ["dynamic_raw_response", "validation_raw_response"]:
            raw = copy.get(key)
            if isinstance(raw, str) and len(raw) > 1000:
                copy[key] = raw[:1000] + "...[truncated]"
        stripped.append(copy)
    return stripped


def case_with_steps(case: Case, steps: list[LogStep]) -> Case:
    return Case(
        case_id=case.case_id,
        problem=case.problem,
        steps=list(steps),
        ground_truth=case.ground_truth,
        final_answer=case.final_answer,
        gold_agent=case.gold_agent,
        gold_step=case.gold_step,
        metadata=dict(case.metadata),
    )


def case_with_side_information(
    case: Case,
    *,
    include_ground_truth: bool,
    include_final_answer: bool,
) -> Case:
    return Case(
        case_id=case.case_id,
        problem=case.problem,
        steps=list(case.steps),
        ground_truth=case.ground_truth if include_ground_truth else None,
        final_answer=case.final_answer if include_final_answer else None,
        gold_agent=case.gold_agent,
        gold_step=case.gold_step,
        metadata=dict(case.metadata),
    )


def agentrx_original_prediction_score(pred: Prediction) -> float:
    trace = pred.trace or {}
    validation_log = trace.get("validation_log")
    violation_count = len(validation_log) if isinstance(validation_log, list) else 0
    has_step = 1.0 if pred.step is not None else 0.0
    has_agent = 1.0 if pred.agent else 0.0
    violation_signal = min(1.0, violation_count / 3.0)
    confidence = as_float(pred.confidence, default=0.0)
    return 0.45 * has_step + 0.25 * has_agent + 0.20 * violation_signal + 0.10 * confidence


def compact_prediction_for_trace(pred: Prediction) -> dict[str, Any]:
    trace = pred.trace or {}
    return {
        "method": pred.method,
        "agent": pred.agent,
        "step": pred.step,
        "confidence": pred.confidence,
        "reason": pred.reason,
        "validation_log": trace.get("validation_log"),
        "judge_result": trace.get("judge_result"),
        "constraint_counts": trace.get("constraint_counts"),
    }


def agentrx_violation_score(violation: dict[str, Any]) -> float:
    recoverable_penalty = 0.25 if as_bool(violation.get("recoverable")) else 0.0
    return (
        0.45 * as_float(violation.get("severity"))
        + 0.35 * as_float(violation.get("confidence"))
        + 0.20
        - recoverable_penalty
    )


def select_validation_log_for_judge(violations: list[dict[str, Any]], max_violations: int) -> list[dict[str, Any]]:
    if max_violations <= 0:
        return violations
    unrecovered = [v for v in violations if not as_bool(v.get("recoverable"))]
    pool = unrecovered or violations
    selected = sorted(pool, key=lambda v: (-float(v.get("score", 0.0)), int(v.get("step", 10**9))))[:max_violations]
    return sorted(selected, key=lambda v: int(v.get("step", 10**9)))


def select_agentrx_fallback(violations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not violations:
        return None
    unrecovered = [v for v in violations if not as_bool(v.get("recoverable"))]
    pool = unrecovered or violations
    return sorted(pool, key=lambda v: (int(v.get("step", 10**9)), -float(v.get("score", 0.0))))[0]


def select_agentrx_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int, threshold: float) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    ranked = sorted(chunk_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected = [item for item in ranked if float(item.get("score", 0.0)) >= threshold][:beam_k]
    if not selected:
        selected = ranked[:beam_k]
    return sorted(selected, key=lambda x: int(x.get("chunk_start", 10**9)))


def resolve_echo_focuses(method_config: dict[str, Any], case: Case | None = None, phase: str = "") -> list[str]:
    configured = method_config.get("focuses")
    if isinstance(configured, list):
        focuses = [str(item) for item in configured if str(item).strip()]
    elif isinstance(configured, str) and configured.strip():
        focuses = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        focuses = ["conservative", "detail_focused", "pattern_focused"]
    analyst_count = int(method_config.get("analyst_count", len(focuses)))
    analyst_count = max(1, analyst_count)
    if as_bool(method_config.get("sample_focuses"), default=False) and case is not None and analyst_count < len(focuses):
        key = f"{case.case_id}:{phase}:{method_config.get('focus_seed', 'echo')}"
        return stable_focus_sample(focuses, analyst_count, key)
    return focuses[:analyst_count]


def stable_focus_sample(focuses: list[str], count: int, key: str) -> list[str]:
    ranked = sorted(
        focuses,
        key=lambda focus: hashlib.sha256(f"{key}:{focus}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


def resolve_echo_temperatures(method_config: dict[str, Any], analyst_count: int) -> list[float]:
    configured = method_config.get("temperatures")
    if isinstance(configured, list) and configured:
        values = [float(item) for item in configured]
        if len(values) >= analyst_count:
            return values[:analyst_count]
        return values + [values[-1]] * (analyst_count - len(values))
    if isinstance(configured, str) and configured.strip():
        values = [float(item.strip()) for item in configured.split(",") if item.strip()]
        if values:
            if len(values) >= analyst_count:
                return values[:analyst_count]
            return values + [values[-1]] * (analyst_count - len(values))
    if analyst_count <= 1:
        return [float(method_config.get("temperature_min", 0.30))]
    low = float(method_config.get("temperature_min", 0.30))
    high = float(method_config.get("temperature_max", 0.90))
    return [round(low + (high - low) * idx / (analyst_count - 1), 3) for idx in range(analyst_count)]


def echo_conversation_summary(steps: list[LogStep]) -> str:
    lines = ["=== CONVERSATION AGENTS ==="]
    for step in steps:
        lines.append(f"Step {step.step} - {step.agent}:")
        lines.append(step.content)
        lines.append("")
    return "\n".join(lines)


def run_echo_analyst_panel(
    case: Case,
    llm: LLM,
    conversation: str,
    focuses: list[str],
    phase: str,
    target_agents: list[str] | None = None,
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for focus in focuses:
        raw = llm.generate(
            echo_objective_analysis_prompt(
                case=case,
                conversation=conversation,
                analyst_focus=focus,
                phase=phase,
                target_agents=target_agents,
            )
        )
        parsed = safe_json(raw)
        parsed["analyst_focus"] = focus
        parsed["phase"] = phase
        parsed["raw_response"] = raw
        conclusion = echo_primary_conclusion(parsed)
        parsed["agent"] = conclusion.get("agent")
        parsed["step"] = conclusion.get("step")
        parsed["confidence"] = conclusion.get("confidence")
        parsed["reason"] = conclusion.get("reason")
        analyses.append(parsed)
    return analyses


def run_echo_original_analyst_panel(
    case: Case,
    llm: LLM,
    conversation: str,
    focuses: list[str],
    phase: str,
    chunk_id: int | None = None,
    target_agents: list[str] | None = None,
    temperatures: list[float] | None = None,
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for idx, focus in enumerate(focuses):
        prompt = (
            echo_original_objective_analysis_prompt(
                case=case,
                conversation=conversation,
                analyst_focus=focus,
                phase=phase,
                target_agents=target_agents,
            )
        )
        temperature = temperatures[idx] if temperatures and idx < len(temperatures) else None
        raw = generate_with_temperature(llm, prompt, temperature)
        parsed = safe_json(raw)
        parsed["analyst_focus"] = focus
        parsed["phase"] = phase
        if target_agents:
            parsed["target_agents"] = target_agents
        if temperature is not None:
            parsed["temperature"] = temperature
        if chunk_id is not None:
            parsed["chunk_id"] = chunk_id
        parsed["raw_response"] = raw
        conclusion = echo_primary_conclusion(parsed)
        parsed["agent"] = conclusion.get("agent")
        parsed["step"] = conclusion.get("step")
        parsed["confidence"] = conclusion.get("confidence")
        parsed["reason"] = conclusion.get("reason")
        analyses.append(parsed)
    return analyses


def run_echo_appendix_strict_analyst_panel(
    case: Case,
    llm: LLM,
    conversation: str,
    focuses: list[str],
    phase: str,
    chunk_id: int | None = None,
    target_agents: list[str] | None = None,
    temperatures: list[float] | None = None,
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for idx, focus in enumerate(focuses):
        prompt = echo_appendix_strict_objective_analysis_prompt(
            case=case,
            conversation=conversation,
            analyst_focus=focus,
        )
        temperature = temperatures[idx] if temperatures and idx < len(temperatures) else None
        raw = generate_with_temperature(llm, prompt, temperature)
        parsed = safe_json(raw)
        parsed["analyst_focus"] = focus
        parsed["phase"] = phase
        if target_agents:
            parsed["target_agents"] = target_agents
        if temperature is not None:
            parsed["temperature"] = temperature
        if chunk_id is not None:
            parsed["chunk_id"] = chunk_id
        parsed["raw_response"] = raw
        conclusion = echo_primary_conclusion(parsed)
        parsed["agent"] = conclusion.get("agent")
        parsed["step"] = conclusion.get("step")
        parsed["confidence"] = conclusion.get("confidence")
        parsed["reason"] = conclusion.get("reason")
        analyses.append(parsed)
    return analyses


def generate_with_temperature(llm: LLM, prompt: str, temperature: float | None) -> str:
    if temperature is None:
        return llm.generate(prompt)
    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "temperature"):
        return llm.generate(prompt)
    old_temperature = getattr(config, "temperature")
    old_omit_temperature = getattr(config, "omit_temperature", None)
    try:
        setattr(config, "temperature", float(temperature))
        if old_omit_temperature is not None:
            setattr(config, "omit_temperature", False)
        return llm.generate(prompt)
    finally:
        setattr(config, "temperature", old_temperature)
        if old_omit_temperature is not None:
            setattr(config, "omit_temperature", old_omit_temperature)


def generate_with_system_prompt(llm: LLM, prompt: str, system_prompt: str) -> str:
    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "system_prompt"):
        return llm.generate(prompt)
    old_system_prompt = getattr(config, "system_prompt")
    try:
        setattr(config, "system_prompt", system_prompt)
        return llm.generate(prompt)
    finally:
        setattr(config, "system_prompt", old_system_prompt)


def echo_primary_conclusion(parsed: dict[str, Any]) -> dict[str, Any]:
    primary = parsed.get("primary_conclusion") if isinstance(parsed.get("primary_conclusion"), dict) else {}
    attribution = primary.get("attribution")
    agent: str | None = None
    if isinstance(attribution, list) and attribution:
        agent = normalize_optional_str(attribution[0])
    elif isinstance(attribution, str):
        agent = normalize_optional_str(attribution)
    step = parse_int_maybe(primary.get("mistake_step") or primary.get("step") or parsed.get("step"))
    confidence = as_float(primary.get("confidence") or parsed.get("confidence"), default=0.0)
    reason = normalize_optional_str(primary.get("reasoning") or parsed.get("analysis_summary") or parsed.get("reason"))
    return {"agent": agent, "step": step, "confidence": confidence, "reason": reason}


def select_echo_target_agents(analyses: list[dict[str, Any]], agent_k: int) -> list[str]:
    weights: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for analysis in analyses:
        candidates: list[tuple[str, float, int]] = []
        agent = normalize_optional_str(analysis.get("agent"))
        if agent:
            candidates.append((agent, as_float(analysis.get("confidence"), default=0.0), parse_int_maybe(analysis.get("step")) or 10**9))
        for item in analysis.get("agent_evaluations") or []:
            if not isinstance(item, dict):
                continue
            eval_agent = normalize_optional_str(item.get("agent_name"))
            if not eval_agent:
                continue
            candidates.append(
                (
                    eval_agent,
                    as_float(item.get("error_likelihood"), default=0.0),
                    parse_int_maybe(item.get("step_index")) or 10**9,
                )
            )
        for agent_name, weight, step in candidates:
            weights[agent_name] = weights.get(agent_name, 0.0) + weight
            first_seen[agent_name] = min(first_seen.get(agent_name, 10**9), step)
    ranked = sorted(weights, key=lambda agent: (-weights[agent], first_seen.get(agent, 10**9), agent))
    return ranked[: max(1, agent_k)]


def resolve_echo_chunk_focuses(method_config: dict[str, Any]) -> list[str]:
    configured = method_config.get("chunk_focuses")
    if isinstance(configured, list):
        focuses = [str(item) for item in configured if str(item).strip()]
    elif isinstance(configured, str) and configured.strip():
        focuses = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        focuses = ["general"]
    count = int(method_config.get("chunk_analyst_count", len(focuses)))
    return focuses[: max(1, count)]


def select_echo_consensus(
    analyses: list[dict[str, Any]],
    confidence_threshold: float = 0.0,
    valid_steps: set[int] | None = None,
) -> dict[str, Any]:
    if not analyses:
        return echo_empty_consensus(confidence_threshold)

    primary_conclusions: list[dict[str, Any]] = []
    all_agent_evaluations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_alternative_hypotheses: list[dict[str, Any]] = []

    for analyst_id, analysis in enumerate(analyses):
        primary = analysis.get("primary_conclusion")
        if isinstance(primary, dict):
            conclusion = dict(primary)
            conclusion["analyst_id"] = analyst_id
            primary_conclusions.append(conclusion)

        evaluations = analysis.get("agent_evaluations")
        if isinstance(evaluations, list):
            for eval_item in evaluations:
                if not isinstance(eval_item, dict):
                    continue
                agent_name = normalize_optional_str(eval_item.get("agent_name"))
                if not agent_name:
                    continue
                all_agent_evaluations[agent_name].append(
                    {
                        "error_likelihood": as_float(eval_item.get("error_likelihood"), default=0.0),
                        "reasoning": normalize_optional_str(eval_item.get("reasoning")) or "",
                        "evidence": normalize_optional_str(eval_item.get("evidence")) or "",
                        "analyst_id": analyst_id,
                        "step_index": parse_int_maybe(eval_item.get("step_index")),
                    }
                )

        alternatives = analysis.get("alternative_hypotheses")
        if isinstance(alternatives, list):
            for alt in alternatives:
                if isinstance(alt, dict):
                    alt_copy = dict(alt)
                    alt_copy["analyst_id"] = analyst_id
                    all_alternative_hypotheses.append(alt_copy)

    consensus = perform_echo_consensus_voting(
        primary_conclusions=primary_conclusions,
        agent_evaluations=all_agent_evaluations,
        alternative_hypotheses=all_alternative_hypotheses,
        confidence_threshold=confidence_threshold,
        valid_steps=valid_steps,
        num_analysts=len(analyses),
    )
    conclusion = consensus.get("consensus_conclusion", {})
    attribution = conclusion.get("attribution")
    agent = attribution[0] if isinstance(attribution, list) and attribution else None
    step = parse_int_maybe(conclusion.get("mistake_step"))
    return {
        **consensus,
        "agent": normalize_optional_str(agent),
        "step": step,
        "confidence": as_float(conclusion.get("confidence"), default=None),  # type: ignore[arg-type]
        "reason": normalize_optional_str(conclusion.get("reasoning")),
    }


def perform_echo_consensus_voting(
    primary_conclusions: list[dict[str, Any]],
    agent_evaluations: dict[str, list[dict[str, Any]]],
    alternative_hypotheses: list[dict[str, Any]],
    confidence_threshold: float,
    valid_steps: set[int] | None,
    num_analysts: int,
) -> dict[str, Any]:
    conclusion_votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conclusion in primary_conclusions:
        confidence = as_float(conclusion.get("confidence"), default=0.0)
        if confidence < confidence_threshold:
            continue
        conclusion_type = normalize_echo_conclusion_type(conclusion.get("type"))
        conclusion_votes[conclusion_type].append(
            {
                "confidence": confidence,
                "attribution": normalize_echo_attribution(conclusion.get("attribution")),
                "mistake_step": parse_int_maybe(conclusion.get("mistake_step") or conclusion.get("step")),
                "reasoning": normalize_optional_str(conclusion.get("reasoning")) or "",
                "analyst_id": conclusion.get("analyst_id"),
            }
        )

    best_conclusion_type: str | None = None
    best_conclusion_info: dict[str, Any] | None = None
    best_weighted_score = 0.0
    for conclusion_type, votes in conclusion_votes.items():
        total_confidence = sum(as_float(vote.get("confidence"), default=0.0) for vote in votes)
        avg_confidence = total_confidence / len(votes) if votes else 0.0
        weighted_score = total_confidence
        if weighted_score > best_weighted_score:
            best_weighted_score = weighted_score
            best_conclusion_type = conclusion_type
            best_conclusion_info = {
                "votes": votes,
                "avg_confidence": avg_confidence,
                "total_confidence": total_confidence,
                "num_votes": len(votes),
            }

    final_attribution = echo_consensus_final_attribution(
        best_conclusion_type,
        best_conclusion_info,
        confidence_threshold,
    )
    aggregated_agent_evaluations = aggregate_echo_agent_evaluations(agent_evaluations)
    consensus_mistake_step, step_attribution_votes = echo_consensus_step_vote(best_conclusion_type, best_conclusion_info, valid_steps)
    disagreement_info = analyze_echo_disagreements(conclusion_votes)

    return {
        "consensus_conclusion": {
            "type": best_conclusion_type or "single_agent",
            "attribution": final_attribution,
            "mistake_step": consensus_mistake_step,
            "confidence": best_conclusion_info["avg_confidence"] if best_conclusion_info else 0.0,
            "reasoning": synthesize_echo_reasoning(best_conclusion_info) if best_conclusion_info else "No clear consensus reached",
        },
        "voting_details": {
            "conclusion_votes": dict(conclusion_votes),
            "step_votes": step_attribution_votes,
            "best_weighted_score": best_weighted_score,
            "disagreement_analysis": disagreement_info,
        },
        "agent_evaluations_summary": aggregated_agent_evaluations,
        "alternative_hypotheses": all_echo_alternatives_top5(alternative_hypotheses),
        "num_analysts": num_analysts,
        "voting_method": "weighted_confidence_consensus",
        "confidence_threshold": confidence_threshold,
    }


def normalize_echo_conclusion_type(value: Any) -> str:
    text = normalize_optional_str(value) or "single_agent"
    normalized = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    aliases = {
        "single": "single_agent",
        "singleagent": "single_agent",
        "single_agent_error": "single_agent",
        "one_agent": "single_agent",
        "multi": "multi_agent",
        "multiagent": "multi_agent",
        "multi_agent_error": "multi_agent",
        "multiple_agent": "multi_agent",
        "multiple_agents": "multi_agent",
    }
    return aliases.get(normalized, normalized if normalized in {"single_agent", "multi_agent"} else "single_agent")


def normalize_echo_attribution(value: Any) -> list[str]:
    if isinstance(value, list):
        return [agent for item in value if (agent := normalize_optional_str(item))]
    agent = normalize_optional_str(value)
    return [agent] if agent else []


def echo_consensus_attribution_agents(consensus: dict[str, Any]) -> list[str]:
    conclusion = consensus.get("consensus_conclusion")
    if not isinstance(conclusion, dict):
        return []
    return normalize_echo_attribution(conclusion.get("attribution"))


def echo_consensus_final_attribution(
    best_conclusion_type: str | None,
    best_conclusion_info: dict[str, Any] | None,
    confidence_threshold: float,
) -> list[str] | None:
    if best_conclusion_type not in {"single_agent", "multi_agent"} or not best_conclusion_info:
        return None
    agent_attribution_votes: dict[str, float] = defaultdict(float)
    for vote in best_conclusion_info.get("votes", []):
        for agent_name in normalize_echo_attribution(vote.get("attribution")):
            agent_attribution_votes[agent_name] += as_float(vote.get("confidence"), default=0.0)
    if not agent_attribution_votes:
        return None
    # Match the appendix consensus logic: sort by accumulated confidence only.
    # Python's stable sort preserves the first-seen analyst/attribution order on ties.
    sorted_agents = sorted(agent_attribution_votes.items(), key=lambda item: -item[1])
    if best_conclusion_type == "single_agent":
        return [sorted_agents[0][0]]
    return [agent for agent, confidence in sorted_agents if confidence >= confidence_threshold] or [sorted_agents[0][0]]


def aggregate_echo_agent_evaluations(agent_evaluations: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for agent_name, evaluations in agent_evaluations.items():
        likelihoods = [as_float(item.get("error_likelihood"), default=0.0) for item in evaluations]
        avg_error_likelihood = sum(likelihoods) / len(likelihoods) if likelihoods else 0.0
        aggregated[agent_name] = {
            "avg_error_likelihood": avg_error_likelihood,
            "num_evaluations": len(evaluations),
            "evaluations": evaluations,
        }
    return aggregated


def echo_consensus_step_vote(
    best_conclusion_type: str | None,
    best_conclusion_info: dict[str, Any] | None,
    valid_steps: set[int] | None,
) -> tuple[int | None, dict[int, float]]:
    step_votes: dict[int, float] = defaultdict(float)
    if best_conclusion_type not in {"single_agent", "multi_agent"} or not best_conclusion_info:
        return None, {}
    for vote in best_conclusion_info.get("votes", []):
        step = parse_int_maybe(vote.get("mistake_step"))
        if step is None:
            continue
        step_votes[step] += as_float(vote.get("confidence"), default=0.0)
    if not step_votes:
        return None, {}
    if valid_steps:
        valid_step_votes = {step: score for step, score in step_votes.items() if step in valid_steps}
    else:
        valid_step_votes = dict(step_votes)
    if not valid_step_votes:
        return None, dict(step_votes)
    consensus_step = sorted(valid_step_votes, key=lambda step: (-valid_step_votes[step], step))[0]
    return consensus_step, dict(step_votes)


def analyze_echo_disagreements(conclusion_votes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    num_conclusion_types = len(conclusion_votes)
    high_disagreement = num_conclusion_types > 2 and all(len(votes) > 0 for votes in conclusion_votes.values())
    all_confidences: list[float] = []
    for votes in conclusion_votes.values():
        all_confidences.extend(as_float(vote.get("confidence"), default=0.0) for vote in votes)
    confidence_spread = max(all_confidences) - min(all_confidences) if all_confidences else 0.0
    return {
        "high_disagreement": high_disagreement,
        "num_different_conclusions": num_conclusion_types,
        "confidence_spread": confidence_spread,
        "requires_review": high_disagreement or confidence_spread > 0.5,
    }


def synthesize_echo_reasoning(best_conclusion_info: dict[str, Any] | None) -> str:
    if not best_conclusion_info or not best_conclusion_info.get("votes"):
        return "No reasoning available"
    votes = best_conclusion_info["votes"]
    num_votes = len(votes)
    avg_confidence = as_float(best_conclusion_info.get("avg_confidence"), default=0.0)
    reasonings = [str(vote.get("reasoning", "")).strip() for vote in votes if str(vote.get("reasoning", "")).strip()]
    if not reasonings:
        return f"Consensus reached by {num_votes} analysts with average confidence {avg_confidence:.2f}."
    synthesis = f"Consensus reached by {num_votes} analysts (avg confidence: {avg_confidence:.2f}). "
    synthesis += f"Primary reasoning: {reasonings[0][:200]}..."
    if len(reasonings) > 1:
        synthesis += f" Additional supporting analysis from {len(reasonings) - 1} other analysts."
    return synthesis


def all_echo_alternatives_top5(alternative_hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        alternative_hypotheses,
        key=lambda item: -as_float(item.get("confidence"), default=0.0),
    )[:5]


def echo_step_index_map(steps: list[LogStep], index_base: int = 0) -> dict[int, int]:
    return {idx + index_base: step.step for idx, step in enumerate(steps)}


def map_echo_consensus_to_dataset_step(consensus: dict[str, Any], step_index_map: dict[int, int]) -> dict[str, Any]:
    mapped = dict(consensus)
    echo_step_index = parse_int_maybe(mapped.get("step"))
    dataset_step = step_index_map.get(echo_step_index) if echo_step_index is not None else None
    mapped["echo_step_index"] = echo_step_index
    mapped["step"] = dataset_step
    conclusion = mapped.get("consensus_conclusion")
    if isinstance(conclusion, dict):
        mapped_conclusion = dict(conclusion)
        mapped_conclusion["echo_mistake_step_index"] = parse_int_maybe(mapped_conclusion.get("mistake_step"))
        if mapped_conclusion["echo_mistake_step_index"] is not None:
            mapped_conclusion["mistake_step"] = step_index_map.get(mapped_conclusion["echo_mistake_step_index"])
        mapped["consensus_conclusion"] = mapped_conclusion
    voting_details = mapped.get("voting_details")
    if isinstance(voting_details, dict) and isinstance(voting_details.get("step_votes"), dict):
        dataset_step_votes: dict[int, float] = {}
        for raw_step, score in voting_details["step_votes"].items():
            step_index = parse_int_maybe(raw_step)
            if step_index is None or step_index not in step_index_map:
                continue
            dataset_step_votes[step_index_map[step_index]] = nonnegative_float(score)
        mapped_voting_details = dict(voting_details)
        mapped_voting_details["dataset_step_votes"] = dataset_step_votes
        mapped["voting_details"] = mapped_voting_details
    return mapped


def nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, out)


def echo_empty_consensus(confidence_threshold: float) -> dict[str, Any]:
    return {
        "agent": None,
        "step": None,
        "confidence": 0.0,
        "reason": "No objective analyses provided",
        "consensus_conclusion": {
            "type": "single_agent",
            "attribution": None,
            "mistake_step": None,
            "confidence": 0.0,
            "reasoning": "No objective analyses provided",
        },
        "voting_details": {
            "conclusion_votes": {},
            "best_weighted_score": 0.0,
            "disagreement_analysis": {
                "high_disagreement": False,
                "num_different_conclusions": 0,
                "confidence_spread": 0.0,
                "requires_review": True,
            },
        },
        "agent_evaluations_summary": {},
        "alternative_hypotheses": [],
        "num_analysts": 0,
        "voting_method": "weighted_confidence_consensus",
        "confidence_threshold": confidence_threshold,
    }


def echo_chunk_score(result: dict[str, Any]) -> float:
    contains = 1.0 if as_bool(result.get("contains_attribution_evidence")) else 0.0
    confidence = as_float(result.get("confidence"), default=0.0)
    has_step = 1.0 if parse_int_maybe(result.get("step")) is not None else 0.0
    return 0.55 * contains + 0.35 * confidence + 0.10 * has_step


def echo_original_chunk_score(result: dict[str, Any]) -> float:
    confidence = as_float(result.get("confidence"), default=0.0)
    has_step = 1.0 if parse_int_maybe(result.get("step")) is not None else 0.0
    has_agent = 1.0 if normalize_optional_str(result.get("agent")) else 0.0
    return 0.70 * confidence + 0.20 * has_step + 0.10 * has_agent


def select_echo_beam_chunks(chunk_results: list[dict[str, Any]], beam_k: int, threshold: float) -> list[dict[str, Any]]:
    if not chunk_results:
        return []
    ranked = sorted(chunk_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected = [item for item in ranked if float(item.get("score", 0.0)) >= threshold][:beam_k]
    if not selected:
        selected = ranked[:beam_k]
    return sorted(selected, key=lambda x: int(x.get("chunk_start", 10**9)))


def select_echo_global_router_chunks(
    router_result: dict[str, Any],
    chunks: list[list[LogStep]],
    chunking: dict[str, Any],
    beam_k: int,
    selected_token_budget: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_id = len(chunks)
    ranges = chunking.get("chunk_ranges", [])
    if not isinstance(ranges, list) or len(ranges) != max_id:
        ranges = [
            {
                "chunk_id": idx + 1,
                "start_step": chunk[0].step,
                "end_step": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
            }
            for idx, chunk in enumerate(chunks)
        ]
    range_by_id = {int(item.get("chunk_id", idx + 1)): item for idx, item in enumerate(ranges)}
    selected_ids: list[int] = []
    seen: set[int] = set()
    score_by_id: dict[int, float] = {}
    reason_by_id: dict[int, str] = {}

    def add_id(value: Any, score: Any = None, reason: Any = None) -> None:
        cid = parse_int_maybe(value)
        if cid is None or cid < 1 or cid > max_id or cid in seen:
            return
        seen.add(cid)
        selected_ids.append(cid)
        if score is not None:
            score_by_id[cid] = as_float(score, default=0.0)
        if reason:
            reason_by_id[cid] = str(reason)

    for key in ("selected_chunk_ids", "top_chunk_ids", "chunk_ids", "selected_chunks"):
        value = router_result.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    add_id(
                        item.get("chunk_id") or item.get("id"),
                        item.get("score") or item.get("confidence"),
                        item.get("reason") or item.get("rationale"),
                    )
                else:
                    add_id(item)
        elif value is not None:
            add_id(value)

    normalized_candidates: list[dict[str, Any]] = []
    raw_candidates = router_result.get("candidates") or router_result.get("candidate_chunks")
    if isinstance(raw_candidates, list):
        for rank, item in enumerate(raw_candidates):
            if isinstance(item, dict):
                cid = parse_int_maybe(item.get("chunk_id") or item.get("id"))
                score = as_float(item.get("score") or item.get("confidence"), default=0.0)
                reason = normalize_optional_str(item.get("reason") or item.get("rationale"))
            else:
                cid = parse_int_maybe(item)
                score = 0.0
                reason = None
            if cid is None or cid < 1 or cid > max_id:
                continue
            normalized_candidates.append(
                {
                    "chunk_id": cid,
                    "score": score,
                    "reason": reason,
                    "router_candidate_rank": rank + 1,
                }
            )
            score_by_id.setdefault(cid, score)
            if reason:
                reason_by_id.setdefault(cid, reason)

    if len(selected_ids) < beam_k:
        ranked_candidates = sorted(
            normalized_candidates,
            key=lambda item: (-float(item.get("score", 0.0)), int(item.get("router_candidate_rank", 10**9))),
        )
        for item in ranked_candidates:
            add_id(item.get("chunk_id"), item.get("score"), item.get("reason"))
            if len(selected_ids) >= beam_k:
                break

    if len(selected_ids) < beam_k:
        for cid in range(1, max_id + 1):
            add_id(cid, 0.0, "Fallback fill: router returned fewer than beam_k valid chunk IDs.")
            if len(selected_ids) >= beam_k:
                break

    selected_records: list[dict[str, Any]] = []
    for preference_rank, cid in enumerate(selected_ids[:beam_k], start=1):
        meta = range_by_id.get(cid, {})
        chunk = chunks[cid - 1]
        selected_records.append(
            {
                "chunk_id": cid,
                "chunk_start": int(meta.get("start_step", chunk[0].step)),
                "chunk_end": int(meta.get("end_step", chunk[-1].step)),
                "step_count": int(meta.get("step_count", len(chunk))),
                "estimated_tokens": meta.get("estimated_tokens"),
                "agent": None,
                "step": None,
                "confidence": score_by_id.get(cid, 0.0),
                "score": score_by_id.get(cid, 0.0),
                "reason": reason_by_id.get(cid),
                "router_preference_rank": preference_rank,
                "selection_source": "global_router",
            }
        )
    if selected_token_budget is not None and selected_token_budget > 0:
        selected_records = select_records_by_token_budget(
            selected_records,
            target_tokens=selected_token_budget,
        )
    return sorted(selected_records, key=lambda x: int(x.get("chunk_start", 10**9))), normalized_candidates


def select_records_by_token_budget(
    records: list[dict[str, Any]],
    *,
    target_tokens: int,
) -> list[dict[str, Any]]:
    if not records or target_tokens <= 0:
        return records
    ranked = sorted(records, key=lambda item: int(item.get("router_preference_rank", 10**9)))
    selected: list[dict[str, Any]] = []
    total = 0
    for item in ranked:
        tokens = int(item.get("estimated_tokens") or 0)
        if not selected:
            selected.append(item)
            total += tokens
            continue
        if total >= target_tokens:
            break
        with_item_total = total + tokens
        without_gap = abs(target_tokens - total)
        with_gap = abs(target_tokens - with_item_total)
        if with_item_total <= target_tokens or with_gap <= without_gap:
            selected.append(item)
            total = with_item_total
        else:
            break
    return selected or ranked[:1]


def parse_echo_router_preferences(router_result: dict[str, Any], max_id: int) -> tuple[list[int], list[dict[str, Any]]]:
    selected_ids: list[int] = []
    seen: set[int] = set()

    def add_id(value: Any) -> None:
        cid = parse_int_maybe(value)
        if cid is None or cid < 1 or cid > max_id or cid in seen:
            return
        seen.add(cid)
        selected_ids.append(cid)

    for key in ("selected_chunk_ids", "top_chunk_ids", "chunk_ids", "selected_chunks"):
        value = router_result.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    add_id(item.get("chunk_id") or item.get("id"))
                else:
                    add_id(item)
        elif value is not None:
            add_id(value)

    normalized_candidates: list[dict[str, Any]] = []
    raw_candidates = router_result.get("candidates") or router_result.get("candidate_chunks")
    if isinstance(raw_candidates, list):
        for rank, item in enumerate(raw_candidates):
            if isinstance(item, dict):
                cid = parse_int_maybe(item.get("chunk_id") or item.get("id"))
                score = as_float(item.get("score") or item.get("confidence"), default=0.0)
                reason = normalize_optional_str(item.get("reason") or item.get("rationale"))
            else:
                cid = parse_int_maybe(item)
                score = 0.0
                reason = None
            if cid is None or cid < 1 or cid > max_id:
                continue
            normalized_candidates.append(
                {
                    "chunk_id": cid,
                    "score": score,
                    "reason": reason,
                    "router_candidate_rank": rank + 1,
                }
            )

    if not selected_ids and normalized_candidates:
        for item in sorted(
            normalized_candidates,
            key=lambda candidate: (-float(candidate.get("score", 0.0)), int(candidate.get("router_candidate_rank", 10**9))),
        ):
            add_id(item.get("chunk_id"))

    return selected_ids, normalized_candidates


def select_echo_panel_router_chunks(
    router_analyses: list[dict[str, Any]],
    chunks: list[list[LogStep]],
    chunking: dict[str, Any],
    beam_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_id = len(chunks)
    ranges = chunking.get("chunk_ranges", [])
    if not isinstance(ranges, list) or len(ranges) != max_id:
        ranges = [
            {
                "chunk_id": idx + 1,
                "start_step": chunk[0].step,
                "end_step": chunk[-1].step,
                "step_count": len(chunk),
                "estimated_tokens": estimate_steps_tokens(chunk),
            }
            for idx, chunk in enumerate(chunks)
        ]
    range_by_id = {int(item.get("chunk_id", idx + 1)): item for idx, item in enumerate(ranges)}

    vote_by_id: dict[int, float] = defaultdict(float)
    support_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    analyst_preferences: list[dict[str, Any]] = []
    for analyst_idx, analysis in enumerate(router_analyses):
        selected_ids, candidates = parse_echo_router_preferences(analysis, max_id=max_id)
        selected_ids = selected_ids[:beam_k]
        analyst_focus = normalize_optional_str(analysis.get("analyst_focus")) or f"analyst_{analyst_idx + 1}"
        analyst_preferences.append(
            {
                "analyst_index": analyst_idx,
                "analyst_focus": analyst_focus,
                "selected_chunk_ids": selected_ids,
                "candidate_chunks": candidates,
            }
        )
        for rank, cid in enumerate(selected_ids, start=1):
            points = 1.0
            vote_by_id[cid] += points
            support_by_id[cid].append(
                {
                    "analyst_index": analyst_idx,
                    "analyst_focus": analyst_focus,
                    "rank": rank,
                    "points": points,
                }
            )

    if len(vote_by_id) < beam_k:
        for cid in range(1, max_id + 1):
            if cid not in vote_by_id:
                vote_by_id[cid] = 0.0
                support_by_id[cid].append(
                    {
                        "analyst_index": None,
                        "analyst_focus": "fallback_fill",
                        "rank": None,
                        "points": 0.0,
                    }
                )
            if len(vote_by_id) >= beam_k:
                break

    ranked_ids = sorted(
        vote_by_id,
        key=lambda cid: (
            -float(vote_by_id[cid]),
            int(range_by_id.get(cid, {}).get("start_step", chunks[cid - 1][0].step)),
            cid,
        ),
    )[:beam_k]

    selected_records: list[dict[str, Any]] = []
    max_vote = max(1, len(router_analyses))
    for preference_rank, cid in enumerate(ranked_ids, start=1):
        meta = range_by_id.get(cid, {})
        chunk = chunks[cid - 1]
        vote = float(vote_by_id[cid])
        selected_records.append(
            {
                "chunk_id": cid,
                "chunk_start": int(meta.get("start_step", chunk[0].step)),
                "chunk_end": int(meta.get("end_step", chunk[-1].step)),
                "step_count": int(meta.get("step_count", len(chunk))),
                "estimated_tokens": meta.get("estimated_tokens"),
                "agent": None,
                "step": None,
                "confidence": vote / max_vote if max_vote else 0.0,
            "score": vote,
                "reason": "Selected by ECHO-style panel approval vote over global chunk-router outputs.",
            "router_preference_rank": preference_rank,
            "router_vote": vote,
            "router_support": support_by_id[cid],
            "selection_source": "panel_router_approval_vote",
        }
    )

    vote_details = {
        "aggregation": "approval_vote_without_confidence_or_rank",
        "approval_points_per_selected_chunk": 1.0,
        "tie_break": "earlier chunk_start, then lower chunk_id",
        "analyst_preferences": analyst_preferences,
        "chunk_votes": {str(cid): float(vote_by_id[cid]) for cid in sorted(vote_by_id)},
        "selected_chunk_ids_by_vote": ranked_ids,
    }
    return sorted(selected_records, key=lambda x: int(x.get("chunk_start", 10**9))), vote_details


def select_echo_fallback_chunk(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunk_results:
        return {"agent": None, "step": None, "confidence": None, "reason": "No ECHO chunk candidates."}
    candidates = [item for item in chunk_results if parse_int_maybe(item.get("step")) is not None]
    pool = candidates or chunk_results
    return sorted(pool, key=lambda x: (-float(x.get("score", 0.0)), int(x.get("chunk_start", 10**9))))[0]


def parse_int_maybe(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def agent_at_step(steps: list[LogStep], step_number: int | None) -> str | None:
    if step_number is None:
        return None
    for step in steps:
        if int(step.step) == int(step_number):
            return step.agent
    return None


def safe_json(raw: str) -> dict[str, Any]:
    try:
        parsed = extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        return {"parse_error": str(exc), "raw_response": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def parse_a2p_official_response(raw: str) -> dict[str, Any]:
    parsed = safe_json(raw)
    if not parsed.get("parse_error") and (parsed_agent(parsed) or parsed_step(parsed) is not None):
        return parsed

    agent_match = re.search(
        r"(?im)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*Agent\s+Name\s*\**\s*:?\s*\**\s*:?\s*(.+?)\s*$",
        raw,
    )
    fallback_agent_match = None
    if agent_match is None:
        # Some otherwise valid official-style responses put only ``AgentName:``
        # on the first line and omit the ``Agent Name`` field label.
        first_nonempty = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        fallback_agent_match = re.fullmatch(
            r"(?:[-*]\s*)?\**\s*([A-Za-z][A-Za-z0-9_.-]{1,80})\s*\**\s*:\s*",
            first_nonempty,
        )
        if fallback_agent_match and fallback_agent_match.group(1).lower() in {
            "agent",
            "reason",
            "step",
            "prediction",
            "answer",
        }:
            fallback_agent_match = None
    agent_value = clean_a2p_field(
        agent_match.group(1) if agent_match else fallback_agent_match.group(1) if fallback_agent_match else None
    )
    step_match = re.search(
        r"(?im)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*Step\s+Number\s*\**\s*:?\s*\**\s*:?\s*(-?\d+)\s*$",
        raw,
    )
    candidate_match = re.search(
        r"(?im)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*Candidate\s+ID\s*\**\s*:?\s*\**\s*:?\s*(\d+)\s*$",
        raw,
    )
    score_match = re.search(
        r"(?im)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*Causal\s+Score\s*\**\s*:?\s*\**\s*:?\s*([01](?:\.\d+)?)\s*$",
        raw,
    )
    contains_match = re.search(
        r"(?im)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*(?:Chunk|Segment|Candidate)\s+(?:Contains\s+Root\s+Cause|Contains\s+Responsible\s+Error|Is\s+Root\s+Cause)\s*\**\s*:?\s*\**\s*:?\s*(Yes|No)\s*$",
        raw,
    )
    reason_match = re.search(
        r"(?ims)^\s*(?:(?:[-*]|\d+[.)])\s*)?\**\s*Reason\s+for\s+Mistake\s*\**\s*:?\s*\**\s*:?\s*(.+)$",
        raw,
    )
    step_value = int(step_match.group(1)) if step_match else None
    if step_value is not None and step_value < 0:
        step_value = None
    contains = contains_match.group(1).strip().lower() == "yes" if contains_match else None
    out: dict[str, Any] = {
        "Agent Name": agent_value,
        "agent": agent_value,
        "Step Number": step_value,
        "step": step_value,
        "Reason for Mistake": clean_a2p_field(reason_match.group(1)) if reason_match else None,
        "reason": clean_a2p_field(reason_match.group(1)) if reason_match else None,
        "raw_response": raw,
    }
    if fallback_agent_match:
        out["agent_recovered_from_unlabeled_lead"] = True
    if candidate_match:
        out["candidate_id"] = int(candidate_match.group(1))
    if score_match:
        out["causal_score"] = as_float(score_match.group(1), default=0.0)
        out["score"] = out["causal_score"]
    if contains is not None:
        out["contains_counterfactual_error"] = contains
        out["would_fix_failure"] = contains
        if not contains:
            out["agent"] = None
            out["Agent Name"] = "NONE"
            out["step"] = None
            out["Step Number"] = None
    if parsed.get("parse_error") and not (agent_match or fallback_agent_match or step_match or reason_match):
        out["parse_error"] = parsed.get("parse_error")
    return out


def parse_who_when_official_step_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    first_line = text.splitlines()[0].strip().lower() if text else ""
    yes_match = re.search(r"(?im)^\s*(?:1\.\s*)?yes\b", raw)
    no_match = re.search(r"(?im)^\s*(?:1\.\s*)?no\b", raw)
    reason_match = re.search(r"(?ims)^\s*(?:2\.\s*)?Reason\s*:?\s*(.+)$", raw)
    contains_error = None
    if yes_match and (not no_match or yes_match.start() <= no_match.start()):
        contains_error = True
    elif no_match:
        contains_error = False
    elif first_line.startswith("yes"):
        contains_error = True
    elif first_line.startswith("no"):
        contains_error = False
    return {
        "contains_error": contains_error,
        "reason": clean_a2p_field(reason_match.group(1)) if reason_match else None,
    }


def parse_who_when_official_binary_response(raw: str) -> str:
    lower = raw.lower()
    upper_idx = lower.find("upper half")
    lower_idx = lower.find("lower half")
    if lower_idx >= 0 and (upper_idx < 0 or lower_idx < upper_idx):
        return "lower"
    return "upper"


def clean_a2p_field(value: Any) -> str | None:
    text = normalize_optional_str(value)
    if text is None:
        return None
    text = re.sub(r"^\s*[-*]+\s*", "", text).strip()
    text = text.strip("*").strip()
    return text or None


def extract_constraints(constraints_obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    if isinstance(constraints_obj, list):
        return [item for item in constraints_obj if isinstance(item, dict)]
    if not isinstance(constraints_obj, dict):
        return None
    constraints = constraints_obj.get("constraints") if isinstance(constraints_obj, dict) else None
    if isinstance(constraints, list):
        return [item for item in constraints if isinstance(item, dict)]
    if isinstance(constraints, dict):
        values = list(constraints.values())
        if all(isinstance(item, dict) for item in values):
            return values
    value = constraints_obj.get("value") if isinstance(constraints_obj, dict) else None
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return extract_constraints(value)
    raw = constraints_obj.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = extract_json(raw)
        except Exception:
            parsed = None
        if parsed is not None:
            return extract_constraints(parsed)
    if any(key in constraints_obj for key in ("assertion_name", "event_trigger", "check_hint", "nl_check")):
        return [constraints_obj]
    return None


def extract_checklist_items(checklist_obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    if isinstance(checklist_obj, list):
        return [item for item in checklist_obj if isinstance(item, dict)]
    if not isinstance(checklist_obj, dict):
        return None
    checklist_items = checklist_obj.get("checklist_items")
    if isinstance(checklist_items, list):
        return [item for item in checklist_items if isinstance(item, dict)]
    if isinstance(checklist_items, dict):
        values = list(checklist_items.values())
        if all(isinstance(item, dict) for item in values):
            return values
    value = checklist_obj.get("value")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return extract_checklist_items(value)
    return None


def resolve_ccv_constraints(
    case: Case,
    llm: LLM,
    method_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    cache_path = normalize_optional_str(method_config.get("constraint_cache_path"))
    fingerprint = case_fingerprint(case)
    if cache_path:
        cache = load_ccv_constraint_cache(cache_path)
        entry = cache.get(fingerprint)
        if entry is None:
            if bool(method_config.get("constraint_cache_strict", True)):
                raise KeyError(
                    "No shared CCV constraints for this trajectory fingerprint: "
                    f"case_id={case.case_id}, fingerprint={fingerprint}, cache={cache_path}"
                )
        else:
            constraints = entry.get("constraints")
            if not isinstance(constraints, list) or not constraints or not all(
                isinstance(constraint, dict) for constraint in constraints
            ):
                raise ValueError(
                    "Shared CCV constraint cache contains an invalid constraint set: "
                    f"case_id={case.case_id}, fingerprint={fingerprint}, cache={cache_path}"
                )
            raw_value = entry.get("constraints_raw_response")
            constraints_raw = (
                raw_value
                if isinstance(raw_value, str) and raw_value.strip()
                else json_dumps_compact({"constraints": constraints})
            )
            return constraints, constraints_raw, {
                "constraint_source": "shared_cache",
                "constraint_cache_path": cache_path,
                "case_fingerprint": fingerprint,
                "constraint_fingerprint": constraint_fingerprint(constraints),
            }

    constraints_raw = llm.generate(ccv_constraint_prompt(case))
    constraints_obj = safe_json(constraints_raw)
    constraints = extract_constraints(constraints_obj)
    used_default = not isinstance(constraints, list) or not constraints
    if used_default:
        constraints = default_constraints()
    return constraints, constraints_raw, {
        "constraint_source": "generated_default" if used_default else "generated",
        "constraint_cache_path": None,
        "case_fingerprint": fingerprint,
        "constraint_fingerprint": constraint_fingerprint(constraints),
    }


def json_dumps_compact(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chunk_level_candidates(chunk_results: list[dict[str, Any]], steps: list[LogStep]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result in chunk_results:
        for item in result.get("candidate_steps", []) or []:
            if not isinstance(item, dict) or item.get("step") is None:
                continue
            step = int(item["step"])
            candidates.append(
                {
                    "step": step,
                    "agent": item.get("agent"),
                    "score": result.get("score", 0.0),
                    "confidence": result.get("confidence", 0.0),
                    "reason": item.get("reason", ""),
                    "source": "chunk",
                    "content": step_content(steps, step),
                }
            )
    return candidates


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_step: dict[int, dict[str, Any]] = {}
    for cand in candidates:
        if cand.get("step") is None:
            continue
        step = int(cand["step"])
        prev = best_by_step.get(step)
        if prev is None or cand.get("score", 0.0) > prev.get("score", 0.0):
            best_by_step[step] = cand
    return list(best_by_step.values())


def context_around(steps: list[LogStep], step_no: int, radius: int = 2) -> list[LogStep]:
    for idx, step in enumerate(steps):
        if step.step == step_no:
            return steps[max(0, idx - radius) : min(len(steps), idx + radius + 1)]
    return []


def step_content(steps: list[LogStep], step_no: int) -> str:
    for step in steps:
        if step.step == step_no:
            return render_steps([step])
    return ""


def default_constraints() -> list[dict[str, str]]:
    return [
        {
            "id": "C1",
            "type": "task",
            "description": "Agents must preserve the user task requirements.",
            "violation_criteria": "An agent changes, ignores, or contradicts a requirement.",
        },
        {
            "id": "C2",
            "type": "evidence",
            "description": "Agents must use verified evidence before drawing conclusions.",
            "violation_criteria": "An agent relies on irrelevant, unverified, or unsupported evidence.",
        },
    ]


def default_checklist_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "T1",
            "category": "task",
            "success_checkpoint": "Preserve and satisfy the requirements stated in the task.",
            "failure_indicator": "An agent changes, ignores, or contradicts a task requirement.",
            "conditional": False,
        },
        {
            "id": "T2",
            "category": "evidence",
            "success_checkpoint": "Use relevant and verified evidence before drawing conclusions.",
            "failure_indicator": "An agent relies on irrelevant, unverified, or unsupported evidence.",
            "conditional": False,
        },
    ]


def parsed_step(parsed: dict[str, Any]) -> int | None:
    for key in ["step", "step_number", "candidate_step", "Step Number", "error_step"]:
        value = parsed.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def parsed_step_matches(parsed: dict[str, Any], expected_step: int) -> bool:
    step = parsed_step(parsed)
    return step is None or step == expected_step


def step_inside_chunk(value: Any, chunk: list[LogStep]) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        step = int(value)
    except (TypeError, ValueError):
        return False
    return step in {item.step for item in chunk}


def parsed_agent(parsed: dict[str, Any]) -> str | None:
    for key in ["agent", "agent_name", "Agent Name", "responsible_agent"]:
        value = normalize_optional_str(parsed.get(key))
        if value and value.strip().upper() not in {"NONE", "NULL", "N/A", "NA"}:
            return value
    return None


def is_human_agent(agent: str | None) -> bool:
    if not agent:
        return False
    return normalize_agent_for_match(agent) in {"human", "user"}


def first_system_step(steps: list[LogStep]) -> LogStep | None:
    for step in steps:
        if not is_human_agent(step.agent):
            return step
    return steps[0] if steps else None


def agents_match(left: str, right: str) -> bool:
    return normalize_agent_for_match(left) == normalize_agent_for_match(right)


def normalize_agent_for_match(agent: str) -> str:
    agent = agent.strip().lower()
    if "(" in agent:
        agent = agent.split("(", 1)[0].strip()
    return agent


def strip_raw_large(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        raw = copy.get("raw_response")
        if isinstance(raw, str) and len(raw) > 1000:
            copy["raw_response"] = raw[:1000] + "...[truncated]"
        stripped.append(copy)
    return stripped


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
