from __future__ import annotations

import re
from typing import Any

from .chunking import render_steps
from .schema import Case, LogStep


def header(case: Case) -> str:
    parts = [
        "Problem:",
        case.problem,
    ]
    if case.ground_truth:
        parts.extend(["", "Ground Truth Answer:", case.ground_truth])
    if case.final_answer:
        parts.extend(["", "Final System Answer:", case.final_answer])
    agents = ", ".join(sorted({s.agent for s in case.steps}))
    parts.extend(["", "Agents:", agents])
    return "\n".join(parts)


def agentrx_header(case: Case) -> str:
    parts = [
        "Problem:",
        case.problem,
    ]
    if case.final_answer:
        parts.extend(["", "Final System Answer:", case.final_answer])
    agents = ", ".join(sorted({s.agent for s in case.steps}))
    parts.extend(["", "Agents:", agents])
    return "\n".join(parts)


def agentrx_tool_agent_inventory(steps: list[LogStep]) -> str:
    agents = sorted({step.agent for step in steps if step.agent})
    return "\n".join(
        [
            f"Observed agents/roles: {', '.join(agents) if agents else 'unknown'}",
            "Formal tool schema: not provided by this benchmark record.",
            "Domain policy: not provided by this benchmark record.",
            "Available evidence format: step-indexed agent messages, tool observations, and final system answer when present.",
        ]
    )


def all_at_once_prompt(case: Case) -> str:
    return f"""You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.

{header(case)}

Conversation:
{render_steps(case.steps)}

Task:
Identify which agent made the decisive error, at which step, and explain the reason for the error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
agent, step, reason, confidence.
step must be the exact step number from the conversation.
"""


def task_only_all_at_once_prompt(case: Case) -> str:
    agents = ", ".join(sorted({step.agent for step in case.steps}))
    return f"""You are an AI assistant tasked with analyzing a failed multi-agent conversation history.

Problem:
{case.problem}

Agents:
{agents}

Conversation:
{render_steps(case.steps)}

Task:
Identify which agent made the decisive error, at which step, and explain the reason for the error.
A decisive error is the earliest system-agent action that, if corrected, could plausibly change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement.
Do not choose a later downstream symptom when an earlier decisive error is available.
Use only the problem and conversation above. No reference answer or gold failure label is available.

Return JSON only with keys:
agent, step, reason, confidence.
step must be the exact global step number shown in the conversation.
"""


def trace_elephant_official_all_at_once_prompt(
    case: Case,
    steps: list[LogStep],
) -> str:
    """Render TraceElephant's official All-at-Once prompt on a compact trace."""
    info_line = f"The Answer for the problem is: {case.ground_truth}\n" if case.ground_truth else ""
    trace_metadata = case.metadata.get("trace_elephant")
    trace_metadata = trace_metadata if isinstance(trace_metadata, dict) else {}
    agent_system_intro = str(
        case.metadata.get("agent_system_intro")
        or trace_metadata.get("agent_system_intro")
        or ""
    ).strip()
    agent_intro_line = ""
    if agent_system_intro:
        agent_intro_line = (
            f"\nAgent System Introduction: {agent_system_intro}\n"
            "This is an introduction to the agent system that generated this trace.\n\n"
        )
    chat_content = "\n".join(
        f"Step {step.step} - {step.agent}: {step.content}"
        for step in steps
    )
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real "
        "world problem.\n"
        f"The problem is: {case.problem}\n"
        f"{info_line}"
        f"{agent_intro_line}"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation (each line is a step, starting from 1):\n\n"
        f"{chat_content}\n\n"
        "Based on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution "
        "to the real world problem. If there are no agents that make obvious mistakes, decide one single agent "
        "in your mind. Directly output the name of the Expert.\n"
        "2. The step number (starting from 1) where the mistake agent first made a mistake. The conversation is "
        "formatted as:\n"
        "Step 1 - agent a: xx\n"
        "Step 2 - agent b: xxxx\n"
        "Step 3 - agent c: xxxxx\n"
        "Step 4 - agent a: xxxxxxx\n"
        "Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer with a JSON block wrapped in triple backticks, in this exact schema:\n"
        "```json\n"
        '{"agent_name": "...", "step_number": 0, "reason": "..."}\n'
        "```\n"
    )


def direct_requirements_full_trace_prompt(
    case: Case,
    requirements: list[dict],
    steps: list[LogStep],
) -> str:
    agents = ", ".join(sorted({step.agent for step in steps}))
    return f"""You are an AI assistant tasked with directly attributing a failed multi-agent conversation.

Problem:
{case.problem}

Agents:
{agents}

Task-success requirements:
{requirements}

Full conversation:
{render_steps(steps)}

Task:
Using the task-success requirements as diagnostic references, identify:
1. the system agent directly responsible for the final failure,
2. the exact step containing that agent's earliest unrecovered decisive error,
3. a brief reason explaining how that error caused the failure.

An earliest unrecovered decisive error is the first system-agent action that remained uncorrected and directly
contributed to the failed result. Do not attribute the failure to the human/user/problem statement. Do not select
a later downstream symptom or an earlier harmless mistake that was subsequently corrected.

Return exactly one valid JSON object. Do not use markdown fences or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "agent": "<agent name>",
  "step": <integer global step number from the conversation>,
  "reason": "<brief explanation>",
  "confidence": <number from 0.0 to 1.0>
}}
"""


WHO_WHEN_PRO_ERROR_TAXONOMY = """P.1 Visual misidentification: Wrong object, entity, or text recognition in visual input.
P.2 Grounding error: Selecting or targeting the wrong spatial region in an image or interface.
R.1 Hallucination: Generating claims not grounded in any retrieved observation.
R.2 Reasoning error: Misapplying correct information, including entity confusion, temporal mix-ups, reversed comparisons, sign errors, missing edge cases, or broken invariants.
R.3 Calculation error: Arithmetic, unit conversion, counting, measurement, off-by-one, or rounding error.
R.4 Task misunderstanding: Building a structurally wrong interpretation of the task or answering a different question.
PL.1 Ineffective planning: Following an unsound high-level plan, decomposition, or strategy.
PL.2 Goal drift: Gradually deviating from the original task objective.
A.1 Tool parameter error: Using the correct tool with wrong arguments, malformed calls, missing parameters, or incomplete action sequences.
A.2 Hallucinated tool or action: Invoking a tool, API, or action that does not exist.
A.3 Output format error: Producing malformed structured output, broken syntax, or a final answer in the wrong format.
A.4 Premature termination: Stopping before the task objectives are fully met.
A.5 Looping behavior: Repeating the same or equivalent actions without progress.
V.1 Context loss: Losing or failing to retrieve relevant information from prior context or observations.
V.2 Inadequate verification: Failing to verify results, verifying incorrectly, accepting conflicts, or overcorrecting a correct answer.
C.1 Delegation and orchestration error: Assigning work to the wrong agent, decomposing improperly, confusing roles, or creating conflicting actions.
C.2 Communication failure: Withholding critical information, ignoring recommendations, or losing shared context across agents.
C.3 Over-reliance on other agents: Replacing a sound independent answer with a less accurate position from another agent."""


def who_when_pro_all_at_once_prompt(case: Case) -> str:
    return f"""# Task
You are an expert at diagnosing failures in agentic systems.
You will be given the transcript of an agentic system attempting to answer a user question. The system failed because of a decisive error somewhere in the transcript. Your job is to identify the first decisive error: the step that most directly causes the system to go wrong and eventually produce an incorrect answer.

Report which agent made that decisive error, the exact step coordinate where it occurred, and the best matching error mode from the taxonomy below. Then briefly explain your reasoning.

## Error Mode Taxonomy
{WHO_WHEN_PRO_ERROR_TAXONOMY}

## User Question
{case.problem}

## Transcript
{render_steps(case.steps)}

## Response Format
Please answer in the following format, exactly:
Agent Name: (the agent ID whose turn first introduces the error)
Step Number: (the step coordinate, exactly as used in the conversation above)
Error Mode: (one of the error modes listed)
Reason: (one or two sentences explaining the error)
"""


def a2p_all_at_once_prompt(case: Case) -> str:
    return f"""You are running an A2P-style failure attribution method.
A2P means Abduct, Act, Predict:
1. Abduct: infer the hidden root cause that explains why a candidate agent action was wrong.
2. Act: define the minimal corrected action the agent should have taken at that step.
3. Predict: predict whether the corrected action would have changed the failed trajectory into a successful one.

{header(case)}

Conversation:
{render_steps(case.steps)}

Task:
Identify the earliest agent action that passes the A2P counterfactual test.
The decisive error is the first system-agent action such that a minimal correction at that step
would plausibly change the final failed trajectory into a successful trajectory.
Do not attribute the failure to the human/user/problem statement.
Do not choose a later downstream symptom if an earlier counterfactual root cause is available.

Return JSON only with keys:
agent, step, abduction, action, prediction, would_fix_failure, causal_score, reason.
step must be the exact step number from the conversation.
would_fix_failure must be true only if the counterfactual prediction supports success.
causal_score must be between 0 and 1 and reflect the strength of the A2P counterfactual link.
"""


def render_steps_a2p_official(steps: list[LogStep], *, global_step_numbers: bool = False) -> str:
    return "\n".join(
        [
            f"Step {step.step if global_step_numbers else idx} - {step.agent}: {step.content}"
            for idx, step in enumerate(steps)
        ]
    )


def a2p_repo_exact_prompt(case: Case) -> str:
    """A2P repository prompt at commit 7953d780, with only case fields substituted."""
    structured_conversation = render_steps_a2p_official(case.steps)
    return (
        "You are an AI assistant performing failure analysis on a multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework. "
        "Multiple agents collaborated to solve a problem but produced an incorrect solution. "
        "Your task is to identify the root cause of failure using structured causal inference.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each agent's potentially problematic action, infer the hidden causal factors:\n"
        "- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?\n"
        "- What latent variables (beliefs, misconceptions, missing information) led to their decision?\n"
        "- Identify the agent whose error represents the most plausible root cause of failure\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the identified critical error:\n"
        "- What specific action should the agent have taken instead?\n"
        "- Define the minimal, concrete intervention that addresses the root cause\n"
        "- How exactly would the correct action differ from what they actually did?\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Test the causal hypothesis by simulating the intervention:\n"
        "- Predict the next 3-5 conversational turns if the correct action had been taken\n"
        "- Would this counterfactual trajectory lead to the correct answer?\n"
        "- Trace the causal chain: intervention → intermediate effects → final success/failure\n\n"
        "FINAL ATTRIBUTION:\n"
        "Based on your A2P analysis, identify:\n"
        "1. The agent whose error is the decisive root cause of failure\n"
        "2. The step number where this critical error first occurred\n"
        "3. A clear explanation of the causal mechanism linking the error to failure\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be "
        "the decisive factor that, if corrected through your defined intervention, would change the outcome from failure to success."
    )


def a2p_official_prompt(case: Case) -> str:
    structured_conversation = render_steps_a2p_official(case.steps)
    return (
        "You are an AI assistant performing failure analysis on a multi-agent conversation using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. "
        "Multiple agents collaborated to solve a problem but produced an incorrect solution. "
        "Your task is to identify the root cause of failure using structured causal inference.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each agent's potentially problematic action, infer the hidden causal factors:\n"
        "- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?\n"
        "- What latent variables (beliefs, misconceptions, missing information) led to their decision?\n"
        "- Identify the agent whose error represents the most plausible root cause of failure\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the identified critical error:\n"
        "- What specific action should the agent have taken instead?\n"
        "- Define the minimal, concrete intervention that addresses the root cause\n"
        "- How exactly would the correct action differ from what they actually did?\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION:\n"
        "Do not choose a later downstream symptom if an earlier counterfactual root cause is available. "
        "Prefer the earliest decisive causal error whose corrective intervention would change the outcome "
        "from failure to success.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The agent whose error is the decisive root cause of failure\n"
        "2. The step number where this critical error first occurred\n"
        "3. A clear explanation of the causal mechanism linking the error to failure\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be "
        "the decisive factor that, if corrected through your defined intervention, would change the outcome from "
        "failure to success."
    )


def a2p_official_chunk_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    structured_conversation = render_steps_a2p_official(chunk, global_step_numbers=True)
    return (
        "You are an AI assistant performing failure analysis on one bounded segment of a long multi-agent "
        "conversation using the A2P (Abduct-Act-Predict) scaffolding framework. "
        "Multiple agents collaborated to solve a problem but produced an incorrect solution. "
        "Your task is to identify whether this segment contains the root cause of failure using structured "
        "causal inference.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        f"TRACE ALLOCATION CONTEXT: The full conversation has {len(case.steps)} steps and is divided into "
        f"{chunk_count} chunks. You are reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-"
        f"{chunk[-1].step}.\n"
        f"Previous Chunk Summary: {prev_summary}\n"
        f"Next Chunk Summary: {next_summary}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each suspicious agent action in this chunk, infer the hidden cause behind the behavior.\n"
        "Examples include knowledge gaps, task misunderstanding, flawed assumptions, or missing information.\n"
        "Identify which agent's error is the most plausible root cause visible in this chunk.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the critical error you found, define the minimal corrective action the agent should have taken "
        "at that exact step. Explain how the actual action differs from the correct action.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION FOR THIS CHUNK:\n"
        "Decide whether this chunk contains the decisive root cause of the final failure.\n"
        "You must select only a step inside this chunk. Do not choose later downstream symptoms.\n"
        "Prefer the earlier counterfactual root cause if one is available.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The decisive root-cause agent inside this chunk.\n"
        "2. The step number where the critical error first occurred inside this chunk.\n"
        "3. The causal mechanism linking that error to the final failure.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Chunk Contains Root Cause: [Yes or No]\n"
        "Agent Name: [Agent name only, or NONE]\n"
        "Step Number: [Number only, or -1]\n"
        "Causal Score: [0.0 to 1.0]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "If this chunk does not contain the root cause, output Chunk Contains Root Cause: No, Agent Name: NONE, "
        "Step Number: -1, and Causal Score: 0.0. The error you identify must be the decisive factor that, "
        "if corrected through your defined intervention, would change the outcome from failure to success."
    )


def a2p_official_reread_prompt(case: Case, chunk: list[LogStep], chunk_candidate: dict) -> str:
    structured_conversation = render_steps_a2p_official(chunk, global_step_numbers=True)
    return (
        "You are an AI assistant rereading a selected segment using the A2P (Abduct-Act-Predict) scaffolding "
        "framework. Multiple agents collaborated to solve a problem but produced an incorrect solution. "
        "Your task is to localize the exact root-cause step inside this selected segment using structured "
        "causal inference.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        "SELECTED SEGMENT NOTE:\n"
        "This segment was selected by a first-pass chunk-ranking stage. Re-evaluate the segment independently "
        "using A2P. Do not assume the first-pass candidate agent, step, score, or reason was correct.\n\n"
        "CONVERSATION HISTORY:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each suspicious agent action in this selected segment, infer the hidden cause behind the behavior.\n"
        "Examples include knowledge gaps, task misunderstanding, flawed assumptions, or missing information.\n"
        "Identify which agent's error is the most plausible root cause visible in this selected segment.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the critical error you found, define the minimal corrective action the agent should have taken "
        "at that exact step. Explain how the actual action differs from the correct action.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION:\n"
        "Decide whether this selected segment contains the decisive root cause of the final failure.\n"
        "You must select only a step inside this selected segment. Do not choose later downstream symptoms.\n"
        "Prefer the earlier counterfactual root cause if one is available.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The decisive root-cause agent inside this selected segment.\n"
        "2. The step number where the critical error first occurred inside this selected segment.\n"
        "3. The causal mechanism linking that error to the final failure.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Segment Contains Root Cause: [Yes or No]\n"
        "Agent Name: [Agent name only, or NONE]\n"
        "Step Number: [Number only, or -1]\n"
        "Causal Score: [0.0 to 1.0]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "If this segment does not contain the root cause, output Segment Contains Root Cause: No, Agent Name: NONE, "
        "Step Number: -1, and Causal Score: 0.0."
    )


def a2p_official_beam_joint_prompt(
    case: Case,
    selected_chunks: list[dict[str, Any]],
) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                content=render_steps_a2p_official(chunk, global_step_numbers=True),
            )
        )
    return (
        "You are an AI assistant rereading selected evidence chunks in temporal order using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. Multiple agents collaborated to solve a problem but "
        "produced an incorrect solution. A previous chunk-ranking stage selected the chunks below. "
        "Your task is to independently identify the earliest decisive root-cause step among the selected chunks. "
        "Do not assume any first-pass candidate agent, step, score, or reason was correct.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        "SELECTED CHUNKS IN TEMPORAL ORDER:\n"
        f"{chr(10).join(rendered_chunks)}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each suspicious agent action in the selected chunks, infer the hidden cause behind the behavior.\n"
        "Examples include knowledge gaps, task misunderstanding, flawed assumptions, or missing information.\n"
        "Identify which agent's error is the most plausible root cause visible in these selected chunks.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the critical error you found, define the minimal corrective action the agent should have taken "
        "at that exact step. Explain how the actual action differs from the correct action.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION FOR SELECTED CHUNKS:\n"
        "Decide whether the selected chunks contain the decisive root cause of the final failure.\n"
        "You must select only a step inside the selected chunks. Do not choose later downstream symptoms.\n"
        "Prefer the earliest counterfactual root cause across the selected chunks if one is available.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The decisive root-cause agent inside the selected chunks.\n"
        "2. The step number where the critical error first occurred inside the selected chunks.\n"
        "3. The causal mechanism linking that error to the final failure.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be "
        "the decisive factor that, if corrected through your defined intervention, would change the outcome from "
        "failure to success."
    )


def a2p_official_global_router_reread_prompt(
    case: Case,
    selected_chunks: list[dict[str, Any]],
) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                content=render_steps_a2p_official(chunk, global_step_numbers=True),
            )
        )
    return (
        "You are an AI assistant rereading selected evidence chunks in temporal order using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. Multiple agents collaborated to solve a problem but "
        "produced an incorrect solution. A previous chunk-ranking stage selected the chunks below. "
        "Your task is to independently identify the earliest decisive root-cause step among the selected chunks.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n"
        "SELECTED CHUNKS IN TEMPORAL ORDER:\n"
        f"{chr(10).join(rendered_chunks)}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each suspicious agent action in the selected chunks, infer the hidden cause behind the behavior.\n"
        "Examples include knowledge gaps, task misunderstanding, flawed assumptions, or missing information.\n"
        "Identify which agent's error is the most plausible root cause visible in these selected chunks.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the critical error you found, define the minimal corrective action the agent should have taken "
        "at that exact step. Explain how the actual action differs from the correct action.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION FOR SELECTED CHUNKS:\n"
        "Decide whether the selected chunks contain the decisive root cause of the final failure.\n"
        "You must select only a step inside the selected chunks. Do not choose later downstream symptoms.\n"
        "Prefer the earliest counterfactual root cause across the selected chunks if one is available.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The decisive root-cause agent inside the selected chunks.\n"
        "2. The step number where the critical error first occurred inside the selected chunks.\n"
        "3. The causal mechanism linking that error to the final failure.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be "
        "the decisive factor that, if corrected through your defined intervention, would change the outcome from "
        "failure to success."
    )


def a2p_official_local_confidence_chunk_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
) -> str:
    structured_conversation = render_steps_a2p_official(chunk, global_step_numbers=True)
    return f"""You are an AI assistant performing failure analysis on one candidate chunk from a long multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.

Your task is NOT to produce the final agent/step attribution.
Your task is to inspect this candidate chunk and assign a confidence score for whether this chunk should be reread with A2P.

PROBLEM: {case.problem}
CORRECT ANSWER: {case.ground_truth or ''}
FINAL SYSTEM ANSWER: {case.final_answer or ''}

Chunk context:
The full conversation is divided into {chunk_count} chunks.
You are reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Candidate chunk conversation:
{structured_conversation}

Chunk selection objective:
- Score how likely this chunk is to contain the earliest decisive causal error.
- Prefer the earliest counterfactual root cause over later downstream symptoms.
- Do not score this chunk highly merely because it contains repeated failures, final symptoms, or confident-looking late consequences.
- Use only the evidence in this chunk, together with the problem, correct answer, and final system answer.

A2P SCAFFOLDING INSTRUCTIONS:
Apply the A2P framework to score this chunk through three sequential steps:

STEP 1 - ABDUCTION (Infer Hidden Root Causes):
For each agent's potentially problematic action in this chunk, infer the hidden causal factors.
For any suspicious agent action in this chunk, infer the hidden cause behind the behavior:
- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?
- What latent variables (beliefs, misconceptions, missing information) led to their decision?
- Identify whether this chunk contains the agent error that represents the most plausible root cause of failure visible from the available evidence.

STEP 2 - ACTION (Define Minimal Corrective Intervention):
For the identified critical error candidate:
- What specific action should the agent have taken instead?
- Define the minimal, concrete intervention that addresses the root cause.
- How exactly would the correct action differ from what they actually did?

STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):
Test the causal hypothesis by simulating the intervention in this chunk:
- Predict the next 3-5 conversational turns if the correct action had been taken.
- Would this counterfactual trajectory lead to the correct answer?
- Trace the causal chain: intervention -> intermediate effects -> final success/failure.

ROUTING DECISION:
Based on your A2P analysis, judge how likely this chunk is to contain:
1. The decisive root-cause agent action
2. The step number where this critical error first occurred
3. A clear causal mechanism linking the error to failure

CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:

Return valid JSON wrapped in <json></json> tags:

<json>
{{
  "chunk_id": {chunk_id},
  "contains_root_cause": true,
  "agent": "agent name only, or NONE",
  "step": 12,
  "confidence": 0.0,
  "reason": "Brief A2P-based reason for the confidence score"
}}
</json>

Rules:
- confidence must be between 0.0 and 1.0.
- confidence means how likely this chunk contains the earliest decisive causal error.
- If this chunk does not contain a plausible root cause, use contains_root_cause=false, agent="NONE", step=-1, and a low confidence score.
- The step must be inside this chunk. Do not choose a step outside this chunk.
- Do not choose a later downstream symptom if an earlier counterfactual root cause is visible in this chunk.

Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be
the decisive factor that, if corrected through your defined intervention, would change the outcome from
failure to success.
"""


def a2p_official_global_chunk_router_prompt(
    case: Case,
    conversation: str,
    chunk_ranges: list[dict[str, Any]],
    beam_k: int,
) -> str:
    chunk_lines: list[str] = []
    for item in chunk_ranges:
        estimated_tokens = item.get("estimated_tokens")
        token_note = f", estimated_tokens={estimated_tokens}" if estimated_tokens is not None else ""
        chunk_lines.append(
            "Chunk {chunk_id}: steps {start_step}-{end_step}, step_count={step_count}{token_note}".format(
                chunk_id=item.get("chunk_id"),
                start_step=item.get("start_step"),
                end_step=item.get("end_step"),
                step_count=item.get("step_count"),
                token_note=token_note,
            )
        )
    chunk_table = "\n".join(chunk_lines)
    return f"""You are an AI assistant performing failure analysis on a long multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.

Your task is NOT to produce the final agent/step attribution.
Your task is to read the whole trace once, inspect the chunk table, and select the top {beam_k} chunk IDs that should be reread with A2P.

PROBLEM: {case.problem}
CORRECT ANSWER: {case.ground_truth or ''}
FINAL SYSTEM ANSWER: {case.final_answer or ''}

Chunk table:
{chunk_table}

Full conversation:
{conversation}

Chunk selection objective:
- Select the top {beam_k} chunk IDs most likely to contain the earliest decisive causal error.
- Prefer the earliest counterfactual root cause over later downstream symptoms.
- Do not select chunks merely because they contain repeated failures, final symptoms, or confident-looking late consequences.
- Use only chunk IDs from the table.

A2P SCAFFOLDING INSTRUCTIONS:
Apply the A2P framework to select the chunks through three sequential steps:

STEP 1 - ABDUCTION (Infer Hidden Root Causes):
For each agent's potentially problematic action, infer the hidden causal factors.
For each chunk that may contain a suspicious agent action, infer the hidden cause behind the behavior:
- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?
- What latent variables (beliefs, misconceptions, missing information) led to their decision?
- Identify which chunk contains the agent error that represents the most plausible root cause of failure visible in the full trace.

STEP 2 - ACTION (Define Minimal Corrective Intervention):
For the identified critical error candidate:
- What specific action should the agent have taken instead?
- Define the minimal, concrete intervention that addresses the root cause.
- How exactly would the correct action differ from what they actually did?

STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):
Test the causal hypothesis by simulating the intervention in the candidate chunk:
- Predict the next 3-5 conversational turns if the correct action had been taken.
- Would this counterfactual trajectory lead to the correct answer?
- Trace the causal chain: intervention → intermediate effects → final success/failure

ROUTING DECISION:
Based on your A2P analysis, select the chunk IDs most likely to contain:
1. The decisive root-cause agent action
2. The step number where this critical error first occurred
3. A clear causal mechanism linking the error to failure

CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:

Return valid JSON wrapped in <json></json> tags:

<json>
{{
  "selected_chunk_ids": [1, 2],
  "rationale": "Brief A2P-based reason for selecting these chunks"
}}
</json>

Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be
the decisive factor that, if corrected through your defined intervention, would change the outcome from
failure to success.
"""


def a2p_official_agent_joint_prompt(
    case: Case,
    target_agents: list[str],
    selected_chunks: list[dict[str, Any]],
) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end} | first-pass score {score}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                score=item.get("score"),
                content=render_steps_a2p_official(chunk, global_step_numbers=True),
            )
        )
    target_agent_text = "\n".join([f"{idx}. {agent}" for idx, agent in enumerate(target_agents, 1)])
    return (
        "You are an AI assistant rereading selected evidence chunks in temporal order using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. Multiple agents collaborated to solve a problem but "
        "produced an incorrect solution. A previous chunk-ranking stage selected the chunks below and "
        "predicted the target agents listed below. Your task is to identify the earliest decisive root-cause "
        "step among the selected chunks, restricted to those target agents unless no target-agent step passes "
        "the A2P counterfactual test.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        f"TARGET AGENTS IN RANK ORDER:\n{target_agent_text}\n\n"
        "SELECTED CHUNKS IN TEMPORAL ORDER:\n"
        f"{chr(10).join(rendered_chunks)}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each suspicious target-agent action in the selected chunks, infer the hidden cause behind the behavior.\n"
        "Examples include knowledge gaps, task misunderstanding, flawed assumptions, or missing information.\n"
        "Identify which target agent's error is the most plausible root cause visible in these selected chunks.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the critical error you found, define the minimal corrective action the agent should have taken "
        "at that exact step. Explain how the actual action differs from the correct action.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Assume the corrective action happened. Predict how the next 3-5 conversational turns would change.\n"
        "Decide whether that counterfactual trajectory would lead to the correct answer.\n"
        "Trace the causal chain: intervention -> intermediate effects -> final success or failure.\n\n"
        "FINAL ATTRIBUTION FOR SELECTED CHUNKS:\n"
        "Decide whether the selected chunks contain the decisive root cause of the final failure.\n"
        "You must select only a step inside the selected chunks. Do not choose later downstream symptoms.\n"
        "Prefer the earliest counterfactual root cause across the selected chunks if one is available.\n"
        "Based on the A2P analysis, identify:\n"
        "1. The decisive root-cause target agent inside the selected chunks.\n"
        "2. The step number where the critical error first occurred inside the selected chunks.\n"
        "3. The causal mechanism linking that error to the final failure.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Segment Contains Root Cause: [Yes or No]\n"
        "Agent Name: [Agent name only, or NONE]\n"
        "Step Number: [Number only, or -1]\n"
        "Causal Score: [0.0 to 1.0]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "If no selected chunk contains a target-agent root cause, output Segment Contains Root Cause: No, "
        "Agent Name: NONE, Step Number: -1, and Causal Score: 0.0."
    )


def a2p_official_rerank_prompt(case: Case, candidates: list[dict]) -> str:
    return (
        "You are an AI assistant choosing among candidate failure attributions using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. Multiple agents collaborated to solve a problem but "
        "produced an incorrect solution. Your task is to choose the candidate that best identifies the "
        "decisive root cause of failure.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        f"CANDIDATE ATTRIBUTIONS:\n{candidates}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "STEP 1 - ABDUCTION: Does the candidate provide a plausible hidden root cause behind the agent's action?\n"
        "STEP 2 - ACTION: Does the candidate define a minimal corrected action the agent should have taken?\n"
        "STEP 3 - PREDICTION: Would that corrected action plausibly change the failed trajectory into a successful one?\n\n"
        "FINAL ATTRIBUTION:\n"
        "Select the candidate whose agent and step most strongly satisfy the A2P causal chain. Prefer the "
        "earliest decisive root cause over later downstream symptoms.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Candidate ID: [Number only]\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Causal Score: [0.0 to 1.0]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]"
    )


def a2p_official_agent_step_prompt(
    case: Case,
    target_agent: str,
    history: list[LogStep],
    current_step: LogStep,
) -> str:
    structured_conversation = render_steps_a2p_official(history, global_step_numbers=True)
    return (
        "You are an AI assistant performing agent-first failure analysis using the A2P "
        "(Abduct-Act-Predict) scaffolding framework. Multiple agents collaborated to solve a problem but "
        "produced an incorrect solution.\n\n"
        f"PROBLEM: {case.problem}\n"
        f"CORRECT ANSWER: {case.ground_truth or ''}\n\n"
        f"TARGET AGENT: {target_agent}\n"
        f"CANDIDATE STEP UNDER REVIEW: Step {current_step.step} - {current_step.agent}\n\n"
        "CONVERSATION HISTORY UP TO THE CANDIDATE STEP:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to decide whether the candidate step is the decisive root cause:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "Infer the hidden causal factors behind the candidate action.\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "Define the minimal corrected action the target agent should have taken at this step.\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Predict whether the corrected action would change the final failed trajectory into success.\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Candidate Is Root Cause: [Yes or No]\n"
        "Agent Name: [Agent name only, or NONE]\n"
        "Step Number: [Number only, or -1]\n"
        "Causal Score: [0.0 to 1.0]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 100 words]\n\n"
        "Only output Yes if this exact candidate step by the target agent is the decisive factor that, if corrected, "
        "would change the outcome from failure to success. Otherwise output No, Agent Name: NONE, Step Number: -1, "
        "and Causal Score: 0.0."
    )


def a2p_chunk_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are running an A2P-style failure attribution method over an adaptive chunk of a long failed multi-agent trace.
A2P means Abduct, Act, Predict:
1. Abduct: infer the hidden root cause that explains why a candidate action in this chunk was wrong.
2. Act: define the minimal corrected action the agent should have taken at that step.
3. Predict: predict whether the corrected action would have changed the final failed trajectory into a successful one.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} adaptive chunks.
You are reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Decide whether this chunk contains the earliest decisive error under the A2P counterfactual test.
Only choose a step from the Current Chunk.
Do not attribute the failure to the human/user/problem statement.
Do not choose downstream symptoms, final-answer mistakes, or harmless failed attempts that were later corrected.
Prefer the earliest action whose minimal correction would plausibly change the final failed trajectory into success.

Return JSON only with keys:
chunk_id, contains_counterfactual_error, agent, step, abduction, action, prediction,
would_fix_failure, causal_score, reason.
step must be an exact step number from the Current Chunk, or null if this chunk is unlikely to contain the decisive error.
would_fix_failure must be true only if the counterfactual prediction supports success.
causal_score must be between 0 and 1 and reflect the strength of the A2P counterfactual link.
"""


def a2p_chunk_context_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    before_context: list[LogStep],
    after_context: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are running an A2P-style failure attribution method over a focal chunk of a long failed multi-agent trace.
A2P means Abduct, Act, Predict:
1. Abduct: infer the hidden root cause that explains why a candidate action in the focal chunk was wrong.
2. Act: define the minimal corrected action the agent should have taken at that step.
3. Predict: predict whether that correction would have changed the final failed trajectory into a successful one.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} adaptive chunks.
You are reviewing Focal Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Immediate Previous Context:
{render_steps(before_context) if before_context else "None"}

Focal Chunk:
{render_steps(chunk)}

Immediate Next Context:
{render_steps(after_context) if after_context else "None"}

Next Chunk Summary:
{next_summary}

Task:
Decide whether the Focal Chunk contains the earliest decisive error under the A2P counterfactual test.
Use previous and next context only to understand causality and distinguish root causes from downstream symptoms.
Only choose a step from the Focal Chunk.
Do not attribute the failure to the human/user/problem statement.
Do not choose downstream symptoms, final-answer mistakes, or harmless failed attempts that were later corrected.
Prefer the earliest action whose minimal correction would plausibly change the final failed trajectory into success.

Return JSON only with keys:
chunk_id, contains_counterfactual_error, agent, step, abduction, action, prediction,
would_fix_failure, causal_score, reason.
step must be an exact step number from the Focal Chunk, or null if the Focal Chunk is unlikely to contain the decisive error.
would_fix_failure must be true only if the counterfactual prediction supports success.
causal_score must be between 0 and 1 and reflect the strength of the A2P counterfactual link.
"""


def a2p_step_prompt(case: Case, chunk: list[LogStep], chunk_candidate: dict) -> str:
    return f"""You are localizing a suspected chunk using the A2P counterfactual failure attribution method.
A2P means Abduct, Act, Predict:
1. Abduct: infer the hidden root cause behind the candidate wrong action.
2. Act: define the minimal corrected action at that exact step.
3. Predict: predict whether that correction would change the final failed trajectory into success.

{header(case)}

First-pass chunk candidate:
{chunk_candidate}

Suspected Chunk:
Steps {chunk[0].step}-{chunk[-1].step}

Chunk Log:
{render_steps(chunk)}

Task:
Find the exact earliest step inside this chunk that passes the A2P counterfactual test.
Do not choose a later symptom if an earlier counterfactual root cause is visible.
Do not attribute the failure to the human/user/problem statement.
If no step inside this chunk passes the counterfactual test, return step null and would_fix_failure false.

Return JSON only with keys:
agent, step, abduction, action, prediction, would_fix_failure, causal_score, reason.
step must be an exact step number from the Suspected Chunk, or null.
causal_score must be between 0 and 1.
"""


def a2p_rerank_prompt(case: Case, candidates: list[dict]) -> str:
    return f"""You are choosing among A2P counterfactual failure candidates from a failed multi-agent trace.

{header(case)}

Candidate Failure Steps:
{candidates}

Task:
Select the candidate that most strongly passes the A2P test:
Abduct a plausible hidden cause, define a minimal corrected action, and predict that this correction
would change the final failed trajectory into success.
Prefer the earliest counterfactual root cause over a later downstream symptom.
Do not attribute the failure to the human/user/problem statement.

Return JSON only with keys:
candidate_id, step, agent, causal_score, reason.
step must be the exact step number from the selected candidate.
causal_score must be between 0 and 1.
"""


def a2p_scaffold_all_at_once_prompt(case: Case) -> str:
    return f"""You are performing failure analysis on a multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
Your task is to identify the root cause of failure using structured causal inference.

PROBLEM:
{case.problem}

CORRECT ANSWER:
{case.ground_truth or "Unknown"}

FINAL SYSTEM ANSWER:
{case.final_answer or "Unknown"}

CONVERSATION HISTORY:
{render_steps(case.steps)}

A2P SCAFFOLDING INSTRUCTIONS:
Apply the A2P framework to identify the critical error through three sequential steps:

STEP 1 - ABDUCTION (Infer Hidden Root Causes):
For each agent's potentially problematic action, infer the hidden causal factors:
- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?
- What latent variables (beliefs, misconceptions, missing information) led to their decision?
- Identify the agent whose error represents the most plausible root cause of failure.

STEP 2 - ACTION (Define Minimal Corrective Intervention):
For the identified critical error:
- What specific action should the agent have taken instead?
- Define the minimal, concrete intervention that addresses the root cause.
- How exactly would the correct action differ from what they actually did?

STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):
Test the causal hypothesis by simulating the intervention:
- Predict the next 3-5 conversational turns if the correct action had been taken.
- Would this counterfactual trajectory lead to the correct answer?
- Trace the causal chain: intervention -> intermediate effects -> final success/failure.

FINAL ATTRIBUTION:
Based on your A2P analysis, identify:
1. The agent whose error is the decisive root cause of failure.
2. The step number where this critical error first occurred.
3. A clear explanation of the causal mechanism linking the error to failure.

Rules:
- Do not attribute the failure to the human/user/problem statement.
- Prefer the earliest decisive root cause over later downstream symptoms.
- The error you identify must be the decisive factor that, if corrected through your defined intervention, would change the outcome from failure to success.

Return JSON only with keys:
agent, step, abduction, action, prediction, would_fix_failure, causal_score, causal_mechanism, reason.
step must be the exact step number from the conversation.
causal_score must be between 0 and 1.
"""


def a2p_scaffold_chunk_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are performing failure analysis on a bounded segment of a long multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
Your task is to identify whether this segment contains the root cause of failure using structured causal inference.

PROBLEM:
{case.problem}

CORRECT ANSWER:
{case.ground_truth or "Unknown"}

FINAL SYSTEM ANSWER:
{case.final_answer or "Unknown"}

TRACE ALLOCATION CONTEXT:
The full conversation has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

CONVERSATION HISTORY:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

A2P SCAFFOLDING INSTRUCTIONS:
Apply the A2P framework to identify the critical error through three sequential steps:

STEP 1 - ABDUCTION (Infer Hidden Root Causes):
For each agent's potentially problematic action in the conversation history above, infer the hidden causal factors:
- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?
- What latent variables (beliefs, misconceptions, missing information) led to their decision?
- Identify the agent whose error represents the most plausible root cause of failure.

STEP 2 - ACTION (Define Minimal Corrective Intervention):
For the identified critical error:
- What specific action should the agent have taken instead?
- Define the minimal, concrete intervention that addresses the root cause.
- How exactly would the correct action differ from what they actually did?

STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):
Test the causal hypothesis by simulating the intervention:
- Predict the next 3-5 conversational turns if the correct action had been taken.
- Would this counterfactual trajectory lead to the correct answer?
- Trace the causal chain: intervention -> intermediate effects -> final success/failure.

FINAL ATTRIBUTION FOR THIS CHUNK:
Based on your A2P analysis, decide whether this chunk contains the earliest decisive root cause visible in this chunk:
1. The agent whose error is the decisive root cause of failure.
2. The step number where this critical error first occurred.
3. A clear explanation of the causal mechanism linking the error to failure.

Rules:
- Only choose a step from the CONVERSATION HISTORY of this chunk.
- Do not attribute the failure to the human/user/problem statement.
- Do not score downstream symptoms highly.
- Prefer an earlier decisive root cause over a later symptom.
- The error you identify must be the decisive factor that, if corrected through your defined intervention, would change the outcome from failure to success.
- If this chunk does not contain such an error, set contains_counterfactual_error=false, agent=null, step=null, would_fix_failure=false, and causal_score=0.

Return JSON only with keys:
chunk_id, contains_counterfactual_error, agent, step, abduction, action, prediction,
would_fix_failure, causal_score, causal_mechanism, reason.
step must be an exact step number from this chunk, or null.
causal_score must be between 0 and 1.
"""


def a2p_scaffold_reread_prompt(case: Case, chunk: list[LogStep], chunk_candidate: dict) -> str:
    return f"""You are rereading a selected candidate segment using the original A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
Your task is to localize the exact root-cause step inside this selected segment using structured causal inference.

PROBLEM:
{case.problem}

CORRECT ANSWER:
{case.ground_truth or "Unknown"}

FINAL SYSTEM ANSWER:
{case.final_answer or "Unknown"}

FIRST-PASS CHUNK CANDIDATE:
{chunk_candidate}

CONVERSATION HISTORY:
{render_steps(chunk)}

A2P SCAFFOLDING INSTRUCTIONS:
Apply the A2P framework to identify the critical error through three sequential steps:

STEP 1 - ABDUCTION (Infer Hidden Root Causes):
For each agent's potentially problematic action in the conversation history above, infer the hidden causal factors:
- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?
- What latent variables (beliefs, misconceptions, missing information) led to their decision?
- Identify the agent whose error represents the most plausible root cause of failure.

STEP 2 - ACTION (Define Minimal Corrective Intervention):
For the identified critical error:
- What specific action should the agent have taken instead?
- Define the minimal, concrete intervention that addresses the root cause.
- How exactly would the correct action differ from what they actually did?

STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):
Test the causal hypothesis by simulating the intervention:
- Predict the next 3-5 conversational turns if the correct action had been taken.
- Would this counterfactual trajectory lead to the correct answer?
- Trace the causal chain: intervention -> intermediate effects -> final success/failure.

FINAL ATTRIBUTION:
Based on your A2P analysis, identify:
1. The agent whose error is the decisive root cause of failure.
2. The step number where this critical error first occurred.
3. A clear explanation of the causal mechanism linking the error to failure.

Rules:
- Only choose a step from the CONVERSATION HISTORY above.
- Do not attribute the failure to the human/user/problem statement.
- Prefer the earliest decisive root cause over later downstream symptoms.
- If no step in this segment passes the A2P counterfactual test, set agent=null, step=null, would_fix_failure=false, and causal_score=0.

Return JSON only with keys:
agent, step, abduction, action, prediction, would_fix_failure, causal_score, causal_mechanism, reason.
step must be an exact step number from the selected segment, or null.
causal_score must be between 0 and 1.
"""


def a2p_scaffold_rerank_prompt(case: Case, candidates: list[dict]) -> str:
    return f"""You are choosing among candidate failure attributions using the A2P (Abduct-Act-Predict) scaffolding framework.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
Your task is to choose the candidate that best identifies the decisive root cause of failure.

PROBLEM:
{case.problem}

CORRECT ANSWER:
{case.ground_truth or "Unknown"}

FINAL SYSTEM ANSWER:
{case.final_answer or "Unknown"}

CANDIDATE ATTRIBUTIONS:
{candidates}

A2P SCAFFOLDING INSTRUCTIONS:
For each candidate, evaluate:

STEP 1 - ABDUCTION:
Does the candidate provide a plausible hidden root cause behind the agent's action?

STEP 2 - ACTION:
Does the candidate define a minimal corrected action the agent should have taken?

STEP 3 - PREDICTION:
Would that corrected action plausibly change the failed trajectory into a successful one?

FINAL ATTRIBUTION:
Select the candidate whose agent and step most strongly satisfy the A2P causal chain.

Rules:
- Prefer the earliest decisive root cause over later downstream symptoms.
- Do not choose an agent/user/problem statement.
- If multiple candidates are plausible, choose the earliest one whose intervention would change failure into success.

Return JSON only with keys:
candidate_id, step, agent, causal_score, reason.
"""


def chunk_all_at_once_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are an AI assistant tasked with analyzing part of a failed multi-agent conversation history.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Conversation:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Identify whether this chunk contains the decisive error, which agent made it, and at which step.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Prefer root causes over downstream symptoms.

Return JSON only with keys:
likely_contains_decisive_error, agent, step, reason, confidence.
step must be the exact step number from the conversation, or null if this chunk is unlikely to contain the decisive error.
"""


def chunk_all_at_once_context_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    before_context: list[LogStep],
    after_context: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are an AI assistant tasked with analyzing part of a failed multi-agent conversation history.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Focal Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Immediate Previous Context:
{render_steps(before_context) if before_context else "None"}

Focal Chunk:
{render_steps(chunk)}

Immediate Next Context:
{render_steps(after_context) if after_context else "None"}

Next Chunk Summary:
{next_summary}

Task:
Identify whether this chunk contains the decisive error, which agent made it, and at which step.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Use the previous and next context only to understand causality and distinguish root causes from symptoms.
Do not choose a step from the previous or next context as the answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Prefer root causes over downstream symptoms.

Return JSON only with keys:
likely_contains_decisive_error, agent, step, reason, confidence.
step must be the exact step number from the Focal Chunk, or null if the Focal Chunk is unlikely to contain the decisive error.
"""


def who_when_beam_joint_prompt(case: Case, selected_chunks: list[dict[str, Any]]) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            f"SELECTED CHUNK {item['chunk_id']} (steps {chunk[0].step}-{chunk[-1].step}):\n"
            f"{render_steps(chunk)}"
        )
    return f"""You are an AI assistant tasked with analyzing selected evidence from a failed multi-agent conversation.

{header(case)}

The full log was adaptively divided into chunks. A first-pass chunk-ranking stage selected the chunks below.
You must reread the selected chunks in temporal order and identify the earliest decisive error visible in them.

Selected Evidence Chunks:
{chr(10).join(rendered_chunks)}

Task:
Identify which agent made the decisive error, at which step, and explain the reason for the error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Prefer the earliest root cause over later downstream symptoms.
Use only steps inside the selected chunks.

Return JSON only with keys:
contains_decisive_error, agent, step, reason, confidence.
step must be the exact step number from the selected chunks, or null if the selected chunks do not contain the decisive error.
confidence must be between 0 and 1.
"""


def render_who_when_official_steps(steps: list[LogStep], *, with_step_numbers: bool = False) -> str:
    if with_step_numbers:
        return "\n".join(f"Step {step.step} - {step.agent}: {step.content}" for step in steps)
    return "\n".join(f"{step.agent}: {step.content}" for step in steps)


def who_when_official_all_at_once_prompt(case: Case) -> str:
    conversation = render_who_when_official_steps(case.steps)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a "
        "real world problem.\n"
        f"The problem is: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n"
        f"{conversation}\n\n"
        "Based on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution "
        "to the real world problem. If there are no agents that make obvious mistakes, decide one single agent "
        "in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as "
        "follows: "
        """ { "agent a": "xx", "agent b": "xxxx", "agent c": "xxxxx", "agent a": "xxxxxxx" }, """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each "
        "agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' "
        "contains the mistake, the step number is 3, and so on. Please determine the step number where the first "
        "mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n "
        "Reason for Mistake: \n"
    )


def who_when_official_global_chunk_router_prompt(
    case: Case,
    conversation: str,
    chunk_ranges: list[dict[str, Any]],
    beam_k: int,
) -> str:
    chunk_lines: list[str] = []
    for item in chunk_ranges:
        estimated_tokens = item.get("estimated_tokens")
        token_note = f", estimated_tokens={estimated_tokens}" if estimated_tokens is not None else ""
        chunk_lines.append(
            "Chunk {chunk_id}: steps {start_step}-{end_step}, step_count={step_count}{token_note}".format(
                chunk_id=item.get("chunk_id"),
                start_step=item.get("start_step"),
                end_step=item.get("end_step"),
                step_count=item.get("step_count"),
                token_note=token_note,
            )
        )
    chunk_table = "\n".join(chunk_lines)
    return (
        "You are an AI assistant tasked with analyzing a long multi-agent conversation history when solving a "
        "real world problem.\n"
        "Multiple agents collaborated to address the problem but produced an incorrect solution.\n\n"
        f"The problem is: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n\n"
        "WHO&WHEN GLOBAL CHUNK ROUTING:\n"
        "Your task is not to produce the final agent or step attribution. Read the full conversation once, "
        f"inspect the chunk table, and select exactly the top {beam_k} chunk IDs that should be reread for final "
        "Who&When attribution.\n\n"
        f"Chunk table:\n{chunk_table}\n\n"
        f"Full conversation:\n{conversation}\n\n"
        "Chunk selection objective:\n"
        f"- Select exactly {beam_k} unique chunk IDs most likely to contain the single critical mistake that "
        "directly contributed to the failure in resolving the user's query.\n"
        "- Prefer the earliest responsible mistake over later downstream symptoms or repeated consequences.\n"
        "- Do not select a chunk merely because it contains the final answer, a conspicuous late failure, or a "
        "confident-looking downstream statement.\n"
        "- If no single clear error is evident, use your best judgment to select the chunks containing the steps "
        "most responsible for the failure.\n"
        "- Use only chunk IDs listed in the chunk table.\n"
        "- Do not output an Agent Name, Step Number, confidence score, or final attribution.\n\n"
        "Return valid JSON wrapped in <json></json> tags and use exactly this schema:\n\n"
        "<json>\n"
        "{\n"
        '  "selected_chunk_ids": [1, 2],\n'
        '  "rationale": "Brief reason for selecting these chunks"\n'
        "}\n"
        "</json>\n"
    )


def who_when_official_step_by_step_prompt(case: Case, history: list[LogStep], current_step: LogStep) -> str:
    conversation = render_who_when_official_steps(history, with_step_numbers=True)
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent "
        "conversation aimed at solving a real-world problem. "
        f"The problem being addressed is: {case.problem}. "
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        f"Here is the conversation history up to the current step:\n{conversation}\n"
        f"The most recent step ({current_step.step}) was by '{current_step.agent}'.\n"
        "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that "
        "could hinder the problem-solving process or lead to an incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the "
        "process."
        "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )


def who_when_official_binary_search_prompt(
    case: Case,
    segment: list[LogStep],
    upper_half: list[LogStep],
    lower_half: list[LogStep],
) -> str:
    conversation = render_who_when_official_steps(segment)
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents "
        "are collaborating to address a user query, with the goal of resolving the query through their collective "
        "dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. "
        "Determine which half of the segment contains the single step where this crucial error occurs, ultimately "
        "leading to the failure in resolving the user's query.\n"
        f"The problem to address is as follows: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        f"Review the following conversation segment from step {segment[0].step} to step {segment[-1].step}:\n\n"
        f"{conversation}\n\n"
        "Based on your analysis, predict whether the most critical error is more likely to be located in the "
        f"upper half (from step {upper_half[0].step} to step {upper_half[-1].step}) or the lower half (from step "
        f"{lower_half[0].step} to step {lower_half[-1].step}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your "
        "answer should be based on identifying the mistake that directly contributes to the failure in resolving "
        "the user's query. If no single clear error is evident, consider the step you believe is most responsible "
        "for the failure, allowing for subjective judgment, and base your answer on that."
    )


def who_when_official_chunk_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    conversation = render_who_when_official_steps(chunk, with_step_numbers=True)
    return (
        "You are an AI assistant tasked with analyzing one segment of a long multi-agent conversation history "
        "when solving a real world problem.\n"
        f"The problem is: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        f"The full conversation has {len(case.steps)} steps and is divided into {chunk_count} chunks. "
        f"You are reviewing chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.\n"
        f"Previous chunk summary: {prev_summary}\n"
        f"Next chunk summary: {next_summary}\n"
        "Identify whether this chunk contains the error that is directly responsible for the wrong solution, "
        "which agent made it, at which step, and why. Prefer the earliest responsible mistake over later "
        "downstream symptoms. Choose only a step inside this chunk.\n"
        "Here's the conversation segment:\n\n"
        f"{conversation}\n\n"
        "Please answer in the format:\n"
        "Chunk Contains Responsible Error: Yes/No\n"
        "Agent Name: (Your prediction, or NONE)\n"
        "Step Number: (Your prediction, or -1)\n"
        "Causal Score: (0.0 to 1.0)\n"
        "Reason for Mistake: (Your reason)\n"
    )


def who_when_official_beam_joint_prompt(case: Case, selected_chunks: list[dict[str, Any]]) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                content=render_who_when_official_steps(chunk, with_step_numbers=True),
            )
        )
    return (
        "You are an AI assistant tasked with analyzing selected evidence from a failed multi-agent conversation "
        "history when solving a real world problem.\n"
        f"The problem is: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        "A previous chunk-ranking stage selected the following chunks from the full conversation. Reread them in "
        "temporal order and identify which agent made the earliest error that should be directly responsible for "
        "the wrong solution, at which step, and why. Do not choose a later downstream symptom if an earlier "
        "responsible mistake is visible. Use only steps inside the selected chunks.\n"
        "Selected conversation chunks:\n\n"
        f"{chr(10).join(rendered_chunks)}\n\n"
        "Please answer in the format:\n"
        "Agent Name: (Your prediction)\n"
        "Step Number: (Your prediction)\n"
        "Reason for Mistake: (Your reason)\n"
    )


def who_when_official_global_router_joint_prompt(
    case: Case,
    selected_chunks: list[dict[str, Any]],
) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                content=render_who_when_official_steps(chunk, with_step_numbers=True),
            )
        )
    return (
        "You are an AI assistant tasked with analyzing selected evidence from a failed multi-agent conversation "
        "history when solving a real world problem.\n"
        f"The problem is: {case.problem}\n"
        f"The Answer for the problem is: {case.ground_truth or ''}\n"
        "A previous chunk-routing stage selected the following chunks from the full conversation. Reread them "
        "together in temporal order and produce the final Who&When attribution.\n\n"
        "Attribution objective:\n"
        "- Identify the single critical mistake that directly contributed to the failure in resolving the "
        "user's query, including the responsible agent and the step where that mistake first occurred.\n"
        "- Prefer the earliest responsible mistake over later downstream symptoms or repeated consequences.\n"
        "- Do not select a step merely because it contains the final answer, a conspicuous late failure, or a "
        "confident-looking downstream statement.\n"
        "- If no single clear error is evident, use your best judgment to select the step most responsible for "
        "the failure.\n"
        "- Use only steps contained in the selected chunks below.\n"
        "- Do not output chunk IDs or a confidence score. Output only the final attribution fields requested "
        "below.\n\n"
        "Selected conversation chunks:\n\n"
        f"{chr(10).join(rendered_chunks)}\n\n"
        "Please answer in the format:\n"
        "Agent Name: (Your prediction)\n"
        "Step Number: (Your prediction)\n"
        "Reason for Mistake: (Your reason)\n"
    )


def chunk_bool_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are an AI assistant tasked with analyzing part of a failed multi-agent conversation history.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Conversation:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Identify whether this chunk contains the decisive error, which agent made it, and at which step.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Prefer root causes over downstream symptoms.
Do not output confidence, likelihood scores, or any numeric scoring field.

Return JSON only with keys:
likely_contains_decisive_error, agent, step, reason.
step must be the exact step number from the conversation, or null if this chunk is unlikely to contain the decisive error.
"""


def chunk_ordinal_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are given a chunk from a multi-agent execution trace.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Your task is to identify whether this chunk contains evidence of the earliest decisive error
that caused the final task failure.

Do not score downstream symptoms highly.
Do not score harmless failed attempts highly if they were later corrected.
Prefer the earliest causal mistake over later consequences.
Do not output confidence.
Do not output a separate boolean.

Return exactly one JSON object:
{{
  "agent": "<agent name or NONE>",
  "step": <integer or null>,
  "blame_score": <0, 1, 2, 3, or 4>
}}

Scoring rubric:
0 = no evidence of a decisive error
1 = weak clue only
2 = local error or symptom, but not clearly decisive
3 = likely decisive error
4 = clearly decisive causal error with identifiable agent and step

Rules:
- If blame_score is 0 or 1, use agent = "NONE" and step = null.
- Use only one numeric score.
- Do not output confidence.
- Do not output a separate boolean.
"""


def chunk_ordinal_reread_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    first_pass: dict,
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are rereading a high-scoring chunk from a failed multi-agent execution trace.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are rereading Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

First-pass candidate from this chunk:
{first_pass}

Previous Chunk Summary:
{prev_summary}

Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Choose the exact agent and step inside this chunk that best explains the earliest decisive error.
Do not choose a later downstream symptom if an earlier causal mistake is visible.
Do not score harmless failed attempts highly if they were later corrected.
If this reread shows that the chunk is not actually causal, lower the blame_score.

Return exactly one JSON object:
{{
  "agent": "<agent name or NONE>",
  "step": <integer or null>,
  "blame_score": <0, 1, 2, 3, or 4>
}}

Scoring rubric:
0 = no evidence of a decisive error
1 = weak clue only
2 = local error or symptom, but not clearly decisive
3 = likely decisive error
4 = clearly decisive causal error with identifiable agent and step

Rules:
- If blame_score is 0 or 1, use agent = "NONE" and step = null.
- Use only one numeric score.
- Do not output confidence.
- Do not output a separate boolean.
"""


def step_by_step_prompt(case: Case, history: list[LogStep], current_step: LogStep) -> str:
    return f"""You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.

{header(case)}

History Up To Current Step:
{render_steps(history)}

Most Recent Step Under Review:
Step {current_step.step}
Agent: {current_step.agent}

Task:
Determine whether the most recent agent's action contains the earliest decisive error that could hinder the problem-solving process.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Avoid being overly critical; choose yes only for a root-cause error, not a later symptom.

Return JSON only with keys:
contains_error, agent, step, reason, confidence.
"""


def binary_search_prompt(
    case: Case,
    segment: list[LogStep],
    earlier_half: list[LogStep],
    later_half: list[LogStep],
) -> str:
    return f"""You are an AI assistant tasked with analyzing a segment of a multi-agent conversation.
Multiple agents are collaborating to address a user query, but the final trajectory failed.

{header(case)}

Segment Under Review:
Steps {segment[0].step}-{segment[-1].step}
{render_steps(segment)}

Earlier Half:
Steps {earlier_half[0].step}-{earlier_half[-1].step}

Later Half:
Steps {later_half[0].step}-{later_half[-1].step}

Task:
Choose whether the earliest decisive error is more likely located in the earlier half or the later half of the segment.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
half, reason, confidence.
half must be exactly "earlier" or "later".
"""


def paper_hybrid_step_prompt(case: Case, target_agent: str, history: list[LogStep], current_step: LogStep) -> str:
    return f"""You are running the hybrid failure attribution method.
The responsible agent has already been predicted by an all-at-once judge.

{header(case)}

Predicted Responsible Agent:
{target_agent}

History Up To Current Candidate Step:
{render_steps(history)}

Candidate Step Under Review:
Step {current_step.step}
Agent: {current_step.agent}

Task:
Determine whether this candidate action by the predicted responsible agent is the earliest decisive error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Avoid choosing a later symptom if an earlier root cause by the same agent is available.

Return JSON only with keys:
contains_error, agent, step, reason, confidence.
"""


def paper_hybrid_target_agent_joint_prompt(
    case: Case,
    target_agent: str,
    candidate_steps: list[dict[str, Any]],
    selection_source: str,
) -> str:
    rendered_candidates: list[str] = []
    for item in candidate_steps:
        step = item["step"]
        before_context = item.get("before_context") or []
        after_context = item.get("after_context") or []
        rendered_candidates.append(
            "[Candidate Step {step_num} | Agent: {agent}]\n"
            "Previous Context:\n{before}\n"
            "Target Agent Action:\n{current}\n"
            "Next Context:\n{after}".format(
                step_num=step.step,
                agent=step.agent,
                before=render_steps(before_context) if before_context else "None",
                current=render_steps([step]),
                after=render_steps(after_context) if after_context else "None",
            )
        )
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are running the hybrid failure attribution method.

{header(case)}

Predicted Responsible Agent:
{target_agent}

Agent Selection Source:
{selection_source}

Target-Agent Candidate Steps:
{chr(10).join(rendered_candidates)}

Task:
The responsible agent has already been selected. Reread all candidate actions by that agent in temporal order and identify the earliest decisive error by the predicted responsible agent.
Choose only a step from the Target-Agent Candidate Steps.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest target-agent action that could be the root cause of the final wrong answer.
- Prefer the first root-cause target-agent action over a later downstream symptom.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible target-agent mistake, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected target-agent step should be one whose minimal correction could plausibly change the final wrong answer into a successful answer.

DECISION PROCEDURE:
1. Read the target-agent candidate steps together in temporal order.
2. Identify all plausible mistakes made by the predicted responsible agent.
3. Discard downstream symptoms and target-agent actions that were clearly recovered.
4. Select the earliest remaining target-agent root-cause mistake.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "agent": "<predicted responsible agent name, or NONE>",
  "step": <integer step number from the Target-Agent Candidate Steps, or null>,
  "reason": "<brief reason explaining why this is the earliest decisive target-agent mistake>"
}}
"""


def paper_hybrid_global_chunk_router_prompt(
    case: Case,
    target_agent: str,
    conversation: str,
    chunk_ranges: list[dict[str, Any]],
    beam_k: int,
) -> str:
    chunk_lines: list[str] = []
    for item in chunk_ranges:
        estimated_tokens = item.get("estimated_tokens")
        token_note = f", estimated_tokens={estimated_tokens}" if estimated_tokens is not None else ""
        chunk_lines.append(
            "Chunk {chunk_id}: steps {start_step}-{end_step}, step_count={step_count}{token_note}".format(
                chunk_id=item.get("chunk_id"),
                start_step=item.get("start_step"),
                end_step=item.get("end_step"),
                step_count=item.get("step_count"),
                token_note=token_note,
            )
        )
    chunk_table = "\n".join(chunk_lines)
    return f"""You are an AI assistant performing failure analysis on a long multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are running a context-allocation wrapper for the hybrid failure attribution method.

The hybrid method has already selected a responsible target agent. Your task is NOT to produce the final step attribution.
Your task is to read the whole trace once, inspect the chunk table, and select the top {beam_k} chunk IDs that should be reread for target-agent hybrid localization.

{header(case)}

Predicted Responsible Agent:
{target_agent}

Chunk table:
{chunk_table}

Full conversation:
{conversation}

Chunk selection objective:
- Select the top {beam_k} chunk IDs most likely to contain the earliest decisive error made by the predicted responsible agent.
- Prefer the earliest target-agent root cause over later downstream symptoms.
- Do not select chunks merely because they contain repeated failures, final symptoms, or confident-looking late consequences.
- Do not select chunks because another agent shows a later visible tool failure if an earlier target-agent causal mistake is more plausible.
- Use only chunk IDs from the table.

FINAL ATTRIBUTION CRITERIA FOR ROUTING:
- A useful chunk should contain, or provide direct local evidence for, the first target-agent action that could be the root cause of the final wrong answer.
- A decisive target-agent error is an action whose minimal correction could plausibly change the failed trajectory into a successful answer.
- Harmless failed attempts that were later corrected should not be prioritized.

DECISION PROCEDURE:
1. Identify the target agent's plausible mistakes across the whole trace.
2. Distinguish early root-cause target-agent mistakes from later symptoms or recovery attempts.
3. Map the most important target-agent candidate regions to chunk IDs.
4. Return at most {beam_k} chunk IDs for the next hybrid rereading stage.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "selected_chunk_ids": [1, 2],
  "rationale": "<brief reason for selecting these chunks for target-agent hybrid localization>"
}}
"""


def paper_hybrid_step_context_prompt(
    case: Case,
    target_agent: str,
    before_context: list[LogStep],
    current_step: LogStep,
    after_context: list[LogStep],
) -> str:
    return f"""You are running the hybrid failure attribution method.
The responsible agent has already been predicted by a chunk-level or all-at-once judge.

{header(case)}

Predicted Responsible Agent:
{target_agent}

Immediate Previous Context:
{render_steps(before_context) if before_context else "None"}

Candidate Step Under Review:
Step {current_step.step}
Agent: {current_step.agent}
Content:
{current_step.content}

Immediate Next Context:
{render_steps(after_context) if after_context else "None"}

Task:
Determine whether this candidate action by the predicted responsible agent is the earliest decisive error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Use the previous and next context only to judge causality and downstream effects.
The returned step must be the Candidate Step Under Review, or null if that candidate is not the earliest decisive error.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Avoid choosing a later symptom if an earlier root cause by the same agent is available.

Return JSON only with keys:
contains_error, agent, step, reason, confidence.
"""


def paper_hybrid_step_bool_prompt(case: Case, target_agent: str, history: list[LogStep], current_step: LogStep) -> str:
    return f"""You are running the hybrid failure attribution method.
The responsible agent has already been predicted by a chunk-level vote.

{header(case)}

Predicted Responsible Agent:
{target_agent}

History Up To Current Candidate Step:
{render_steps(history)}

Candidate Step Under Review:
Step {current_step.step}
Agent: {current_step.agent}

Task:
Determine whether this candidate action by the predicted responsible agent is the earliest decisive error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Avoid choosing a later symptom if an earlier root cause by the same agent is available.
Do not output confidence, likelihood scores, or any numeric scoring field.

Return JSON only with keys:
contains_error, agent, step, reason.
"""


def paper_hybrid_step_ordinal_prompt(case: Case, target_agent: str, history: list[LogStep], current_step: LogStep) -> str:
    return f"""You are running the hybrid failure attribution method.
The responsible agent has already been selected from the highest-scoring chunk.

{header(case)}

Predicted Responsible Agent:
{target_agent}

History Up To Current Candidate Step:
{render_steps(history)}

Candidate Step Under Review:
Step {current_step.step}
Agent: {current_step.agent}

Task:
Score whether this candidate action by the predicted responsible agent is the earliest decisive error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Do not score downstream symptoms highly.
Do not score harmless failed attempts highly if they were later corrected.
Prefer the earliest causal mistake over later consequences.
Do not output confidence.
Do not output a separate boolean.

Return exactly one JSON object:
{{
  "agent": "<agent name or NONE>",
  "step": <integer or null>,
  "blame_score": <0, 1, 2, 3, or 4>
}}

Scoring rubric:
0 = no evidence of a decisive error
1 = weak clue only
2 = local error or symptom, but not clearly decisive
3 = likely decisive error
4 = clearly decisive causal error with identifiable agent and step

Rules:
- If blame_score is 0 or 1, use agent = "NONE" and step = null.
- Use only one numeric score.
- Do not output confidence.
- Do not output a separate boolean.
"""


def mvbs_chunk_scoring_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are judging a failed multi-agent trajectory.

{header(case)}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Decide whether this chunk likely contains the earliest decisive error.
A decisive error is the first agent action that, if corrected, could change the failed trajectory into a successful one.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Evaluate from four views:
1. Forward onset: does the trajectory first go wrong here?
2. Backward causality: does the final failure trace back to this chunk?
3. Agent-conditioned: which agent action is most suspicious?
4. Answer contrast: does this chunk conflict with the task or ground truth?

Prefer root causes over downstream symptoms.
Return JSON only with keys:
chunk_id, likely_contains_decisive_error, onset_score, causal_impact_score,
answer_contrast_score, agent_specificity_score, symptom_penalty, confidence,
candidate_steps.
candidate_steps should be a list of objects with step, agent, reason.
Scores must be between 0 and 1.
"""


def mvbs_window_prompt(
    case: Case,
    region: list[LogStep],
    window: list[LogStep],
    before: list[LogStep],
    after: list[LogStep],
) -> str:
    return f"""You are localizing the earliest decisive error inside a suspected region.

{header(case)}

Suspected Region:
Steps {region[0].step}-{region[-1].step}

Window Under Review:
Steps {window[0].step}-{window[-1].step}

Log Window:
{render_steps(window)}

Immediate Previous Context:
{render_steps(before) if before else "None"}

Immediate Next Context:
{render_steps(after) if after else "None"}

Task:
Identify whether the earliest decisive error occurs inside this window.
If yes, choose the exact step and responsible agent.
Prefer root causes over downstream symptoms.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
contains_decisive_error, candidate_step, agent, root_cause_score,
is_downstream_symptom, confidence, reason.
"""


def mvbs_pairwise_prompt(
    case: Case,
    candidate_a: dict,
    candidate_b: dict,
    context_a: list[LogStep],
    context_b: list[LogStep],
) -> str:
    return f"""You are comparing two candidate decisive error steps.

{header(case)}

Candidate A:
Step {candidate_a.get("step")}
Agent: {candidate_a.get("agent")}
Content:
{candidate_a.get("content", "")}
Reason:
{candidate_a.get("reason", "")}

Candidate B:
Step {candidate_b.get("step")}
Agent: {candidate_b.get("agent")}
Content:
{candidate_b.get("content", "")}
Reason:
{candidate_b.get("reason", "")}

Relevant Context Around Candidate A:
{render_steps(context_a)}

Relevant Context Around Candidate B:
{render_steps(context_b)}

Task:
Choose which candidate is more likely to be the earliest decisive error.
The decisive error is the first action that caused the trajectory to become unrecoverably wrong.
Do not choose a later symptom if an earlier root cause is available.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
winner, confidence, reason.
winner must be "A" or "B".
"""


def ccv_constraint_prompt(case: Case) -> str:
    gt_note = "Use the ground truth answer when available." if case.ground_truth else "No ground truth answer is available; rely on task requirements."
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are preparing constraints for diagnosing the failed multi-agent task.

{header(case)}

Task:
Generate task-specific constraints that must hold for the agents to solve the problem correctly.
{gt_note}
Include evidence, calculation, tool-use, interpretation, and final-answer constraints when relevant.
Each constraint should have clear violation criteria.

CONSTRAINT DESIGN CRITERIA:
- Each constraint should be directly checkable against agent actions or tool outputs.
- Each violation criterion should help identify a possible hidden cause of failure, such as missing evidence, wrong assumptions, calculation errors, tool misuse, or misinterpretation.
- Prefer constraints that can distinguish an early root-cause mistake from a later downstream symptom.
- When possible, phrase violation criteria so that a minimal corrected action would be clear.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "constraints": [
    {{
      "id": "<short constraint id>",
      "type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
      "description": "<what must hold for success>",
      "violation_criteria": "<how to recognize a violation in the trace>"
    }}
  ]
}}
"""


def ccv_checklist_equivalent_generation_prompt(case: Case) -> str:
    gt_note = "Use the ground truth answer when available." if case.ground_truth else "No ground truth answer is available; rely on task requirements."
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are preparing checklist items for diagnosing the failed multi-agent task.

{header(case)}

Task:
Generate task-specific checklist items that must be satisfied for the agents to solve the problem correctly.
{gt_note}
Include evidence, calculation, tool-use, interpretation, and final-answer checklist items when relevant.
Each checklist item should have a clear failure indicator.

CHECKLIST DESIGN CRITERIA:
- Each checklist item should be directly checkable against agent actions or tool outputs.
- Each failure indicator should help identify a possible hidden cause of failure, such as missing evidence, wrong assumptions, calculation errors, tool misuse, or misinterpretation.
- Prefer checklist items that can distinguish an early root-cause mistake from a later downstream symptom.
- When possible, phrase failure indicators so that a minimal corrected action would be clear.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "checklist_items": [
    {{
      "id": "<short checklist item id>",
      "category": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
      "success_checkpoint": "<what must hold for success>",
      "failure_indicator": "<how to recognize a breakdown in the trace>"
    }}
  ]
}}
"""


def task_checklist_generation_prompt(case: Case) -> str:
    agents = ", ".join(sorted({step.agent for step in case.steps}))
    return f"""You are an AI assistant preparing a task-success checklist for diagnosing a failed multi-agent task.

Problem:
{case.problem}

Agents:
{agents}

Task:
Generate task-specific checklist items that describe what the agents need to accomplish to solve the problem correctly.
You do not know the ground-truth answer. Derive the checklist only from the problem statement.
Include evidence, calculation, tool-use, interpretation, and answer-quality checkpoints when relevant.
Each checklist item should have a clear failure indicator that can be checked in the execution trace.

CHECKLIST DESIGN CRITERIA:
- Include only checkpoints that are entailed by the task or broadly necessary for a correct solution.
- Do not invent a specific answer, source, fact, numeric value, or unique solution path that is not given in the problem.
- Allow multiple valid solution strategies. Mark an item as conditional when it applies only to one strategy.
- Each checklist item should be directly checkable against agent actions or tool outputs.
- Each failure indicator should expose a possible hidden cause of failure, such as missing evidence, wrong assumptions, calculation errors, tool misuse, or misinterpretation.
- Prefer checklist items that can distinguish an early root-cause mistake from a later downstream symptom.
- When possible, phrase the failure indicator so that a minimal corrected action would be clear.
- An answer-quality item may require completeness, support, consistency, or format compliance, but must not assert an unknown answer value.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "checklist_items": [
    {{
      "id": "<short checklist item id>",
      "category": "<evidence|calculation|tool-use|interpretation|answer-quality|other>",
      "success_checkpoint": "<what the agents need to accomplish>",
      "failure_indicator": "<how to recognize a breakdown in the trace>",
      "conditional": <true or false>
    }}
  ]
}}
"""


def reference_checklist_generation_prompt(case: Case) -> str:
    agents = ", ".join(sorted({step.agent for step in case.steps}))
    return f"""You are an AI assistant preparing a task-success checklist for diagnosing a failed multi-agent task.

Problem:
{case.problem}

Ground Truth Answer:
{case.ground_truth or ""}

Agents:
{agents}

Task:
Generate task-specific checklist items that describe what the agents need to accomplish to solve the problem correctly.
Use the ground-truth answer as a reference for identifying the evidence, calculations, tool use, and interpretations needed for success.
Include evidence, calculation, tool-use, interpretation, and answer-quality checkpoints when relevant.
Each checklist item should have a clear failure indicator that can be checked in the execution trace.

CHECKLIST DESIGN CRITERIA:
- Compile the reference answer into diagnostic intermediate checkpoints rather than merely restating the answer.
- Include checkpoints that connect the task requirements to the evidence and reasoning needed to support the reference answer.
- Do not make "the final answer must equal the ground truth" the only or dominant checklist item.
- Allow multiple valid solution strategies. Mark an item as conditional when it applies only to one strategy.
- Each checklist item should be directly checkable against agent actions or tool outputs.
- Each failure indicator should expose a possible hidden cause of failure, such as missing evidence, wrong assumptions, calculation errors, tool misuse, or misinterpretation.
- Prefer checklist items that can distinguish an early root-cause mistake from a later downstream symptom.
- When possible, phrase the failure indicator so that a minimal corrected action would be clear.
- An answer-quality item may check whether the response is complete, supported, consistent, correctly formatted, and compatible with the reference answer.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "checklist_items": [
    {{
      "id": "<short checklist item id>",
      "category": "<evidence|calculation|tool-use|interpretation|answer-quality|other>",
      "success_checkpoint": "<what the agents need to accomplish>",
      "failure_indicator": "<how to recognize a breakdown in the trace>",
      "conditional": <true or false>
    }}
  ]
}}
"""


def ccv_global_chunk_router_prompt(
    case: Case,
    constraints: list[dict],
    conversation: str,
    chunk_ranges: list[dict[str, Any]],
    beam_k: int,
) -> str:
    chunk_lines: list[str] = []
    for item in chunk_ranges:
        estimated_tokens = item.get("estimated_tokens")
        token_note = f", estimated_tokens={estimated_tokens}" if estimated_tokens is not None else ""
        chunk_lines.append(
            "Chunk {chunk_id}: steps {start_step}-{end_step}, step_count={step_count}{token_note}".format(
                chunk_id=item.get("chunk_id"),
                start_step=item.get("start_step"),
                end_step=item.get("end_step"),
                step_count=item.get("step_count"),
                token_note=token_note,
            )
        )
    chunk_table = "\n".join(chunk_lines)
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are selecting evidence chunks for a constraint-guided verifier in the failed multi-agent task.

{header(case)}

Constraints:
{constraints}

Your task is NOT to produce the final agent/step attribution.
Your task is to read the whole trace once, inspect the chunk table, and select the top {beam_k} chunk IDs that should be reread for constraint-violation localization.
Use the task constraints as the primary diagnostic lens while routing chunks.

Chunk table:
{chunk_table}

Full conversation:
{conversation}

ROUTING CRITERIA:
- Select the top {beam_k} chunk IDs most likely to contain the earliest meaningful constraint violation that caused the final task failure.
- Prefer the earliest unrecovered root-cause violation over later downstream symptoms.
- Prefer chunks where the violation appears tied to a hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation.
- Prefer chunks where a minimal correction at that step could plausibly change the final wrong answer into a successful answer.
- Do not select chunks merely because they contain repeated failures, final symptoms, or confident-looking late consequences.
- Do not select harmless failed attempts that were later corrected.
- Use only chunk IDs from the table.

ROUTING DECISION:
Based on the constraints and the full trace, select the chunk IDs most likely to contain:
1. the decisive root-cause agent action,
2. the first step where a task-success constraint was violated,
3. the minimal corrective intervention that would have prevented the failure,
4. evidence that the violation causally led to the final wrong answer.

OUTPUT FORMAT:
Return valid JSON wrapped in <json></json> tags. Do not output any text outside the tags.

<json>
{{
  "selected_chunk_ids": [<integer chunk id>, <integer chunk id>],
  "rationale": "<brief constraint-based reason for selecting these chunks>"
}}
</json>
"""


def ccv_full_trace_prompt(case: Case, constraints: list[dict], steps: list[LogStep]) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing constraint-guided failure localization for the failed multi-agent task.

{header(case)}

Constraints:
{constraints}

Full conversation:
{render_steps(steps)}

Task:
Using the constraints as the primary diagnostic lens, identify the earliest unrecovered constraint violation
that causally led to the final wrong answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that violated a task-success constraint.
- Prefer the first root-cause agent action over later downstream symptoms.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible violation, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected violation should be one whose minimal correction could plausibly change the final wrong answer into a successful answer.

DECISION PROCEDURE:
1. Scan the conversation in temporal order.
2. Identify steps where an agent violates a task-success constraint.
3. For each plausible violation, check the hidden cause, minimal corrected action, and likely outcome change.
4. Ignore violations that are clearly recovered before the final answer.
5. Select the earliest unrecovered violation whose minimal correction could plausibly change the final failure into success.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "violated_constraint": "<constraint id or description>",
  "violation_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the constraint violation and causal link>"
}}
"""


def ccv_causal_no_requirements_prompt(case: Case, steps: list[LogStep]) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing causal failure localization for the failed multi-agent task.

{header(case)}

Full conversation:
{render_steps(steps)}

Task:
Identify the earliest unrecovered decisive error that causally led to the final wrong answer.
Do not attribute the failure to the human/user/problem statement.
Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that directly contributed to the final task failure.
- Prefer the first root-cause agent action over later downstream symptoms.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible error, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected error should be one whose minimal correction could plausibly change the final wrong answer into a successful answer.

DECISION PROCEDURE:
1. Scan the conversation in temporal order.
2. Identify candidate agent actions that could have caused the final task failure.
3. For each plausible candidate, check the hidden cause, minimal corrected action, and likely outcome change.
4. Ignore candidate errors that are clearly recovered before the final answer.
5. Select the earliest unrecovered error whose minimal correction could plausibly change the final failure into success.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "decisive_error": "<brief description of the selected error>",
  "error_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the error and its causal link to the final failure>"
}}
"""


def ccv_requirements_direct_prompt(
    case: Case,
    constraints: list[dict],
    steps: list[LogStep],
) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing requirement-guided direct failure attribution for the failed multi-agent task.

{header(case)}

Constraints:
{constraints}

Full conversation:
{render_steps(steps)}

Task:
Using the constraints as the primary diagnostic references, identify the earliest decisive error in the full conversation:
1. the system agent that made the earliest decisive error,
2. the exact step where that error first occurred,
3. the task-success constraint violated at that step,
4. a brief reason for the attribution.

Base the attribution on evidence in the full conversation.
Prefer the earliest responsible error over a later downstream consequence.
Do not attribute the failure to the human/user/problem statement.
Choose an action taken by the multi-agent system.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "violated_constraint": "<constraint id or description>",
  "violation_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the attribution>"
}}
"""


def tsr_direct_no_requirements_prompt(
    case: Case,
    steps: list[LogStep],
) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing direct failure attribution for the failed multi-agent task.

{header(case)}

Full conversation:
{render_steps(steps)}

Task:
Identify the earliest decisive error in the full conversation:
1. the system agent that made the earliest decisive error,
2. the exact step where that error first occurred,
3. the error type,
4. a brief reason for the attribution.

Base the final attribution on evidence in the full conversation.
Prefer the earliest responsible error over a later downstream consequence.
Do not attribute the failure to the human/user/problem statement.
Choose an action taken by the multi-agent system.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "error_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the attribution>"
}}
"""


def _tsr_minimal_agent_step_prompt(
    case: Case,
    steps: list[LogStep],
    requirements: list[dict] | None,
) -> str:
    agents = ", ".join(sorted({step.agent for step in steps}))
    requirements_block = ""
    if requirements is not None:
        requirements_block = (
            "\nTASK-SUCCESS REQUIREMENTS:\n"
            f"{requirements}\n\n"
            "Use these requirements as diagnostic references.\n"
        )
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.

PROBLEM:
{case.problem}

AGENTS:
{agents}
{requirements_block}
FULL CONVERSATION:
{render_steps(steps)}

TASK:
Identify:
1. the system agent that made the earliest decisive error,
2. the exact step where that error first occurred.

Return exactly one valid JSON object with no markdown or extra commentary:
{{
  "agent": "<agent name>",
  "step": <integer>
}}
"""


def tsr_minimal_r0_prompt(
    case: Case,
    steps: list[LogStep],
) -> str:
    """Render the agent-and-step-only condition without requirements."""
    return _tsr_minimal_agent_step_prompt(
        case=case,
        steps=steps,
        requirements=None,
    )


def tsr_minimal_r1_prompt(
    case: Case,
    requirements: list[dict],
    steps: list[LogStep],
) -> str:
    """Render the matched agent-and-step-only condition with requirements."""
    return _tsr_minimal_agent_step_prompt(
        case=case,
        steps=steps,
        requirements=requirements,
    )


def ccv_trace_elephant_full_trace_prompt(
    case: Case,
    constraints: list[dict],
    steps: list[LogStep],
) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing constraint-guided failure localization under the TraceElephant attribution definition.

{header(case)}

Constraints:
{constraints}

Full conversation:
{render_steps(steps)}

Task:
Using the constraints as the primary diagnostic lens, identify the earliest execution step at which task
failure becomes inevitable under the system's existing agents, roles, tools, and remaining recovery opportunities.
Do not simply choose the first visible mistake. Do not attribute the failure to the human/user/problem statement.
Choose an action taken by the multi-agent system.

TRACE-ELEPHANT ATTRIBUTION CRITERIA:
- Use a role-aware and recoverability-aware definition of responsibility.
- A step is decisive only when, after that step, no feasible continuation by the existing system could still
  satisfy the task-success constraints and produce a successful answer.
- If an upstream mistake remains recoverable because a later verifier, orchestrator, or other responsible agent
  is expected and able to check or correct it, do not attribute the failure to that upstream mistake.
- In that situation, attribute the failure to the step where the responsible component misses the last meaningful
  recovery opportunity and the failure becomes inevitable.
- Select the agent that acts at the decisive step.
- Do not choose a final downstream symptom merely because it is conspicuous; choose it only if it is itself the
  point where the final recovery opportunity is lost.

DECISION PROCEDURE:
1. Scan the conversation in temporal order and identify violations of the task-success constraints.
2. For every plausible violation, identify which later agents or roles are responsible for detecting, correcting,
   verifying, or recovering from it.
3. Determine whether a feasible successful continuation still exists after each candidate step using the system's
   observed roles, tools, information, and remaining actions.
4. Reject an early visible mistake when the trace shows a later explicit recovery responsibility and a realistic
   opportunity to fulfill it.
5. Select the earliest step after which no feasible continuation could recover the task, and select the agent acting
   at that step.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name acting at that step>",
  "violated_constraint": "<constraint id or description>",
  "recovery_status": "<why recovery remained possible before this step but not after it>",
  "missed_recovery_responsibility": "<responsible role/action, or NONE if the step itself made failure immediately inevitable>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief causal explanation of why this is the earliest inevitable failure step>"
}}
"""


def ccv_constraint_only_full_trace_prompt(
    constraints: list[dict],
    steps: list[LogStep],
) -> str:
    return f"""You are an AI assistant checking a failed multi-agent execution against a provided constraint set.

Constraints:
{constraints}

Full conversation:
{render_steps(steps)}

Task:
Use only the provided constraints and the explicit evidence in the conversation.
Identify the single agent action and step that are best supported as the failure attribution by the observed
constraint violations.

NEUTRAL JUDGMENT RULES:
- Check agent actions and tool outputs only against the provided constraints.
- Select one exact step from the conversation and the agent acting at that step.
- Do not use an unstated task plan, external facts, or assumptions not encoded in the constraints or trace.
- Do not prefer an earlier or later step merely because of its position.
- Do not apply an additional earliest-root-cause, recoverability, counterfactual, or downstream-symptom rule.
- Do not output a confidence score.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name acting at that step>",
  "violated_constraint": "<constraint id or description>",
  "reason": "<brief explanation grounded only in the constraint violation and trace evidence>"
}}
"""


def ccv_checklist_equivalent_full_trace_prompt(
    case: Case,
    checklist_items: list[dict],
    steps: list[LogStep],
) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing checklist-guided failure localization for the failed multi-agent task.

{header(case)}

Task-success checklist:
{checklist_items}

Full conversation:
{render_steps(steps)}

Task:
Using the checklist as the primary diagnostic lens, identify the earliest unrecovered checklist breakdown
that causally led to the final wrong answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that failed to satisfy a task-success checklist item.
- Prefer the first root-cause agent action over later downstream symptoms.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected breakdown should be one whose minimal correction could plausibly change the final wrong answer into a successful answer.

DECISION PROCEDURE:
1. Scan the conversation in temporal order.
2. Identify steps where an agent fails to satisfy a task-success checklist item.
3. For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
4. Ignore breakdowns that are clearly recovered before the final answer.
5. Select the earliest unrecovered breakdown whose minimal correction could plausibly change the final failure into success.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "failed_checklist_item": "<checklist item id or success checkpoint>",
  "failure_category": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the checklist breakdown and causal link>"
}}
"""


def task_checklist_full_trace_prompt(case: Case, checklist_items: list[dict], steps: list[LogStep]) -> str:
    agents = ", ".join(sorted({step.agent for step in case.steps}))
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing task-checklist-guided failure localization without access to a ground-truth answer.

Problem:
{case.problem}

Agents:
{agents}

Task-success checklist:
{checklist_items}

Full conversation:
{render_steps(steps)}

Task:
Using only the problem, the task-success checklist, and evidence in the conversation, identify the earliest
unrecovered checklist breakdown that causally led to the failed result.
Do not infer or assume an unavailable ground-truth answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that failed to satisfy a relevant checklist item.
- Prefer the first root-cause agent action over later downstream symptoms.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected breakdown should be one whose minimal correction could plausibly change the failed result into a successful result.
- Base the decision on observable evidence in the trace, not on a guessed correct answer.

DECISION PROCEDURE:
1. Scan the conversation in temporal order.
2. Identify steps where an agent fails to satisfy a relevant checklist item.
3. For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
4. Ignore breakdowns that are clearly recovered before the final response.
5. Select the earliest unrecovered checklist breakdown whose minimal correction could plausibly change the failure into success.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "failed_checklist_item": "<checklist item id or success checkpoint>",
  "failure_category": "<evidence|calculation|tool-use|interpretation|answer-quality|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the checklist breakdown and causal link>"
}}
"""


def reference_checklist_full_trace_prompt(case: Case, checklist_items: list[dict], steps: list[LogStep]) -> str:
    agents = ", ".join(sorted({step.agent for step in case.steps}))
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are performing reference-checklist-guided failure localization with access to a ground-truth answer.

Problem:
{case.problem}

Ground Truth Answer:
{case.ground_truth or ""}

Agents:
{agents}

Reference-derived task-success checklist:
{checklist_items}

Full conversation:
{render_steps(steps)}

Task:
Using the problem, the ground-truth answer, the reference-derived checklist, and evidence in the conversation,
identify the earliest unrecovered checklist breakdown that causally led to the failed result.
Use the reference answer to evaluate the required evidence and reasoning, not merely to select the final response step.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that failed to satisfy a relevant checklist item.
- Prefer the first root-cause agent action over later downstream symptoms or a final-answer mismatch.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected breakdown should be one whose minimal correction could plausibly change the failed result into a successful result.
- Use the reference answer to verify causal relevance, but ground the selected location in observable trace evidence.

DECISION PROCEDURE:
1. Scan the conversation in temporal order.
2. Identify steps where an agent fails to satisfy a relevant checklist item.
3. For each plausible breakdown, check the hidden cause, minimal corrected action, and likely outcome change.
4. Ignore breakdowns that are clearly recovered before the final response.
5. Select the earliest unrecovered checklist breakdown whose minimal correction could plausibly change the failure into success.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number from the conversation>,
  "agent": "<agent name>",
  "failed_checklist_item": "<checklist item id or success checkpoint>",
  "failure_category": "<evidence|calculation|tool-use|interpretation|answer-quality|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining the checklist breakdown and causal link>"
}}
"""


def ccv_chunk_prompt(
    case: Case,
    constraints: list[dict],
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are checking whether a chunk contains a decisive task-success constraint violation.

{header(case)}

Constraints:
{constraints}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Using the task constraints as the primary diagnostic lens, determine whether this chunk contains the first meaningful constraint violation that causally explains the final failure.
Prefer the earliest unrecovered root-cause violation over later symptoms.
Do not score harmless failed attempts highly if they were later corrected.
When a violation looks plausible, consider whether it reflects a hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation.
Also consider whether a minimal correction at that step could plausibly change the final wrong answer into a successful answer.
Score severity, irreversibility, evidence strength, and confidence according to counterfactual causal impact:
high scores mean the violation is grounded, unrecovered, and likely to change the final answer if corrected.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
chunk_id, contains_violation, violated_constraints, earliest_suspected_step,
agent, severity, irreversibility, evidence_strength, downstream_symptom_penalty,
confidence, reason.
Scores must be between 0 and 1.
"""


def ccv_chunk_context_prompt(
    case: Case,
    constraints: list[dict],
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    before_context: list[LogStep],
    after_context: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are checking whether a focal chunk contains a decisive task-success constraint violation.

{header(case)}

Constraints:
{constraints}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Focal Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Immediate Previous Context:
{render_steps(before_context) if before_context else "None"}

Focal Chunk:
{render_steps(chunk)}

Immediate Next Context:
{render_steps(after_context) if after_context else "None"}

Next Chunk Summary:
{next_summary}

Task:
Using the task constraints as the primary diagnostic lens, determine whether this focal chunk contains the first meaningful constraint violation that causally explains the final failure.
Use the previous and next context only to understand causality, recovery, and downstream symptoms.
Do not choose a step from the previous or next context as the answer.
Prefer the earliest unrecovered root-cause violation over later symptoms.
Do not score harmless failed attempts highly if they were later corrected.
When a focal violation looks plausible, consider whether it reflects a hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation.
Also consider whether a minimal correction at that focal step could plausibly change the final wrong answer into a successful answer.
Score severity, irreversibility, evidence strength, and confidence for the focal chunk only according to counterfactual causal impact.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
chunk_id, contains_violation, violated_constraints, earliest_suspected_step,
agent, severity, irreversibility, evidence_strength, downstream_symptom_penalty,
confidence, reason.
earliest_suspected_step must be an exact step number from the Focal Chunk, or null if the Focal Chunk is unlikely to contain the root-cause violation.
Scores must be between 0 and 1.
"""


def ccv_scalar_chunk_prompt(
    case: Case,
    constraints: list[dict],
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are checking whether a chunk contains the decisive root-cause constraint violation in a failed multi-agent task.

{header(case)}

Constraints:
{constraints}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Using the task constraints as the primary diagnostic lens, determine whether this chunk contains the earliest root-cause constraint violation that causally explains the final failure.
Use a single scalar root-cause score only. Do not separately score severity,
irreversibility, evidence strength, confidence, or symptom penalty.
Prefer the earliest unrecovered root cause over a later downstream symptom.
Do not score harmless failed attempts highly if they were later corrected.
Give a high root_cause_score only when the suspected violation is grounded, unrecovered, and a minimal correction at that step could plausibly change the final wrong answer into a successful answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
chunk_id, contains_violation, violated_constraints, earliest_suspected_step,
agent, root_cause_score, reason.
root_cause_score must be between 0 and 1, where 1 means this chunk is very likely to contain the earliest decisive error.
"""


def ccv_ordinal_chunk_prompt(
    case: Case,
    constraints: list[dict],
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are checking whether a chunk contains the decisive root-cause constraint violation in a failed multi-agent task.

{header(case)}

Constraints:
{constraints}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now reviewing Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Using the task constraints as the primary diagnostic lens, identify whether this chunk contains evidence of the earliest decisive error
that caused the final task failure.
Do not score downstream symptoms highly.
Do not score harmless failed attempts highly if they were later corrected.
Prefer the earliest unrecovered causal constraint violation over later consequences.
When assigning blame_score, consider whether the violation reflects a hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation, and whether a minimal correction could plausibly change the final answer.
Do not output confidence.
Do not output a separate boolean.

Return exactly one JSON object:
{{
  "agent": "<agent name or NONE>",
  "step": <integer or null>,
  "blame_score": <0, 1, 2, 3, or 4>
}}

Scoring rubric:
0 = no evidence of a decisive error
1 = weak clue only
2 = local constraint violation or downstream symptom, but not clearly decisive
3 = likely decisive constraint violation
4 = clearly decisive causal constraint violation with identifiable agent and step

Rules:
- If blame_score is 0 or 1, use agent = "NONE" and step = null.
- Use only one numeric score.
- Do not output confidence.
- Do not output a separate boolean.
"""


def ccv_step_prompt(case: Case, constraints: list[dict], chunk: list[LogStep]) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are localizing the first decisive constraint violation inside a suspected chunk.

{header(case)}

Constraints:
{constraints}

Suspected Chunk:
Steps {chunk[0].step}-{chunk[-1].step}

Chunk Log:
{render_steps(chunk)}

Task:
Using the constraints as the primary diagnostic lens, find the earliest step where a constraint is violated in a way that could cause final task failure.
Return the responsible agent and exact step.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

LOCALIZATION CRITERIA:
- Select only a step inside the suspected chunk.
- If several steps violate constraints, choose the earliest unrecovered root cause, not the clearest later symptom.
- Do not choose a harmless failed attempt if it was later corrected.
- The reason should mention the violated constraint, the hidden cause such as misunderstanding or missing evidence, the minimal corrected action, and why that correction would change the failure trajectory.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number inside the suspected chunk>,
  "agent": "<agent name>",
  "violated_constraint": "<constraint id or description>",
  "violation_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason explaining why this is the earliest decisive constraint violation in the chunk>"
}}
"""


def ccv_selected_chunks_joint_prompt(
    case: Case,
    constraints: list[dict],
    selected_chunks: list[dict[str, Any]],
) -> str:
    rendered_chunks: list[str] = []
    for item in selected_chunks:
        chunk = item["chunk"]
        rendered_chunks.append(
            "[Chunk {chunk_id} | steps {start}-{end}]\n{content}".format(
                chunk_id=item.get("chunk_id"),
                start=chunk[0].step,
                end=chunk[-1].step,
                content=render_steps(chunk),
            )
        )
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are rereading selected evidence chunks in temporal order for constraint-guided failure localization.

{header(case)}

Constraints:
{constraints}

Selected Chunks:
{chr(10).join(rendered_chunks)}

Task:
Using the constraints as the primary diagnostic lens, identify the earliest unrecovered constraint violation across the selected chunks that causally led to the final wrong answer.
Choose only a step that appears inside the selected chunks.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

FINAL ATTRIBUTION CRITERIA:
- Choose the earliest unrecovered agent action that violated a task-success constraint across the selected chunks.
- Prefer the first root-cause agent action over later downstream symptoms.
- Do not choose a harmless failed attempt if it was later corrected.
- For each plausible violation, check the hidden cause, minimal corrected action, and likely outcome change.
- The selected violation should be one whose minimal correction could plausibly change the final wrong answer into a successful answer.

DECISION PROCEDURE:
1. Read the selected chunks together in temporal order.
2. Identify all plausible constraint violations inside the selected chunks.
3. Discard downstream symptoms and violations that were clearly recovered.
4. Among the remaining candidates, select the earliest root-cause violation whose correction could plausibly change the final outcome.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "step": <integer step number inside the selected chunks>,
  "agent": "<agent name>",
  "violated_constraint": "<constraint id or description>",
  "violation_type": "<evidence|calculation|tool-use|interpretation|final-answer|other>",
  "reason": "<brief reason explaining why this is the earliest decisive constraint violation across the selected chunks>"
}}
"""


def ccv_step_context_prompt(
    case: Case,
    constraints: list[dict],
    chunk: list[LogStep],
    before_context: list[LogStep],
    after_context: list[LogStep],
) -> str:
    return f"""You are localizing the first decisive constraint violation inside a suspected focal chunk.

{header(case)}

Constraints:
{constraints}

Suspected Focal Chunk:
Steps {chunk[0].step}-{chunk[-1].step}

Immediate Previous Context:
{render_steps(before_context) if before_context else "None"}

Focal Chunk Log:
{render_steps(chunk)}

Immediate Next Context:
{render_steps(after_context) if after_context else "None"}

Task:
Using the constraints as the primary diagnostic lens, find the earliest step inside the Focal Chunk where a constraint is violated in a way that could cause final task failure.
Use the previous and next context only to judge whether the focal step is causal or merely a symptom.
Do not return a step from the previous or next context.
If several focal steps violate constraints, choose the earliest unrecovered root cause, not the clearest later symptom.
Do not choose a harmless failed attempt if it was later corrected.
Your reason should mention the violated constraint, the hidden cause such as misunderstanding or missing evidence, the minimal corrected action, and why that correction would change the failure trajectory.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return JSON only with keys:
step, agent, violated_constraint, violation_type, confidence, reason.
step must be an exact step number from the Focal Chunk.
"""


def ccv_step_bool_prompt(case: Case, constraints: list[dict], chunk: list[LogStep]) -> str:
    return f"""You are localizing the first decisive constraint violation inside a suspected chunk.

{header(case)}

Constraints:
{constraints}

Suspected Chunk:
Steps {chunk[0].step}-{chunk[-1].step}

Chunk Log:
{render_steps(chunk)}

Task:
Using the constraints as the primary diagnostic lens, find the earliest step where a constraint is violated in a way that could cause final task failure.
Return the responsible agent and exact step.
If several steps violate constraints, choose the earliest unrecovered root cause, not the clearest later symptom.
Do not choose a harmless failed attempt if it was later corrected.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Do not output confidence, likelihood scores, or any numeric scoring field.

Return JSON only with keys:
step, agent, violated_constraint, violation_type, reason.
"""


def cgv_step_validation_prompt(
    case: Case,
    constraints: list[dict],
    current_step: LogStep,
    evidence_steps: list[LogStep],
    evidence_label: str,
) -> str:
    return f"""You are performing constraint-guided step validation for a failed multi-agent task.

{header(case)}

Constraints:
{constraints}

Evidence Trace:
{evidence_label}
{render_steps(evidence_steps)}

Current Step Under Review:
{render_steps([current_step])}

Task:
Evaluate only the Current Step Under Review against the constraints.
Decide whether this step violates a task-success constraint in a way that could contribute to the final failure.
Use the Evidence Trace only to judge causality, recovery, and downstream symptoms.
Do not mark a later downstream symptom as the root violation if an earlier visible violation caused it.
Do not mark harmless failed attempts as root violations if they were later corrected.
When the current step looks like a violation, judge whether it reflects a hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation, and whether a minimal correction at this step could plausibly change the final wrong answer.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{{" and the last character must be "}}".
Return keys:
step, agent, trigger_applies, verdict, violated_constraints, violation_type,
severity, causal_relevance, recoverability, confidence, evidence, reason.

Rules:
- step must be {current_step.step}.
- agent must be "{current_step.agent}" unless the current step explicitly contains a different acting system agent.
- verdict must be one of SKIP, SAT, VIOL.
- Use VIOL only when the current step breaks a constraint with grounded evidence and causal relevance to the final failure.
- recoverability must be one of recovered, unrecovered, unclear, or not_applicable.
- severity, causal_relevance, and confidence must be between 0 and 1.
- If verdict is not VIOL, use violated_constraints = [] and keep severity and causal_relevance low.
"""


def cgv_final_judge_prompt(
    case: Case,
    constraints: list[dict],
    evidence_steps: list[LogStep],
    validation_log: list[dict[str, Any]],
    evidence_label: str,
) -> str:
    return f"""You are the final judge in a constraint-guided failure localization system.

{header(case)}

Constraints:
{constraints}

Evidence Trace:
{evidence_label}
{render_steps(evidence_steps)}

Validation Log:
{validation_log}

Task:
Use the Evidence Trace and Validation Log to identify the earliest unrecovered root-cause failure step.
The Validation Log is evidence, not ground truth. It may contain false positives or miss relevant failures.
Prefer the first agent action that violated a task-success constraint and was not later recovered.
Do not choose a later downstream symptom if an earlier root-cause violation is available.
Do not choose a harmless failed attempt if it was later corrected.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

Decision procedure:
1. Review candidate violations in temporal order.
2. Ignore violations that were clearly recovered before the final answer.
3. For plausible candidates, check the hidden cause, minimal corrected action, and likely outcome change.
4. Select the earliest unrecovered violation whose minimal correction would plausibly change the final failure into success.
5. If the validation log is empty or unhelpful, use the Evidence Trace directly and still choose the earliest root-cause system-agent step.

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{{" and the last character must be "}}".
Return keys:
step, agent, violated_constraint, violation_type, confidence, reason.
step must be an exact step number from the Evidence Trace.
"""


def ccv_beam_rerank_prompt(
    case: Case,
    constraints: list[dict],
    candidates: list[dict],
) -> str:
    return f"""You are an AI assistant performing failure analysis on a multi-agent conversation.
Multiple agents collaborated to solve a problem but produced an incorrect solution.
You are choosing among candidate constraint-violation failure steps.

{header(case)}

Constraints:
{constraints}

Candidate Failure Steps:
{candidates}

Task:
Select the candidate that is most likely to be the earliest root-cause constraint violation
that caused the final task failure.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.

RERANKING CRITERIA:
- Prefer an earlier unrecovered root cause over a later high-severity symptom.
- Prefer candidates whose minimal corrected action would plausibly change the final answer, even if a later symptom looks more explicit.
- Do not choose a harmless failed attempt if it was later corrected.
- When candidates are close, prefer the one whose reason best explains the hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation behind the violation.

OUTPUT FORMAT:
Return exactly one valid JSON object. Do not use markdown fences, prose, bullet lists, or extra commentary.
The first character of your response must be "{{" and the last character must be "}}".

{{
  "candidate_id": <integer candidate id>,
  "step": <exact step number from the selected candidate>,
  "agent": "<agent name from the selected candidate>",
  "confidence": <number from 0.0 to 1.0>,
  "reason": "<brief reason for selecting this candidate>"
}}
"""


def ccv_beam_rerank_bool_prompt(
    case: Case,
    constraints: list[dict],
    candidates: list[dict],
) -> str:
    return f"""You are choosing among candidate constraint-violation failure steps.

{header(case)}

Constraints:
{constraints}

Candidate Failure Steps:
{candidates}

Task:
Select the candidate that is most likely to be the earliest root-cause constraint violation
that caused the final task failure.
Prefer an earlier unrecovered root cause over a later symptom.
Prefer candidates whose minimal corrected action would plausibly change the final answer, even if a later symptom looks more explicit.
Do not choose a harmless failed attempt if it was later corrected.
When candidates are close, prefer the one whose reason best explains the hidden misunderstanding, missing evidence, wrong assumption, or tool-output misinterpretation behind the violation.
Do not attribute the failure to the human/user/problem statement. Choose an action taken by the multi-agent system.
Do not output confidence, likelihood scores, or any numeric scoring field.

Return JSON only with keys:
candidate_id, step, agent, reason.
step must be the exact step number from the selected candidate.
"""


def agentrx_constraint_prompt(case: Case) -> str:
    gt_note = "Use only the task, failed final answer, and trajectory text. Do not use gold labels or ground-truth answers."
    return f"""You are implementing an AGENTRX-like diagnostic baseline for a failed AI agent trajectory.

{agentrx_header(case)}

Task:
Synthesize trajectory-level correctness constraints that can be checked against execution steps.
{gt_note}
Include constraints about instruction/plan adherence, tool output interpretation, evidence grounding,
information invention, tool invocation validity, and final-answer consistency when relevant.

Return JSON only with key constraints.
constraints should be a list of objects with id, type, description, violation_criteria.
"""


def agentrx_validation_prompt(
    case: Case,
    constraints: list[dict],
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are validating a chunk of a failed AI agent execution trajectory using synthesized constraints.

{agentrx_header(case)}

Constraints:
{constraints}

The full log has {len(case.steps)} steps and is divided into {chunk_count} chunks.
You are now validating Chunk {chunk_id}, covering steps {chunk[0].step}-{chunk[-1].step}.

Previous Chunk Summary:
{prev_summary}

Chunk Under Validation:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Task:
Produce an auditable validation log of constraint violations in this chunk.
Only include violations with concrete evidence in the chunk.
For each violation, identify the exact step, responsible agent, violated constraint,
failure category, whether it appears recoverable downstream, and a short evidence quote.
Prefer root-cause violations over later symptoms, but record all meaningful violations.
Do not attribute the failure to the human/user/problem statement.

Return JSON only with keys:
chunk_id, violations, chunk_summary.
violations should be a list of objects with keys:
step, agent, violated_constraint, failure_category, severity, confidence,
recoverable, evidence, reason.
severity and confidence must be between 0 and 1.
"""


def agentrx_judge_prompt(
    case: Case,
    constraints: list[dict],
    validation_log: list[dict],
    chunk_summaries: list[str],
) -> str:
    return f"""You are the final judge in an AGENTRX-like diagnostic framework.

{agentrx_header(case)}

Constraints:
{constraints}

Chunk Summaries:
{chunk_summaries}

Validation Log:
{validation_log}

Task:
Using only the validation log and summaries, identify the critical failure step and responsible agent.
The critical failure is the earliest unrecovered violation that plausibly caused the final task failure.
If several violations are present, avoid choosing a later symptom when an earlier unrecovered root cause exists.
Do not attribute the failure to the human/user/problem statement.

Return JSON only with keys:
step, agent, failure_category, confidence, reason.
step must be the exact step number from the trajectory.
"""


AGENTRX_TAXONOMY = """1. Instruction/Plan Adherence Failure: the agent fails to follow directions or the agreed plan.
2. Invention of New Information: the agent introduces, removes, or alters unsupported information.
3. Invalid Invocation: the agent makes an ill-formed tool/API call or invalid request.
4. Misinterpretation of Tool Output / Handoff Failure: the agent reasons incorrectly about tool or handoff output.
5. Intent-Plan Misalignment: the agent misunderstands the user's goal or constraints.
6. Underspecified User Intent: the run lacks required information and the agent fails because of that gap.
7. Intent Not Supported: the requested action is not supported by available tools/capabilities.
8. Guardrails Triggered: a policy/access/CAPTCHA/login/paywall block prevents execution.
9. System Failure: an infrastructure/connectivity/service failure blocks a tool or step.
10. Inconclusive: use sparingly when no category clearly applies."""


AGENTRX_TAXONOMY_CHECKLIST = """Use the taxonomy as a checklist, not just as labels:
1. Instruction/Plan Adherence Failure: Was a required plan, instruction, ordering, confirmation, or policy step skipped, reordered, or contradicted while enough information was available?
2. Invention of New Information: Did the agent introduce, omit, or alter a claim/value that was not supported by prior user text, context, or tool output, and did that unsupported claim affect the outcome?
3. Invalid Invocation: Did a concrete tool/API/query invocation fail because its arguments, syntax, schema, or request shape were invalid rather than because of external infrastructure?
4. Misinterpretation of Tool Output / Handoff Failure: Did the agent receive relevant tool or handoff output and then reason from it incorrectly, ignore a crucial part, or compute the wrong implication?
5. Intent-Plan Misalignment: Did the agent pursue the wrong objective or violate a key user constraint because it misunderstood the user's goal?
6. Underspecified User Intent: Was a necessary piece of information absent, and did the agent fail by proceeding without it or by not asking for it?
7. Intent Not Supported: Did the requested action require a capability or tool that was unavailable in the environment?
8. Guardrails Triggered: Was there an explicit policy/access block such as refusal, login, CAPTCHA, paywall, 403, robots.txt, or automation denial?
9. System Failure: Was the decisive problem an infrastructure/connectivity/service failure rather than agent reasoning or invalid arguments?
10. Inconclusive: Use only when no category above fits; provide a concrete residual failure description."""


def agentrx_official_judge_prompt(case: Case) -> str:
    return f"""You are an Expert Failure-Categorization Judge. You will be provided with a trajectory of an agent's interaction with a user.
Given: a full trajectory of an agent's conversation with the user (step-indexed).

YOUR TASK is determine why the agent failed, which failure category applies from the taxonomy below,
and exactly which step index the failure occurred at.

Failure taxonomy:
{AGENTRX_TAXONOMY}

Taxonomy checklist:
{AGENTRX_TAXONOMY_CHECKLIST}

How to Judge (Decision Procedure):
1. Step 1 - Locate the first failure: Scan the trajectory step-by-step from the start. The first step where the agent deviates from the intended plan or emits an error is the first failure. Record the step index and a short failure note.
2. Step 2 - Check if that failure was resolved: Look ahead in the trajectory for evidence that the error was resolved. If yes, mark it Resolved; if no such evidence, mark it Not resolved.
3. Step 3 - Decide and continue:
- If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
- If Not resolved: treat this step as the root-cause failure for the run and assign the taxonomy at this step.

{agentrx_header(case)}

Trajectory:
{render_steps(case.steps)}

Return JSON only with keys:
taxonomy_checklist_reasoning, reason_for_failure, failure_case, reason_for_index, index, agent.
index must be the exact step number from the trajectory.
agent should be the responsible agent at that step; if unsure, use the agent name shown at the selected step.
"""


def agentrx_original_global_constraints_prompt(case: Case) -> str:
    return f"""You are implementing the AGENTRX global constraint synthesis stage for a failed AI-agent execution.

AGENTRX global constraints are synthesized once from the task instruction, domain policy, and tool/agent schema.
This benchmark record does not provide an external domain policy or formal tool schema, so infer only stable,
domain-general obligations justified by the task text, observed agent/tool inventory, and trajectory format.
Do not inspect the full failed trajectory at this stage. Do not use gold failure labels or ground-truth answers.

{agentrx_header(case)}

Observed Tool/Agent Inventory:
{agentrx_tool_agent_inventory(case.steps)}

Task:
Synthesize global constraints that should hold throughout the trajectory.
Global constraints should encode stable rules such as instruction adherence, grounded use of observations,
valid tool invocation, consistency with tool outputs, handoff correctness, and final-answer grounding.
Each constraint must define when it is triggered and how to check it.
Return at most 2 global constraints. Prefer broad high-value nl_check constraints over many narrow constraints.
Keep every string field under 20 words. Keep examples compact; an empty object is allowed.

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{" and the last character must be "}".
constraints must be a list of objects with keys:
id, assertion_name, taxonomy_targets, constraint_type, event_trigger, check_hint, examples, check_type, python_check, nl_check.
constraint_type must be one of SCHEMA, PROTOCOL, RELATIONAL_POST, PROVENANCE, TEMPORAL, CAPABILITY, ANY.
event_trigger must include step_index, substep_index, role_name, content_regex, and tool_name fields using "*" when broad.
check_type must be "nl_check" unless a deterministic text-only python_check is clearly possible.
python_check should be an object, empty when check_type is nl_check.
nl_check should include judge_scope_notes, focus_steps_instruction, judge_rubric, and output_format_template.
taxonomy_targets should use labels from the AGENTRX taxonomy.
Required top-level shape:
{{"constraints": [{{"id": "G1", "assertion_name": "...", "taxonomy_targets": ["..."], "constraint_type": "PROVENANCE", "event_trigger": {{"step_index": "*", "substep_index": "*", "role_name": "*", "content_regex": "*", "tool_name": "*"}}, "check_hint": "...", "examples": {{}}, "check_type": "nl_check", "python_check": {{}}, "nl_check": {{"judge_scope_notes": "...", "focus_steps_instruction": "...", "judge_rubric": "...", "output_format_template": "..."}}}}]}}
"""


def agentrx_original_dynamic_constraints_prompt(
    case: Case,
    global_constraints: list[dict],
    prefix: list[LogStep],
    current_step: LogStep,
) -> str:
    return f"""You are implementing the AGENTRX dynamic constraint synthesis stage.

AGENTRX creates dynamic constraints from the task instruction, the trajectory prefix up to the current step,
and the global constraint store. Dynamic constraints capture trajectory-specific obligations introduced by
earlier observations, tool outputs, assumptions, user requirements, and handoffs.
Do not use gold failure labels.

{agentrx_header(case)}

Global Constraint Store:
{global_constraints}

Trajectory Prefix T<=k:
{render_steps(prefix)}

Current Step sk:
{render_steps([current_step])}

Task:
Synthesize only constraints that become relevant because of this prefix and current step.
Prefer concrete obligations that can be checked against the current step and nearby evidence.
Return at most 1 dynamic constraint. If the current prefix adds no essential new obligation, return an empty list.
Keep every string field under 20 words. Keep examples compact; an empty object is allowed.

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{" and the last character must be "}".
constraints must be a list of objects with keys:
id, assertion_name, taxonomy_targets, constraint_type, event_trigger, check_hint, examples, check_type, python_check, nl_check.
constraint_type must be one of SCHEMA, PROTOCOL, RELATIONAL_POST, PROVENANCE, TEMPORAL, CAPABILITY, ANY.
event_trigger must include step_index, substep_index, role_name, content_regex, and tool_name fields using "*" when broad.
check_type must be "nl_check" unless a deterministic text-only python_check is clearly possible.
python_check should be an object, empty when check_type is nl_check.
nl_check should include judge_scope_notes, focus_steps_instruction, judge_rubric, and output_format_template.
If no new dynamic constraints are needed, return {{"constraints": []}}.
Required top-level shape:
{{"constraints": [{{"id": "D1", "assertion_name": "...", "taxonomy_targets": ["..."], "constraint_type": "TEMPORAL", "event_trigger": {{"step_index": "*", "substep_index": "*", "role_name": "*", "content_regex": "*", "tool_name": "*"}}, "check_hint": "...", "examples": {{}}, "check_type": "nl_check", "python_check": {{}}, "nl_check": {{"judge_scope_notes": "...", "focus_steps_instruction": "...", "judge_rubric": "...", "output_format_template": "..."}}}}]}}
"""


def agentrx_original_step_validation_prompt(
    case: Case,
    constraints: list[dict],
    prefix: list[LogStep],
    current_step: LogStep,
    window: list[LogStep],
) -> str:
    return f"""You are implementing the AGENTRX guarded constraint evaluation stage for one trajectory step.

For each constraint, first decide whether its event_trigger applies to the current step and prefix. If the trigger
does not apply, mark SKIP. If it applies, evaluate the python_check or nl_check using only the provided prefix,
current step, and context window. A violation must have concrete grounded evidence. Ambiguous or unsupported
accusations should not be marked as violations.

{agentrx_header(case)}

Applicable Constraint Store Ck:
{constraints}

Trajectory Prefix T<=k:
{render_steps(prefix)}

Current Step sk:
{render_steps([current_step])}

Context Window:
{render_steps(window)}

Task:
Produce the AGENTRX validation-log entries for violated constraints at this step.
Evaluate all provided constraints, but keep evidence and reasons compact. Include at most 3 violations.

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{" and the last character must be "}".
Return keys:
step_index, checks, violations.
checks should be a list of objects with keys:
constraint_id, assertion_name, trigger_applies, verdict, evidence, taxonomy_targets, reason.
verdict must be one of SKIP, SAT, VIOL.
violations should include only checks whose verdict is VIOL, with keys:
step, agent, assertion_name, constraint_type, check_type, severity, evidence, taxonomy_targets, reason.
Use step = {current_step.step} for any violation at this current step.
Required top-level shape:
{{"step_index": {current_step.step}, "checks": [], "violations": []}}
"""


def agentrx_original_judge_prompt(
    case: Case,
    validation_log: list[dict[str, Any]],
) -> str:
    return f"""You are an Expert Failure-Categorization Judge. You will be provided with a trajectory of an agent's interaction with a user.
Given: a full trajectory of an agent's conversation with the user (step-indexed).

YOUR TASK is determine why the agent failed, which failure category applies from the taxonomy below,
and exactly which step index the failure occurred at.

Failure taxonomy:
{AGENTRX_TAXONOMY}

Taxonomy checklist:
{AGENTRX_TAXONOMY_CHECKLIST}

You are also provided a list of violations generated through the trajectory by AGENTRX constraints.
Use these violations to help identify the root-cause category and failure step. Static/global constraints
represent stable policy/tool/task obligations. Dynamic constraints represent trajectory-specific obligations.
Each constraint returns a boolean-style verdict, and violated constraints include grounded evidence.
Some violations may be false positives and not all violations are relevant to the root cause failure.

Validation Log:
{validation_log}

How to Judge (Decision Procedure):
1. Locate the first failure: scan the trajectory step-by-step from the start. The first step where the agent deviates from the intended plan or emits an error is the first failure. Record the step index and a short failure note.
2. Check if that failure was resolved: look ahead in the trajectory for evidence that the error was resolved. If yes, mark it Resolved; if no such evidence, mark it Not resolved.
3. Decide and continue:
- If Resolved: continue scanning from the next step to find the next new failure, then repeat Step 2 for it.
- If Not resolved: treat this step as the root-cause failure for the run and assign the taxonomy at this step.

{agentrx_header(case)}

Trajectory:
{render_steps(case.steps)}

Return exactly one JSON object. Do not use markdown fences, prose, bullet lists, or commentary.
The first character of your response must be "{" and the last character must be "}".
Return keys:
taxonomy_checklist_reasoning, reason_for_failure, failure_case, reason_for_index, index.
index must be the exact step number from the trajectory.
Required top-level shape:
{{"taxonomy_checklist_reasoning": "...", "reason_for_failure": "...", "failure_case": 4, "reason_for_index": "...", "index": 0}}
"""


def agentrx_official_wrapper_judge_prompt(
    case: Case,
    selected_chunks: list[dict[str, Any]],
    validation_log: list[dict[str, Any]],
) -> str:
    rendered_chunks = []
    for item in selected_chunks:
        rendered_chunks.append(
            f"SELECTED CHUNK {item['chunk_id']} (steps {item['chunk'][0].step}-{item['chunk'][-1].step}):\n"
            f"{render_steps(item['chunk'])}"
        )
    return f"""You are an Expert Failure-Categorization Judge using the AGENTRX decision procedure.
You are given selected high-signal trajectory chunks and an auditable validation log of possible violations.

Failure taxonomy:
{AGENTRX_TAXONOMY}

Taxonomy checklist:
{AGENTRX_TAXONOMY_CHECKLIST}

Decision Procedure:
1. Locate the first failure candidate in the selected evidence.
2. Check whether the candidate was resolved later in the selected evidence or validation log.
3. If it was not resolved, treat it as the root-cause failure; if it was resolved, continue to the next candidate.
Prefer the first unrecovered root-cause failure over downstream symptoms.

{agentrx_header(case)}

Selected Evidence Chunks:
{chr(10).join(rendered_chunks)}

Validation Log:
{validation_log}

Return JSON only with keys:
reason_for_failure, failure_case, reason_for_index, index, agent, confidence.
index must be the exact step number from the selected chunks.
confidence must be between 0 and 1.
"""


ECHO_FOCUS_INSTRUCTIONS = {
    "conservative": "You are a conservative analyst with high confidence thresholds. Only attribute errors when you have strong, clear evidence.",
    "liberal": "You are a liberal analyst willing to make attributions based on reasonable evidence. Consider subtle and multi-agent causes.",
    "detail_focused": "You are detail-oriented. Focus on exact wording, fine-grained evidence, and concrete inconsistencies.",
    "pattern_focused": "You focus on broad reasoning patterns and how errors propagate through the conversation.",
    "skeptical": "You are skeptical. Challenge assumptions and consider alternative explanations.",
    "general": "You are a balanced general analyst considering all evidence equally.",
}


ECHO_ORIGINAL_FOCUS_INSTRUCTIONS = {
    "conservative": (
        "You are a conservative analyst with high confidence thresholds. Only attribute errors when you have "
        "strong, clear evidence. Prefer single-agent attributions over multi-agent ones. Be cautious about "
        "making attributions without definitive proof."
    ),
    "liberal": (
        "You are a liberal analyst more willing to make attributions based on reasonable evidence. Consider "
        "multi-agent scenarios and subtle errors that might be overlooked. Be open to making attributions "
        "even with moderate confidence."
    ),
    "detail_focused": (
        "You are detail-oriented and focus on specific evidence, exact wording, and fine-grained analysis. "
        "Look for subtle inconsistencies, minor logical gaps, and precise factual inaccuracies. Prioritize "
        "concrete evidence over general patterns."
    ),
    "pattern_focused": (
        "You are focused on recognizing broader patterns and systemic issues in reasoning chains. Look for "
        "recurring themes, logical flow problems, and how errors propagate through the conversation. Consider "
        "the overall reasoning structure."
    ),
    "skeptical": (
        "You are highly skeptical and question all assumptions. Look for alternative explanations, consider "
        "whether apparent errors might be valid reasoning, and examine if the ground truth itself could be "
        "questioned. Challenge conventional attributions."
    ),
    "general": (
        "You are a balanced general analyst with no specific specialization. Approach the analysis with broad "
        "perspective, considering all types of evidence equally. Look for the most obvious and impactful "
        "mistakes based on objective evaluation."
    ),
}


_ECHO_DECISION_KEYWORDS = (
    "answer",
    "conclude",
    "decide",
    "decision",
    "evidence",
    "final",
    "found",
    "incorrect",
    "missing",
    "plan",
    "result",
    "search",
    "select",
    "therefore",
    "tool",
    "verify",
    "wrong",
)


def _echo_clean_snippet(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _echo_key_decision(text: str, limit: int = 220) -> str:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    for chunk in chunks:
        lowered = chunk.lower()
        if any(keyword in lowered for keyword in _ECHO_DECISION_KEYWORDS):
            return _echo_clean_snippet(chunk, limit)
    return _echo_clean_snippet(text, limit)


def _echo_appendix_extract(agent_content: str, max_words: int, context_type: str) -> str:
    cleaned = re.sub(r"\s+", " ", agent_content or "").strip()
    if not cleaned:
        return "No content available"
    pattern_groups = {
        "handoff": [
            r"(?:received|got|obtained|from)\s+([^.!?]*[.!?])",
            r"(?:passing|providing|sending|to)\s+([^.!?]*[.!?])",
            r"(?:based on|using)\s+([^.!?]*[.!?])",
        ],
        "decision_quality": [
            r"(?:I (?:conclude|determine|decide|believe|think))\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The (?:answer|solution|result))\s+(?:is|appears)\s+([^.!?]*[.!?])",
            r"(?:Based on|Given)\s+([^.!?]*[.!?])",
        ],
        "error_propagation": [
            r"(?:error|mistake|wrong|incorrect|failed)\s+([^.!?]*[.!?])",
            r"(?:cannot|unable|couldn'?t|can'?t)\s+([^.!?]*[.!?])",
            r"(?:However|But|Unfortunately)\s+([^.!?]*[.!?])",
        ],
        "milestone": [
            r"(?:completed|finished|achieved|accomplished)\s+([^.!?]*[.!?])",
            r"(?:created|generated|produced|built)\s+([^.!?]*[.!?])",
            r"(?:successfully|finally)\s+([^.!?]*[.!?])",
        ],
        "general": [
            r"(?:I (?:will|should|need to|decided to|conclude that|believe|think|determine))\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The answer|The result|The solution)\s+(?:is|appears to be|seems to be)\s+([^.!?]*[.!?])",
            r"Let me\s+([^.!?]*[.!?])",
        ],
    }
    patterns = pattern_groups.get(context_type, pattern_groups["general"])
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        if matches:
            text = str(matches[0]).strip()
            words = text.split()[:max_words]
            return " ".join(words) + ("..." if len(text.split()) > max_words else "")
    first = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
    words = first.split()[:max_words]
    return " ".join(words) + ("..." if len(first.split()) > max_words else "")


def _echo_appendix_extract_key_decision(
    agent_content: str,
    max_words: int = 50,
    context_type: str = "decision_quality",
) -> str:
    if not agent_content.strip():
        return "No content available"
    if context_type == "handoff":
        patterns = [
            r"(?:received|got|obtained|from)\s+([^.!?]*[.!?])",
            r"(?:passing|providing|sending|to)\s+([^.!?]*[.!?])",
            r"(?:based on|using)\s+([^.!?]*[.!?])",
            r"(?:will|need to|should)\s+([^.!?]*(?:next|continue)[^.!?]*[.!?])",
        ]
    elif context_type == "decision_quality":
        patterns = [
            r"(?:I (?:conclude|determine|decide|believe|think))\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The (?:answer|solution|result))\s+(?:is|appears)\s+([^.!?]*[.!?])",
            r"(?:Based on|Given)\s+([^.!?]*[.!?])",
        ]
    elif context_type == "error_propagation":
        patterns = [
            r"(?:error|mistake|wrong|incorrect|failed)\s+([^.!?]*[.!?])",
            r"(?:cannot|unable|couldn'?t|can'?t)\s+([^.!?]*[.!?])",
            r"(?:However|But|Unfortunately)\s+([^.!?]*[.!?])",
        ]
    else:
        patterns = [
            r"(?:I (?:will|should|need to|decided to|conclude that|believe|think|determine))\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The answer|The result|The solution)\s+(?:is|appears to be|seems to be)\s+([^.!?]*[.!?])",
            r"Let me\s+([^.!?]*[.!?])",
            r"(?:My approach|My strategy|My plan)\s+(?:is|will be)\s+([^.!?]*[.!?])",
        ]
    for pattern in patterns:
        matches = re.findall(pattern, agent_content, re.IGNORECASE)
        if matches:
            decision = str(matches[0]).strip()
            words = decision.split()[:max_words]
            return " ".join(words) + ("..." if len(decision.split()) > max_words else "")
    sentences = agent_content.split(". ")
    if sentences:
        first_sentence = sentences[0].strip()
        if not first_sentence.endswith("."):
            first_sentence += "."
        words = first_sentence.split()[:max_words]
        return " ".join(words) + ("..." if len(first_sentence.split()) > max_words else "")
    words = agent_content.split()[:max_words]
    return " ".join(words) + ("..." if len(agent_content.split()) > max_words else "")


def _echo_appendix_summarize_agent(
    agent_content: str,
    max_words: int = 20,
    context_type: str = "general",
) -> str:
    if not agent_content.strip():
        return "No content available"
    cleaned_content = " ".join(agent_content.split())
    if context_type == "handoff":
        patterns = [
            r"(?:received|got|obtained)\s+([^.!?]*[.!?])",
            r"(?:providing|sending)\s+([^.!?]*[.!?])",
        ]
    elif context_type == "decision_quality":
        patterns = [
            r"(?:conclude|determine|decide)\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So),?\s+([^.!?]*[.!?])",
        ]
    elif context_type == "error_propagation":
        patterns = [
            r"(?:error|mistake|failed)\s+([^.!?]*[.!?])",
            r"(?:cannot|unable)\s+([^.!?]*[.!?])",
        ]
    else:
        patterns = [
            r"(?:In conclusion|To conclude|Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The (?:answer|result|solution|output))\s+(?:is|appears to be|seems to be)\s+([^.!?]*[.!?])",
            r"(?:I (?:found|determined|concluded|calculated))\s+([^.!?]*[.!?])",
        ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
        if matches:
            summary = str(matches[0]).strip()
            words = summary.split()[:max_words]
            return " ".join(words) + ("..." if len(summary.split()) > max_words else "")
    sentences = cleaned_content.split(". ")
    if sentences:
        first_sentence = sentences[0].strip()
        words = first_sentence.split()[:max_words]
        return " ".join(words) + ("..." if len(first_sentence.split()) > max_words else "")
    words = cleaned_content.split()[:max_words]
    return " ".join(words) + ("..." if len(cleaned_content.split()) > max_words else "")


def _echo_appendix_obtain_milestones(
    agent_content: str,
    max_words: int = 15,
    context_type: str = "general",
) -> str:
    if not agent_content.strip():
        return "No milestones available"
    cleaned_content = " ".join(agent_content.split())
    if context_type == "handoff":
        patterns = [
            r"(?:received|obtained|got)\s+([^.!?]*(?:from|data|information)[^.!?]*[.!?])",
            r"(?:provided|sent|passed)\s+([^.!?]*(?:to|data|information)[^.!?]*[.!?])",
            r"(?:completed|finished)\s+([^.!?]*(?:handoff|transfer)[^.!?]*[.!?])",
        ]
    elif context_type == "decision_quality":
        patterns = [
            r"(?:decided|determined|concluded)\s+([^.!?]*[.!?])",
            r"(?:evaluated|assessed|analyzed)\s+([^.!?]*[.!?])",
            r"(?:final decision|ultimate choice)\s*[:-]?\s*([^.!?]*[.!?])",
        ]
    elif context_type == "error_propagation":
        patterns = [
            r"(?:error|mistake|failure)\s+(?:occurred|detected)\s+([^.!?]*[.!?])",
            r"(?:identified|found)\s+(?:error|issue|problem)\s+([^.!?]*[.!?])",
            r"(?:corrected|fixed|resolved)\s+([^.!?]*[.!?])",
        ]
    else:
        patterns = [
            r"(?:completed|finished|achieved|accomplished)\s+([^.!?]*[.!?])",
            r"(?:created|generated|produced|built)\s+([^.!?]*[.!?])",
            r"(?:step\s+\d+|phase\s+\d+|stage\s+\d+)\s*[:-]?\s*([^.!?]*[.!?])",
            r"(?:successfully|finally)\s+([^.!?]*[.!?])",
        ]
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_content, re.IGNORECASE)
        if matches:
            milestone = str(matches[0]).strip()
            words = milestone.split()[:max_words]
            return " ".join(words) + ("..." if len(milestone.split()) > max_words else "")
    sentences = cleaned_content.split(". ")
    if sentences:
        first_sentence = sentences[0].strip()
        words = first_sentence.split()[:max_words]
        return " ".join(words) + ("..." if len(first_sentence.split()) > max_words else "")
    words = cleaned_content.split()[:max_words]
    return " ".join(words) + ("..." if len(cleaned_content.split()) > max_words else "")


def _echo_appendix_contexts(steps: list[LogStep]) -> list[dict[str, Any]]:
    conversation = [
        {
            "index": idx,
            "name": step.agent,
            "role": step.agent,
            "content": step.content,
            "dataset_step": step.step,
        }
        for idx, step in enumerate(steps)
    ]
    contexts: list[dict[str, Any]] = []
    for current_idx, current_agent in enumerate(conversation):
        context = {
            "current_agent": current_agent,
            "context_levels": {
                "immediate": [],
                "nearby": [],
                "distant": [],
                "milestones": [],
            },
        }
        for idx, agent in enumerate(conversation):
            if idx == current_idx:
                continue
            distance = abs(idx - current_idx)
            agent_info = {
                "index": idx,
                "name": agent["name"],
                "role": agent["role"],
                "distance": distance,
                "dataset_step": agent["dataset_step"],
            }
            if distance == 1:
                agent_info["content"] = agent["content"]
                agent_info["detail_level"] = "full"
                context["context_levels"]["immediate"].append(agent_info)
            elif distance <= 3:
                agent_info["content"] = _echo_appendix_extract_key_decision(agent["content"])
                agent_info["detail_level"] = "key_decisions"
                context["context_levels"]["nearby"].append(agent_info)
            elif distance <= 6:
                agent_info["content"] = _echo_appendix_summarize_agent(agent["content"])
                agent_info["detail_level"] = "summary"
                context["context_levels"]["distant"].append(agent_info)
            else:
                agent_info["content"] = _echo_appendix_obtain_milestones(agent["content"])
                agent_info["detail_level"] = "milestones"
                context["context_levels"]["milestones"].append(agent_info)
        for level in context["context_levels"].values():
            level.sort(key=lambda item: int(item["index"]))
        contexts.append(context)
    return contexts


def _echo_appendix_format_context(context: dict[str, Any], limit_per_level: int = 8) -> str:
    current = context.get("current_agent", {})
    lines = [
        f"Current Step {current.get('index')} - {current.get('name')} ({current.get('role')}):",
        str(current.get("content", "")),
    ]
    levels = context.get("context_levels", {})
    for label in ("immediate", "nearby", "distant", "milestones"):
        items = levels.get(label, []) if isinstance(levels, dict) else []
        lines.append(f"{label.upper()} CONTEXT:")
        if not items:
            lines.append("None")
            continue
        for item in items[:limit_per_level]:
            lines.append(
                "Step {index} - {name} ({detail}, distance={distance}): {content}".format(
                    index=item.get("index"),
                    name=item.get("name"),
                    detail=item.get("detail_level"),
                    distance=item.get("distance"),
                    content=item.get("content"),
                )
            )
    return "\n".join(lines)


def echo_appendix_strict_conversation_summary(
    steps: list[LogStep],
    index_base: int = 0,
    target_agents: list[str] | None = None,
) -> str:
    contexts = _echo_appendix_contexts(steps)
    lines = ["=== CONVERSATION AGENTS ==="]
    for idx, step in enumerate(steps):
        display_step = idx + index_base
        lines.append(f"Step {display_step} - {step.agent} ({step.agent}):")
        lines.append(step.content)
        lines.append("")
    lines.append("=== HIERARCHICAL CONTEXT EXAMPLE ===")
    if contexts:
        sample_context = contexts[0]
        lines.append("Context structure for Agent 1 (showing hierarchical detail levels):")
        formatted = _echo_appendix_format_context(sample_context)
        lines.append(formatted[:1000] + "..." if len(formatted) > 1000 else formatted)
    return "\n".join(lines)


def echo_appendix_conversation_summary(
    steps: list[LogStep],
    index_base: int = 0,
    target_agents: list[str] | None = None,
) -> str:
    contexts = _echo_appendix_contexts(steps)
    lines = ["=== CONVERSATION AGENTS ==="]
    for idx, step in enumerate(steps):
        display_step = idx + index_base
        lines.append(f"Step {display_step} - {step.agent} ({step.agent}):")
        lines.append(step.content)
        lines.append("")
    lines.append("=== HIERARCHICAL CONTEXT EXAMPLE ===")
    selected_contexts = contexts
    if target_agents:
        target_norm = {agent.strip().lower() for agent in target_agents}
        selected_contexts = [
            context
            for context in contexts
            if str(context.get("current_agent", {}).get("name", "")).strip().lower() in target_norm
        ] or contexts
        lines.append(f"Target-agent context focus: {', '.join(target_agents)}")
    if selected_contexts:
        lines.append("Context structure for representative focal steps:")
        for context in selected_contexts[:3]:
            lines.append(_echo_appendix_format_context(context))
            lines.append("")
    return "\n".join(lines)


def _echo_agent_matches(agent: str, targets: list[str] | None) -> bool:
    if not targets:
        return True
    normalized = agent.strip().lower()
    return any(normalized == target.strip().lower() for target in targets)


def _echo_context_indices(steps: list[LogStep], target_agents: list[str] | None) -> list[int]:
    if not target_agents:
        return list(range(len(steps)))
    indices = [idx for idx, step in enumerate(steps) if _echo_agent_matches(step.agent, target_agents)]
    expanded: set[int] = set()
    for idx in indices:
        expanded.add(idx)
        if idx > 0:
            expanded.add(idx - 1)
        if idx + 1 < len(steps):
            expanded.add(idx + 1)
    return sorted(expanded)


def _echo_hierarchical_context_for_step(steps: list[LogStep], idx: int, index_base: int) -> list[str]:
    current = steps[idx]
    lines = [f"Context for Step {idx + index_base} - {current.agent}:"]

    immediate: list[str] = []
    for neighbor in (idx - 1, idx + 1):
        if 0 <= neighbor < len(steps):
            step = steps[neighbor]
            immediate.append(
                f"Step {neighbor + index_base} - {step.agent}: {_echo_clean_snippet(step.content, 180)}"
            )
    lines.append("  L1 immediate full context: " + (" | ".join(immediate) if immediate else "None"))

    nearby: list[str] = []
    for neighbor in range(max(0, idx - 3), min(len(steps), idx + 4)):
        if neighbor == idx or abs(neighbor - idx) <= 1:
            continue
        step = steps[neighbor]
        nearby.append(
            f"Step {neighbor + index_base} - {step.agent}: {_echo_key_decision(step.content, 160)}"
        )
    lines.append("  L2 nearby key decisions: " + (" | ".join(nearby) if nearby else "None"))

    distant: list[str] = []
    for neighbor, step in enumerate(steps):
        if abs(neighbor - idx) <= 3:
            continue
        snippet = _echo_key_decision(step.content, 130)
        lowered = snippet.lower()
        if any(keyword in lowered for keyword in ("final", "answer", "tool", "result", "search", "decide")):
            distant.append(f"Step {neighbor + index_base} - {step.agent}: {snippet}")
        if len(distant) >= 6:
            break
    lines.append("  L3 distant summaries / milestones: " + (" | ".join(distant) if distant else "None"))
    return lines


def echo_original_conversation_summary(
    steps: list[LogStep],
    index_base: int = 0,
    target_agents: list[str] | None = None,
) -> str:
    lines = ["=== CONVERSATION AGENTS ==="]
    lines.append("The step indices below are ECHO-style 0-based conversation indices for this provided trace segment.")
    for idx, step in enumerate(steps):
        display_step = idx + index_base
        lines.append(f"Step {display_step} - {step.agent} ({step.agent}):")
        lines.append(step.content)
        lines.append("")
    lines.append("=== HIERARCHICAL CONTEXT REPRESENTATION ===")
    lines.append(
        "Each focal step receives L1 adjacent context, L2 nearby key decisions, and L3 distant summaries/milestones."
    )
    if target_agents:
        lines.append(f"Target-agent context focus: {', '.join(target_agents)}")
    for idx in _echo_context_indices(steps, target_agents):
        lines.extend(_echo_hierarchical_context_for_step(steps, idx, index_base))
    return "\n".join(lines)


def echo_original_objective_analysis_prompt(
    case: Case,
    conversation: str,
    analyst_focus: str,
    phase: str = "full",
    target_agents: list[str] | None = None,
) -> str:
    focus = ECHO_ORIGINAL_FOCUS_INSTRUCTIONS.get(analyst_focus, ECHO_ORIGINAL_FOCUS_INSTRUCTIONS["general"])
    ground_truth_section = f"Ground Truth: {case.ground_truth or 'Not provided'}"
    final_answer = case.final_answer or "Not provided"
    phase_note = ""
    if phase == "agent":
        phase_note = (
            "\nAnalysis focus: identify the most likely responsible agent(s); the step index can be provisional "
            "but should point to the earliest causal evidence when visible.\n"
        )
    elif phase == "step":
        phase_note = (
            "\nAnalysis focus: pinpoint the earliest causal mistake step. If target agent candidates are provided "
            f"({target_agents or []}), consider them carefully, unless the evidence clearly contradicts them.\n"
        )
    elif phase in {"chunk", "selected_reread"}:
        phase_note = (
            "\nAnalysis focus: analyze only the provided trace segment. If this segment lacks the decisive root cause, "
            "keep confidence low and avoid selecting downstream symptoms.\n"
        )
    return f"""You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.
ANALYST SPECIALIZATION: {focus}
Your task:
1. Analyze ALL agents in the conversation objectively (not just one specific agent)
2. Determine which agent(s) most likely caused the final wrong answer
3. Determine which step/turn in the conversation the mistake occurred
4. Provide confidence scores and reasoning for your conclusions
{phase_note}
You have access to hierarchical context showing:
- Immediate agents: Full details
- Nearby agents: Key decisions
- Distant agents: Brief summaries
The agents are numbered sequentially (Agent 1, Agent 2, etc.) corresponding to their step/turn index in the conversation.
Possible conclusions:
- Single agent error: One specific agent caused the mistake at a specific step
- Multi-agent error: Multiple agents contributed to the mistake across specific steps
Output your response as valid JSON wrapped in <json></json> tags:
<json>
{{
  "analysis_summary": "Brief overview of your investigation approach and findings",
    "agent_evaluations": [
    {{
      "agent_name": "agent_name",
      "step_index": 0,
      "error_likelihood": 0.0,
      "reasoning": "Why this agent may or may not have caused the error",
      "evidence": "Specific evidence supporting your assessment"
    }}
  ],
  "primary_conclusion": {{
    "type": "single_agent",
    "attribution": ["agent_name(s)"],
    "mistake_step": 0,
    "confidence": 0.0,
    "reasoning": "Detailed explanation of your primary conclusion including which step the error occurred"
  }},
  "alternative_hypotheses": [
    {{
      "type": "conclusion_type",
      "attribution": ["agent_name(s)"],
      "mistake_step": 0,
      "confidence": 0.0,
      "reasoning": "Alternative explanation"
    }}
  ]
}}
</json>
Be thorough, objective, and consider all possibilities including that no single agent may be clearly at fault.
Pay special attention to identifying the specific step/turn where the error occurred.

Original Query: {case.problem}
{ground_truth_section}
Final Answer: {final_answer}
Conversation Analysis:
{conversation}
Please conduct an objective analysis of this conversation to determine error attribution.
Focus on identifying which specific agent(s) caused the error that led to the incorrect final answer.
Output your analysis in the JSON format specified in your instructions.
"""


def echo_appendix_strict_objective_analysis_prompt(
    case: Case,
    conversation: str,
    analyst_focus: str,
) -> str:
    focus = ECHO_ORIGINAL_FOCUS_INSTRUCTIONS.get(analyst_focus, ECHO_ORIGINAL_FOCUS_INSTRUCTIONS["general"])
    ground_truth_section = f"Ground Truth: {case.ground_truth or 'Not provided'}"
    final_answer = case.final_answer or "Not provided"
    return f"""You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.
ANALYST SPECIALIZATION: {focus}
Your task:
1. Analyze ALL agents in the conversation objectively (not just one specific agent)
2. Determine which agent(s) most likely caused the final wrong answer
3. Determine which step/turn in the conversation the mistake occurred
4. Provide confidence scores and reasoning for your conclusions

You have access to hierarchical context showing:
- Immediate agents: Full details
- Nearby agents: Key decisions
- Distant agents: Brief summaries
The agents are numbered sequentially (Agent 1, Agent 2, etc.) corresponding to their step/turn index in the conversation.
Possible conclusions:
- Single agent error: One specific agent caused the mistake at a specific step
- Multi-agent error: Multiple agents contributed to the mistake across specific steps
Output your response as valid JSON wrapped in <json></json> tags:
<json>
{{
  "analysis_summary": "Brief overview of your investigation approach and findings",
  "agent_evaluations": [
    {{
      "agent_name": "agent_name",
      "step_index": 0,
      "error_likelihood": 0.0,
      "reasoning": "Why this agent may or may not have caused the error",
      "evidence": "Specific evidence supporting your assessment"
    }}
  ],
  "primary_conclusion": {{
    "type": "single_agent",
    "attribution": ["agent_name(s)"],
    "mistake_step": 0,
    "confidence": 0.0,
    "reasoning": "Detailed explanation of your primary conclusion including which step the error occurred"
  }},
  "alternative_hypotheses": [
    {{
      "type": "conclusion_type",
      "attribution": ["agent_name(s)"],
      "mistake_step": 0,
      "confidence": 0.0,
      "reasoning": "Alternative explanation"
    }}
  ]
}}
</json>
Be thorough, objective, and consider all possibilities including that no single agent may be clearly at fault.
Pay special attention to identifying the specific step/turn where the error occurred.

Original Query: {case.problem}
{ground_truth_section}
Final Answer: {final_answer}
Conversation Analysis:
{conversation}
Please conduct an objective analysis of this conversation to determine error attribution.
Focus on identifying which specific agent(s) caused the error that led to the incorrect final answer.
Output your analysis in the JSON format specified in your instructions.
"""


def echo_original_global_chunk_router_prompt(
    case: Case,
    conversation: str,
    chunk_ranges: list[dict[str, Any]],
    beam_k: int,
    analyst_focus: str = "general",
) -> str:
    chunk_lines: list[str] = []
    for item in chunk_ranges:
        estimated_tokens = item.get("estimated_tokens")
        token_note = f", estimated_tokens={estimated_tokens}" if estimated_tokens is not None else ""
        chunk_lines.append(
            "Chunk {chunk_id}: steps {start_step}-{end_step}, step_count={step_count}{token_note}".format(
                chunk_id=item.get("chunk_id"),
                start_step=item.get("start_step"),
                end_step=item.get("end_step"),
                step_count=item.get("step_count"),
                token_note=token_note,
            )
        )
    chunk_table = "\n".join(chunk_lines)
    ground_truth_section = f"Ground Truth: {case.ground_truth or 'Not provided'}"
    final_answer = case.final_answer or "Not provided"
    focus = ECHO_ORIGINAL_FOCUS_INSTRUCTIONS.get(analyst_focus, ECHO_ORIGINAL_FOCUS_INSTRUCTIONS["general"])
    return f"""You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.
ANALYST SPECIALIZATION: {focus}

Your task is NOT to produce the final agent/step attribution.
Your task is to read the whole trace once, inspect the chunk table, and select the top {beam_k} chunk IDs most likely to contain the earliest decisive causal error that led to the final wrong answer.

Chunk selection objective:
- Select the top {beam_k} chunk IDs most likely to contain the earliest decisive causal error that led to the final wrong answer.
- Prefer an early root cause over later downstream symptoms.
- Do not over-select chunks merely because they contain the final failure, repeated symptoms, or confident-looking but late consequences.
- If the decisive error could be in several nearby chunks, keep those chunks in the beam.
- Use only chunk IDs from the table.

Return valid JSON wrapped in <json></json> tags:
<json>
{{
  "selected_chunk_ids": [1, 2, 3],
  "rationale": "Brief reason for the selected chunk beam",
  "candidates": [
    {{
      "chunk_id": 1,
      "reason": "Why this chunk may contain the earliest decisive causal error"
    }}
  ]
}}
</json>

Rules:
- selected_chunk_ids must contain at most {beam_k} unique IDs.
- Treat selected_chunk_ids as an approval set: include every chunk that plausibly contains the earliest decisive causal error.
- The aggregation step will use chunk membership across analysts, not raw confidence or fine-grained rank.
- candidates may include more chunks than selected_chunk_ids, but candidates are explanatory only.
- Do not output final attribution fields such as final agent or final step; this is only chunk routing.

Original Query: {case.problem}
{ground_truth_section}
Final Answer: {final_answer}

Chunk table:
{chunk_table}

Full conversation with 0-based step indices:
{conversation}
"""


def echo_objective_analysis_prompt(
    case: Case,
    conversation: str,
    analyst_focus: str,
    phase: str,
    target_agents: list[str] | None = None,
) -> str:
    focus = ECHO_FOCUS_INSTRUCTIONS.get(analyst_focus, ECHO_FOCUS_INSTRUCTIONS["general"])
    target_note = ""
    if target_agents:
        target_note = f"\nTarget responsible agent candidates from phase 1: {target_agents}\nFocus the step-level attribution on these agents unless the evidence clearly contradicts them.\n"
    phase_note = (
        "Phase: agent-level attribution. Identify the responsible agent or agents first."
        if phase == "agent"
        else "Phase: step-level attribution. Pinpoint the exact error step for the responsible agent(s)."
    )
    return f"""You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.
ANALYST SPECIALIZATION: {focus}

Your task:
1. Analyze all agents objectively.
2. Determine which agent(s) most likely caused the final wrong answer.
3. Determine which step/turn in the conversation the mistake occurred.
4. Provide confidence scores and reasoning for your conclusions.

You have access to hierarchical context showing the conversation in step order. The steps are numbered sequentially and correspond to the step/turn index.
Possible conclusions:
- single_agent: one specific agent caused the mistake at a specific step.
- multi_agent: multiple agents contributed across specific steps.

{phase_note}
{target_note}

{header(case)}

Conversation Analysis:
{conversation}

Output your response as valid JSON wrapped in <json></json> tags:
<json>
{{
  "analysis_summary": "Brief overview of your investigation approach and findings",
  "agent_evaluations": [
    {{
      "agent_name": "agent_name",
      "step_index": 0,
      "error_likelihood": 0.0,
      "reasoning": "Why this agent may or may not have caused the error",
      "evidence": "Specific evidence supporting your assessment"
    }}
  ],
  "primary_conclusion": {{
    "type": "single_agent",
    "attribution": ["agent_name"],
    "mistake_step": 0,
    "confidence": 0.0,
    "reasoning": "Detailed explanation of the primary conclusion"
  }},
  "alternative_hypotheses": [
    {{
      "type": "single_agent",
      "attribution": ["agent_name"],
      "mistake_step": 0,
      "confidence": 0.0,
      "reasoning": "Alternative explanation"
    }}
  ]
}}
</json>
Be thorough, objective, and pay special attention to the exact step where the error occurred.
"""


def echo_chunk_ranking_prompt(
    case: Case,
    chunk_id: int,
    chunk_count: int,
    chunk: list[LogStep],
    prev_summary: str,
    next_summary: str,
) -> str:
    return f"""You are applying ECHO-style objective analysis as a first-pass chunk selector.
Your task is not final attribution. Your task is to decide whether this chunk contains evidence useful for final error attribution.

{header(case)}

Full trace allocation:
- Total steps: {len(case.steps)}
- Chunk count: {chunk_count}
- Current chunk: {chunk_id}, steps {chunk[0].step}-{chunk[-1].step}

Previous Chunk Summary:
{prev_summary}

Current Chunk:
{render_steps(chunk)}

Next Chunk Summary:
{next_summary}

Analyze objectively:
1. Which agent in this chunk is most likely related to the final wrong answer?
2. Which step is the strongest local candidate?
3. How strong is this chunk as evidence for final attribution?

Return JSON only with keys:
chunk_id, contains_attribution_evidence, agent, step, confidence, reasoning.
confidence must be between 0 and 1.
If the chunk is not useful, set contains_attribution_evidence=false, agent="NONE", step=-1, confidence=0.
"""
