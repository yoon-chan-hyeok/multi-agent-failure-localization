# Method

## Problem definition

Given a task `Q` and a failed multi-agent trajectory
`L = {l_1, ..., l_n}`, predict:

- the system agent responsible for the decisive failure;
- the exact global step where the earliest unrecovered error occurred.

The evaluator reports agent accuracy and exact-step accuracy separately. It also
supports `±3`, `±5`, and mean absolute step distance.

## TSR-Loc

### Stage 1: pre-trace task interpretation

The requirement compiler receives the task description and allowed agent-name
vocabulary. In the No-GT condition, it does not receive:

- the execution trajectory;
- the reference answer;
- a separately stored final-system answer;
- the gold failure agent or step.

It produces checkable natural-language conditions for successful task completion.
The list is frozen before localization.

### Stage 2: failure localization

The localizer receives the frozen requirements and the step-indexed full trajectory.
It inspects the execution in temporal order, rejects failed attempts that were later
recovered, and selects the earliest remaining error whose minimal correction could
change the outcome.

The strict output coordinate is the original global `(agent, step)` pair.

## Worked example

```text
Task
  Find the value from the authoritative source, convert the unit, and report it.

Compiled success requirements
  R1. Use evidence from an authoritative source.
  R2. Extract the quantity associated with the requested object.
  R3. Apply the correct unit conversion.
  R4. Return the converted value in the requested format.

Trace excerpt
  Step 1 Planner: Decomposes the task.
  Step 2 WebSurfer: Uses an unrelated forum as the source.
  Step 3 Assistant: Extracts a value from that source.
  Step 4 Calculator: Converts the propagated value.
  Step 5 Assistant: Returns the wrong final value.

Attribution
  Agent = WebSurfer
  Step = 2
  Reason = Step 2 is the earliest unrecovered violation of R1; later arithmetic
           propagates the unsupported evidence and is a downstream consequence.
```

![Worked example](../assets/tsr_loc_worked_example_academic.png)

## Multi-View Beam Log Search

`mvbs10` is an earlier custom exploratory method retained in the codebase.

1. split the trajectory into ten contiguous chunks;
2. score every chunk using four views: forward onset, backward causality,
   agent-conditioned suspicion, and answer contrast;
3. retain the top three chunks and expand them by two neighboring steps;
4. localize candidates over five-step windows with stride two;
5. compare candidate pairs with the LLM and choose the candidate with the most wins.

This is a complete implemented pipeline, but the four views are fields in one prompt,
not four independent traversals or four independent models. `answer_contrast` is also
not a separate final-answer pass.

## Chunking utilities

The repository retains fixed and adaptive chunking utilities from the exploratory
phase:

- balanced fixed-count chunks;
- whole-step token-budget chunks;
- short-trace bypass;
- token, step, or combined allocation criteria;
- fixed or fractional top-k beams;
- selected-chunk rereading.

The token estimate used by these experiments is a whitespace proxy unless a backend
reports native token usage. Chunking did not yield universal gains and is not the
central claim of TSR-Loc.

