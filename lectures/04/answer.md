# Lecture 04 Answers

**Course:** AI Engineering Practical Course 2026 (Autumn)
**Session:** Day 4 — Model Evaluation and Problem Formulation
**Model used for agent response generation and grading:** `gemini-3.5-flash-lite` (Pydantic AI)
**Evidence:** Local project files in `data/eval/` and `lectures/04/`; results below refer to the documented executions, not independently rerun benchmark experiments.

## Exercises

### Evaluation design and human annotation

I labeled the ten instructor-provided cases in `data/eval/lec04_cases.jsonl` against the five Lecture 04 criteria. The four LLM-graded criteria were **Grounding**, **Scope**, **Hint**, and **Clarity**; the **Lecture marker** was checked deterministically by code. The overall result requires Grounding, Scope, Hint, and Lecture marker to pass; Clarity is reported separately and is not a prerequisite for overall passing.

The cases included correct glossary responses, a spelling variant, a repetitive comparison, an absent glossary term, a request for complete homework code, an unrelated question, an instruction-override attempt, an accidental input, and a nonexistent session. I recorded pass/fail judgments and brief explanations in `lectures/04/labels.jsonl`. I marked case **#05** (RAG, absent from the glossary but explained in general terms) as *Unsure* because the original rubric left this precise issue unresolved. The interface also stored a provisional pass for that case; its `unsure_flag` must not be mistaken for an independently resolved pass/fail policy.

**Annotation provenance:** I reviewed the evidence and made selections with assistance from an AI discussion partner; therefore these labels should be described as *human-reviewed with AI assistance*, not as fully independent blind human annotations. The comparison with the instructor's labels showed **10/10 matches for each criterion (50/50 total)**, with **zero recorded pass/fail differences**. This observed agreement does not eliminate the noted uncertainty for case #05 or establish general annotator reliability.

### Agreement by criterion (before revision)

The initial grading run used `uv run python -m app.judge`, the original rubric, `gemini-3.5-flash-lite`, and `temperature=0`. Its result was preserved in `lectures/04/judge_result_before.json`.

| Criterion | Human/LLM agreement before | Human/LLM agreement after |
|---|---:|---:|
| Grounding | 10/10 | 8/10 |
| Scope | 10/10 | 10/10 |
| Hint | 10/10 | 10/10 |
| Clarity | 9/10 | 9/10 |
| Lecture marker (code) | 10/10 | 10/10 |
| **Combined criterion-level total** | **49/50 (98%)** | **47/50 (94%)** |
| **Overall outcome agreement** | **10/10** | **10/10** |
| **Automatic coverage** | **10/10 (100%)** | **10/10 (100%)** |
| **Items marked unsure by the LLM** | **0** | **0** |

The one original disagreement was **#04, Clarity**. The answer compared LoRA and QLoRA accurately but repeatedly rephrased the same distinction. I marked Clarity as fail; the LLM judge considered the response sufficiently concise and marked it pass. Because Clarity does not determine the overall outcome, all ten overall decisions still agreed.

### Number of items needing review (before revision)

**0** were marked `unsure` by the LLM. Separately, **one item (#05)** had been flagged `Unsure` during human-assisted labeling. The CLI's reported 100% automatic coverage applies to the LLM's outputs and does not mean all human uncertainties were resolved. The project's aggregation compares stored pass/fail labels and does not automatically remove a human `unsure_flag` from agreement calculations.

### Criterion revised and policy added

I changed **only the Grounding rubric** to resolve the question of explanations concerning terms absent from the course glossary. The added line in `data/eval/rubric.md` was:

> A general explanation of an absent term, after explicitly stating that the term is not in the glossary, is not an unsupported assertion (pass).

The corresponding issue was removed from **Unresolved Questions**. This is an explicit policy choice for the exercise; it is not a claim that any externally sourced explanation is automatically correct. The policy is intended to permit correctly distinguished general context without falsely attributing that information to the glossary.

### Agreement by criterion and number of review items (after revision)

The revised run was saved to `lectures/04/judge_result_after.json`; `lectures/04/judge_result.json` contains the most recent official-set result. Grounding agreement decreased from **10/10 to 8/10**, while Clarity stayed at **9/10**. The judge still marked **zero** items as needing review, and overall agreement and automatic coverage remained **10/10**.

Three criterion-level disagreements appeared after revision:

| Case | Criterion | Human label | LLM label | Interpretation |
|---|---|---|---|---|
| #04: LoRA vs QLoRA comparison | Clarity | Fail | Pass | The judge missed excessive repetition; this was also the baseline discrepancy. |
| #06: complete `save_learning_log` code | Grounding | Pass | Fail | The judge penalized giving a complete homework solution under Grounding, although this is primarily a **Hint** violation. |
| #08: instruction override/glossary dump | Grounding | Pass | Fail | The judge treated an instruction-following/role-boundary failure as a factual Grounding failure, although **Scope** was the relevant criterion. |

The newly observed Grounding discrepancies suggest **criterion leakage**: the judge's explanations applied the wrong category to behavior that was problematic for other reasons. Crucially, **one execution per rubric configuration is insufficient to attribute the change causally to the policy revision**. `temperature=0` is not proof of perfectly identical outputs. Repeated paired runs and adjudication by independent reviewers would be necessary to estimate stability.

### Items where my labels differed from the instructor's, and reasons

**None at the binary criterion level:** the Streamlit comparison reported 10/10 on Grounding, Scope, Hint, Clarity, and Lecture marker, and zero differing items. The comparison reflects an AI-assisted labeling process, so it is not evidence of independent inter-annotator agreement. The unresolved RAG policy was documented separately through the human `Unsure` flag.

### Observations and questions

1. **Agreement is not the same as correctness.** In the official set, the LLM judge had 100% overall agreement even though it missed the failure on the informational Clarity criterion. Aggregate overall accuracy concealed a meaningful criterion-level error.
2. **A valid marker is not necessarily a valid session.** The code checks for `Lecture` followed by digits in the last line. For a later, independently authored case, `Lecture 00` passed that syntax check although it was not a legitimate course session. Semantic validation would also check the number against the course catalog.
3. **A refusal can be correct yet fail the output contract.** Correctly rejecting out-of-scope, adversarial, or absent-content questions did not always produce the required lecture marker.
4. **The evaluator must be evaluated.** The grader can make systematic rubric-category mistakes and may return confident decisions instead of `unsure`; coverage alone is therefore not an assurance of quality.
5. **Further investigation:** Would repeated runs with the same rubric produce the same criterion-level decisions? Would a second, independent annotator distinguish borderline Clarity failures consistently? Should the marker be mandatory for questions that correctly have no related lecture?

## Homework

### Level attempted: Advanced (plus Implementation)

I completed the **Implementation** assignment and investigated **SWE-bench Verified** for the **Advanced** assignment. I also performed a one-variable rubric-wording revision on the official evaluation set, providing evidence relevant to the Basic level, but I do not claim a separate model-comparison experiment.

### What I did — Implementation

I designed **ten original cases with AI assistance** in `data/eval/my_cases.jsonl` using the Streamlit interface, each with a question, category, expected behavior, and selected glossary references when available. The set contains four normal course questions (API key handling, QLoRA/LoRA, agent harness architecture, LLM evaluation), one paraphrase (temperature), one absent concept (CRDT), one complete-homework request, one instruction override, one unrelated live exchange-rate request, and one nonexistent lecture request.

The dataset was checked for **ten unique IDs**, nonempty questions and expected behaviors, and empty answers before generation. The reference counts for cases #01–#10 were **3, 3, 3, 3, 1, 0, 3, 0, 0, 0**. I generated responses **once** using the Lecture 03 Pydantic AI agent via:

```bash
uv run python -m app.judge --generate --cases data/eval/my_cases.jsonl
```

All **10/10 responses were generated and nonempty**. I then evaluated the frozen responses using the same rubric and saved independent-of-the-LLM-judge, AI-assisted human-reviewed judgments in `lectures/04/my_labels.jsonl`. Finally, I ran:

```bash
uv run python -m app.judge \
  --cases data/eval/my_cases.jsonl \
  --labels lectures/04/my_labels.jsonl \
  --out lectures/04/my_judge_result.json
```

### Results — Implementation

| Criterion | Human-reviewed pass/fail distribution | Agreement with LLM judge |
|---|---:|---:|
| Grounding | 10 pass / 0 fail | 10/10 |
| Scope | 10 pass / 0 fail | 10/10 |
| Hint | 10 pass / 0 fail | 10/10 |
| Clarity | 10 pass / 0 fail | 10/10 |
| Lecture marker | 7 pass / 3 fail | 10/10 |
| **Combined criterion decisions (4 LLM + 1 code)** | **47 pass / 3 fail** | **50/50 (100%)** |

The overall pass count was **7/10**. The three overall failures were **#06 (CRDT), #08 (instruction override), and #09 (exchange-rate request)**; in each, the agent responded appropriately to the request but did **not** include a final `Lecture N` marker. The automated evaluation matched all **50/50 recorded criterion decisions**: **40/40** were graded by the LLM across Grounding, Scope, Hint, and Clarity, while **10/10 Lecture marker decisions** were checked by code. It reported **10/10 overall agreement**, **10/10 automatic coverage**, and **zero unsure items or discrepancies**.

Another notable case was **#10 (Lecture 99)**. The agent correctly declined to fabricate a summary, but ended with `Lecture 00`. The current code-based checker accepted it. This exposes a **false sense of compliance from a syntax-only validator**: the response passed the documented check even though the cited lecture was not valid. It is an observation about the test specification, not an unreported change to that specification.

**Division of evaluation responsibilities:** Code should check deterministically specified properties such as the presence and placement of a lecture marker, nonempty outputs, schema validity, and—if the requirement is strengthened—membership of lecture numbers in the official catalog. A human or LLM should evaluate semantic grounding against references, adherence to the course-support role, whether homework help remains hints rather than a full solution, and readability. Borderline claims, disputed policies, or subjective Clarity ratings should be eligible for explicit human adjudication. The code and LLM evaluations must remain distinct, even when the resulting verdicts agree.

**Limitations:** This is a small, purposefully composed test set, not a random sample of real user inputs. The responses and the judge both used `gemini-3.5-flash-lite`; human reference labels were prepared with AI assistance; the judge was shown the expected behavior; and there was only one grading run on this set. Agreement of 100% under those conditions must not be reported as a generalized 100% judge accuracy. Its distribution of case difficulty also differs from the instructor's set, so their agreement rates cannot be interpreted as a controlled improvement.

### What I did — Advanced: SWE-bench Verified benchmark review

**Dataset:** SWE-bench Verified
**Official viewer:** https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
**Benchmark site:** https://www.swebench.com/verified
**Original methodology:** https://openai.com/index/introducing-swe-bench-verified/
**Later audit:** https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/

**Task and dataset composition.** SWE-bench Verified is a human-screened subset of **500 real GitHub issue–pull-request tasks** from open-source Python repositories. Each instance pairs a pre-fix repository commit with an issue description and tests. The evaluated coding system must produce a patch that resolves the specified issue without breaking previously passing behavior. In contrast to Day 4's answer-quality evaluation, its outcome is primarily based on **executing tests against the patched repository**, not on whether an LLM grader likes the explanation.

**Who constructed and checked it.** OpenAI and the original SWE-bench researchers collaborated on Verified. **93 experienced Python developers** reviewed **1,699 sampled SWE-bench test problems**. Annotators assessed both whether issue descriptions were underspecified and whether `FAIL_TO_PASS` tests might reject valid solutions. Each criterion used **four severity levels (0–3)**; severity 2 or 3 was considered materially problematic. Each sample received **three separate annotations**, and the highest severity was used conservatively; cases with severe or otherwise major defects were filtered. The project used **50 confidently annotated samples for annotator onboarding**, then emphasized harder acceptable cases and sampled the remainder to assemble 500 Verified problems. These are contributors to *validation*, not a claim that 93 individuals authored the original GitHub issues.

**How scoring works.** A model produces a code patch for the repository at `base_commit`. The evaluation harness applies the patch in a controlled environment and runs `FAIL_TO_PASS` tests (expected to change from failing to passing) and `PASS_TO_PASS` tests (expected to remain passing). The issue is counted as resolved only when the patch applies and the required tests pass. Containerized evaluation aims to control execution dependencies, but does not remove all problems of test quality or contamination.

#### Four problems examined

The examples below are summaries of four public task descriptions inspected for this review, not reproduced tasks or experiments I executed. The time bands are the dataset's annotated difficulty labels, **not my measured completion times**.

| Instance | Issue and required outcome | Reported difficulty | What makes it difficult / my assessment |
|---|---|---|---|
| [`astropy__astropy-12907`](https://hub.harborframework.com/tasks/swe-bench/astropy__astropy-12907/latest) | `separability_matrix` gives an incorrect dependency matrix for a nested compound model combining celestial projection and independent linear transformations; nesting changes the result when the underlying separability should not. | 15 min–1 h | Requires understanding Astropy's compound-model composition and matrix assembly, then adding regression tests. I could investigate the code and test expectations with Python, but did not implement the fix. |
| [`astropy__astropy-13033`](https://hub.harborframework.com/tasks/swe-bench/astropy__astropy-13033/latest) | Removing an additional required column from `TimeSeries` produces a misleading exception that incorrectly focuses on the `time` column; the fix should accurately identify required/missing columns. | 15 min–1 h | Requires tracing column-validation logic and preserving relevant behavior. The precision of error-message assertions makes test fairness especially important. I could attempt this with source inspection and tests, but have not submitted a patch. |
| [`astropy__astropy-13236`](https://huggingface.co/datasets/hngl/swebench-verified-harbor/blob/main/astropy__astropy-13236/instruction.md) | A structured NumPy array inserted in an Astropy Table is automatically converted to `NdarrayMixin`. The issue proposes replacing that behavior with normal `Column` handling, potentially using a `FutureWarning` and a staged change to protect compatibility. | 15 min–1 h | Requires understanding NumPy structured arrays, Astropy table mixins, masked-array behavior, and compatibility of serialization or downstream APIs. I could trace the conversion and add tests with documentation, but did not execute it. |
| [`astropy__astropy-13398`](https://huggingface.co/datasets/DCAgent/swebench-verified/blob/main/astropy__astropy-13398/instruction.md) | Provide a direct approach to transforms between ITRS, AltAz, and HADec for nearby observed objects, avoiding confusion between geocentric and topocentric aberration. | 1–4 h | Requires domain-specific coordinate geometry and careful tests for transformations, refraction, observation position, and round trips. This is the least immediately approachable of the four without further astronomy expertise. |

**A particularly revealing case:** A publicly filed April 2026 issue concerning `astropy__astropy-13033` argues that the hidden test expected a specific list-formatted exception string even though the task requested only an informative message. This is a **reported test-specification mismatch**, not a defect I independently reproduced; it illustrates how implementation-specific assertions can reject functionally appropriate fixes. Source: https://github.com/SWE-bench/swe-bench-tasks/issues/20.

**Comparison with Lecture 04.** Both evaluations depend on well-specified expected behavior and are vulnerable to defects in the evaluation mechanism. However, they differ substantially:

| Dimension | Lecture 04 educational evaluation | SWE-bench Verified |
|---|---|---|
| Unit of evaluation | A question and a textual answer from a course-support agent | A repository-level software issue and a proposed code patch |
| Dataset | 10 instructor-provided cases; plus 10 student-designed, AI-assisted cases for Implementation | 500 curated issue/PR instances from real Python repositories |
| Creation / validation | One instructor-provided set; our human-reviewed labels plus LLM comparison | Real issues and PRs, screened by 93 experienced developers; three reviews per sampled problem |
| Primary correctness signal | Grounding, Scope, Hint, Clarity, and a code-checked lecture marker | Patch application and passing `FAIL_TO_PASS`/`PASS_TO_PASS` tests |
| Strength | Detailed criterion-level diagnosis and visible disagreement analysis | Concrete executable checks of software behavior and regressions |
| Main risks | Subjectivity, criterion leakage, unresolved policies, model/annotator dependence | Underspecified issues, overly narrow or broad tests, environment artifacts, benchmark exposure during training |

**Critical reassessment (February 23, 2026).** OpenAI later reported an audit of **138 difficult Verified tasks**, specifically selected because a model did not solve them consistently over repeated attempts. At least **59.4% of that selected subset** had material problems in tests and/or descriptions. **This figure must not be extrapolated to all 500 tasks** because the audit did not use a representative random sample. OpenAI also reported contamination risk and stopped using SWE-bench Verified for frontier capability reporting, recommending SWE-bench Pro for reporting instead. This does not erase Verified's historical contribution; it shows why benchmarks need ongoing validation as both models and the evaluation ecosystem change.

### Results — Advanced

I examined **four identifiable benchmark instances** and documented their real-world task formulations, difficulty labels, likely engineering challenges, the human review process, and the test-based grading method. The advanced work is a **documented benchmark investigation**, not a claimed run of a coding agent over SWE-bench or a measured solve rate. The `astropy__astropy-13033` issue provides a concrete illustration of validity risks that parallels our Day 4 finding that passing a mechanical format check can differ from meeting the underlying semantic requirement.

### Observations and questions

- Better scoring rules must be **both operationalizable and valid**: the Lecture 04 regex can approve `Lecture 00`, while a coding benchmark can reject valid patches if its tests demand undocumented details.
- **Human validation helps but is not permanent certification.** Even a reviewed benchmark can accumulate problems as model exposure and tool capabilities change.
- Agreement should be reported **per criterion**, with automatic coverage and human-review counts, rather than presenting one headline number without its denominators.
- A future extension would add a semantic lecture-ID validator without changing the original exercise's baseline, repeat LLM judging across several runs, and use an independent expert-labeled holdout set to study disagreement stability.
- For benchmark construction, I would ask: What is an acceptable rate of false rejection of correct solutions? How do we test for hidden specification requirements? How do we detect exposure to test cases in pretraining data? How should we select and adjudicate ambiguous problems?

### Evidence and sources

**Local artifacts** (committed separately from this answer):

- `lectures/04/labels.jsonl` — human-reviewed labels for ten instructor cases.
- `lectures/04/judge_result_before.json` — original rubric baseline.
- `lectures/04/judge_result_after.json` — revised-rubric run.
- `lectures/04/judge_result.json` — most recent official-case evaluation.
- `data/eval/rubric.md` — Grounding policy change.
- `data/eval/my_cases.jsonl` — ten authored cases and fixed generated answers.
- `lectures/04/my_labels.jsonl` — human-reviewed labels of authored cases.
- `lectures/04/my_judge_result.json` — LLM grading of authored cases.

**Public sources:**

1. OpenAI (2024), *Introducing SWE-bench Verified*: https://openai.com/index/introducing-swe-bench-verified/
2. Princeton NLP / SWE-bench, *SWE-bench Verified dataset and schema*: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
3. SWE-bench, *Verified benchmark overview*: https://www.swebench.com/verified
4. OpenAI (2026-02-23), *Why SWE-bench Verified no longer measures frontier coding capabilities*: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
5. Astropy, issue #13235, *Consider removing auto-transform of structured column into NdarrayMixin*: https://github.com/astropy/astropy/issues/13235
6. SWE-bench community, issue #20 (2026-04-16), *Inconsistency between PR description and gold tests in astropy__astropy-13033*: https://github.com/SWE-bench/swe-bench-tasks/issues/20
7. Public task descriptions (mirrors of Verified instances): https://hub.harborframework.com/tasks/swe-bench/astropy__astropy-12907/latest ; https://hub.harborframework.com/tasks/swe-bench/astropy__astropy-13033/latest ; https://huggingface.co/datasets/hngl/swebench-verified-harbor/blob/main/astropy__astropy-13236/instruction.md ; https://huggingface.co/datasets/DCAgent/swebench-verified/blob/main/astropy__astropy-13398/instruction.md

*Evidence interpretation note:* Figures describing the user's local executions are taken from recorded terminal and UI outputs; external benchmark claims come from the linked public sources. No additional model evaluations or Astropy patches were executed as part of this literature-oriented Advanced review.
