# Lecture 03 Answers

## Exercises

- **Differences between the Lecture 01 (bare LLM) and agent answers:**

  I compared `app.ask` and `app.agent` using the same model (`gemini-3.5-flash-lite`) and the same question: "What are the 3 homework levels in the AIE2026 course?"

  The bare LLM explicitly stated that it did not have access to the course-specific information and could not identify the three homework levels. Its execution used 18 input tokens and 45 output tokens, with a reported execution time of 0.691 seconds.

  In contrast, the Lecture 03 agent correctly identified Basic, Implementation, and Advanced, explained that completing any one level satisfies the homework requirement, and referenced Lecture 01. Its answer was consistent with the `homework-levels` entry in `data/glossary.json`.

  Both executions used the same model, but only the agent had access to the course-specific tools. This demonstrates the practical benefit of connecting an LLM to external information through an agent harness. The initial homework-levels agent execution was not traced, so its exact tool-call sequence was not independently recorded.

- **Tool calls shown by `--trace` (tool names and arguments):**

  For "What is the difference between LoRA and QLoRA?", the agent called:

  ```text
  search_glossary({"query": "LoRA"})
  search_glossary({"query": "QLoRA"})
  ```

  Both calls returned relevant glossary definitions. The final response correctly distinguished low-rank adaptation from the additional base-model quantization used by QLoRA and referenced Lecture 02.

  After implementing the quiz tool, the question "Give me one review question from Lecture 02" produced:

  ```text
  make_quiz({"lecture": 2})
  ```

  The tool returned a question about Hugging Face Hub, which the agent presented as a review question for Lecture 02.

- **Number of LLM calls for one question (`requests`):**

  | Evaluation | Requests | Tool calls | Input tokens | Output tokens |
  |---|---:|---:|---:|---:|
  | LoRA vs QLoRA | 2 | 2 | 1,125 | 302 |
  | Lecture 02 review question | 2 | 1 | 847 | 84 |

  These traces demonstrate the agent loop: the model requests tool calls, the application executes them, and the results are returned to the model before the final response.

- **Tool added and the description in its docstring:**

  I implemented `make_quiz(lecture: int) -> str` in `app/tools.py` and registered it using `agent.tool_plain(tools.make_quiz)` in `app/agent.py`.

  The function filters glossary entries by lecture, randomly selects a matching term, and returns a JSON object containing a question, reference answer, and lecture number.

  Its docstring specifies when the tool should be used and describes the expected lecture-number argument. The implementation also handles lectures without matching entries by returning a descriptive message rather than raising an exception.

  The tool listing confirmed that all three tools were registered: `search_glossary`, `get_lecture_summary`, and `make_quiz`.

  Validation completed with **15 passing tests**, one dependency deprecation warning, and no whitespace errors reported by `git diff --check`.

## Homework

- **Level attempted: Advanced — Options (a) and (b).**

- **What you did:**

  **Advanced (a): Google ADK**

  I rebuilt the learning support agent with Google's Agent Development Kit while preserving its existing instructions and tool implementations.

  The ADK implementation was created in `adk_agent/glossary_agent/`. It imports `INSTRUCTIONS` from `app.agent` and reuses `search_glossary`, `get_lecture_summary`, and `make_quiz` from `app.tools`.

  The ADK agent was configured with `gemini-3.5-flash-lite` and successfully initialized with all three tools. I executed it using `adk run` and submitted the same three evaluation questions used with Pydantic AI.

  **Advanced (b): Pi Coding Agent**

  I installed Pi Coding Agent v1.5.0 in GitHub Codespaces and executed it with Gemini 3.5 Flash Lite.

  To isolate the analysis from the coursework repository, I created `/tmp/aie2026-pi-review/`, containing the AIE2026 agent instructions and a copy of Pi's public system prompt construction source:

  `packages/coding-agent/src/core/system-prompt.ts`

  The retrieved source file had the following SHA-256 checksum:

  `eef891d637974d09a6611d33e52d66d5f2a93d6e9512259ba45f593c42143617`

  I asked Pi to perform a read-only architectural comparison. The captured execution showed file listing and reading operations followed by an analysis of both instruction designs.

- **Results:**

  **Advanced (a): Framework comparison**

  | Aspect | Pydantic AI | Google ADK |
  |---|---|---|
  | Agent construction | `Agent(instructions=...)` | `root_agent = Agent(...)` |
  | Tool registration | `agent.tool_plain(...)` | `tools=[...]` |
  | Execution | Python CLI | ADK interactive CLI |
  | Model | Gemini 3.5 Flash Lite | Gemini 3.5 Flash Lite |
  | Domain functions | Original glossary tools | Same imported functions |

  The observed answers were consistent with the course material:

  - Both identified the three homework levels and referenced Lecture 01.
  - Both correctly explained LoRA versus QLoRA and referenced Lecture 02.
  - Both generated a Lecture 02 review question. Pydantic AI selected Hugging Face Hub; ADK selected Epoch.

  The quiz-question difference is expected because `make_quiz` randomly selects a term. The Pydantic AI trace showed actual tool calls, while the captured ADK terminal session showed responses without exposing equivalent internal traces.

  This experiment demonstrates that the same domain-specific instructions and tools can be reused across different agent frameworks with relatively small changes to their assembly code.

  **Advanced (b): System prompt architecture comparison**

  | Dimension | AIE2026 agent | Pi Coding Agent |
  |---|---|---|
  | Main purpose | Learning support | General-purpose coding |
  | Instructions | Short, domain-specific rule set | Dynamically assembled prompt |
  | Tools | Glossary, lecture summaries, quiz | Read, bash, edit, write by default |
  | Context | Course information accessed through tools | Project context, working directory, Skills |
  | Output guidance | Grounded answers, hints, lecture attribution | Coding assistance and clear file references |
  | Prompt composition | Fixed instruction text | Structured, configurable sections |

  Pi's source contains functions including `normalizeBuildSystemPromptOptions`, `buildSystemPromptSections`, `buildRules`, and `diffSystemPromptSections`.

  These functions normalize tool configurations, compose prompt sections, generate tool-dependent rules, and identify changes between prompt states.

  The AIE2026 agent instead emphasizes knowledge grounding and pedagogical boundaries. It explicitly instructs the model to consult the glossary and disclose when requested information is absent.

  A key distinction is that Pi's prompt construction adapts to selected tools, project context, and reusable Skills, while our educational agent uses a smaller and more specialized instruction set.

- **Observations and questions:**

  The two Advanced experiments examine different aspects of agent engineering.

  Advanced (a) demonstrates framework portability by reusing the same instructions and tools in Pydantic AI and Google ADK. Advanced (b) demonstrates how a more general-purpose coding harness assembles its instructions and capabilities dynamically.

  Three engineering lessons emerged:

  1. **Separate domain logic from framework integration.** Reusing tools across frameworks reduces duplicated implementation and facilitates comparison.
  2. **Treat prompt composition as part of system architecture.** Tool descriptions, project context, Skills, and runtime configuration affect the capabilities exposed to the model.
  3. **Distinguish behavioral instructions from enforced controls.** Instructions to avoid unsupported answers are not guarantees against hallucinations. Likewise, prompt-section validation does not replace filesystem permissions, approval mechanisms, or sandbox isolation.

  The main limitations are nondeterministic model responses, the absence of explicit ADK tool-call traces in the captured terminal session, and the fact that examining Pi's public prompt construction source does not constitute a full runtime or security audit.

  Future experiments could measure tool-call precision, execution latency, token consumption, behavior under malformed tool results, and recovery from failed actions.

  Overall, these experiments reinforced that reliable AI agents depend on the design of the harness, tools, context management, and operational controls—not only on the capabilities of the underlying language model.