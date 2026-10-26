# Lecture 01 Answers

## Exercises

- **Model's answer to the course-specific question (the 3 homework levels; a summary is fine):** The models did not reliably return the correct course-specific names. Instead, they generated plausible generic tiers such as Core/Practice, Application/Intermediate, and Challenge/Advanced, with some responses partially matching the actual terminology.

- **What you noticed when comparing it with the correct meaning in the glossary:** The generated responses were fluent and plausible, but they contained unsupported or fabricated course-specific details. They lacked grounding in the actual course information, where the levels are Basic, Implementation, and Advanced. This showed that a model's confidence and fluency do not necessarily imply factual correctness.

- **Token counts you displayed (input / output):** In one representative run of “What is LoRA?”, the counts were Input: 6 / Output: 687. Output length varied across models and repeated executions.

## Homework

- **Level attempted:** Advanced

- **What you did:** I expanded the Basic comparison by implementing additional instrumentation in the script to measure input, output, thinking, and total token counts, as well as end-to-end response time. I systematically compared three models (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, and `gemini-3.1-flash-lite`) across two questions: a general-knowledge question and a course-specific question not provided in the model context. I obtained three successful measurements for each combination, resulting in 18 controlled observations.

- **Results:**
  - The general question (“What is LoRA?”) consistently produced longer visible outputs than the course-specific question (“What are the 3 homework levels in the AIE2026 course?”).
  - The Lite models (`gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`) reported `thoughts=0` across the measured executions.
  - In contrast, `gemini-3.5-flash` used approximately 1,000 thinking tokens per request on average. These tokens were included in `total_token_count` but were not part of the visible candidate response.
  - `gemini-3.5-flash` also showed the highest average latency in this experiment. However, this cannot be attributed solely to thinking tokens because the underlying model and serving behavior also differed.
  - Repeated executions with the same model and prompt still produced different output lengths and response times.

- **Observations and questions:**
  - **Variability:** Output length and latency depended on the model, question type, and individual execution. Even with the same prompt and model, noticeable variation occurred. For example, `gemini-3.1-flash-lite` ranged from approximately 2.9 seconds to 8.0 seconds for the same LoRA prompt.
  - **Knowledge vs. hallucination:** The models handled general knowledge more reliably, while course-specific information that was absent from their context led to plausible but unsupported assumptions. This illustrates the importance of grounding when factual domain-specific information is required.
  - **Thinking tokens:** Visible output tokens do not necessarily represent the model's complete token usage. In `gemini-3.5-flash`, the API reported additional thinking tokens that contributed substantially to `total_token_count`.
  - **Rate-limit nuance:** Rate limits should not be simplified to “more total tokens means a rate-limit error.” Different dimensions matter:
    - Number of calls contributes to **RPM** (Requests Per Minute) and **RPD** (Requests Per Day).
    - Input and context size contribute to **input TPM** (Tokens Per Minute).
    - Output and thinking tokens affect generation usage, latency, and the available output-token budget.
