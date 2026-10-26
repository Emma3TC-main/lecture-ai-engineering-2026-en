# Lecture 01 Answers

## Exercises

- **Model's answer to the course-specific question (the 3 homework levels; a summary is fine):** The models did not reliably return the correct course-specific names. Instead, they generated plausible generic tiers such as Core/Practice, Application/Intermediate, and Challenge/Advanced, with some models partially matching the actual terminology.

- **What you noticed when comparing it with the correct meaning in the glossary:** The responses were fluent and convincing, but they contained unsupported or fabricated course-specific details. The glossary defines the actual levels as Basic, Implementation, and Advanced. This showed that a confident and plausible response is not necessarily factually correct when the model lacks the required context.

- **Token counts you displayed (input / output):** In one representative run of “What is LoRA?”, the counts were Input: 6 / Output: 687. I also observed that repeated runs could produce different output lengths.

## Homework

- **Level attempted:** Basic

- **What you did:** I compared the responses of three models (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, and `gemini-3.1-flash-lite`) using the same two questions: a general-knowledge question (“What is LoRA?”) and a course-specific question (“What are the 3 homework levels in the AIE2026 course?”).

- **Results:**
  - For the general LoRA question:
    - `gemini-3.5-flash-lite`: 687 output tokens
    - `gemini-3.5-flash`: 886 output tokens
    - `gemini-3.1-flash-lite`: 736 output tokens
  - For the course-specific AIE2026 question:
    - `gemini-3.5-flash-lite`: 311 output tokens
    - `gemini-3.5-flash`: 369 output tokens
    - `gemini-3.1-flash-lite`: 359 output tokens

- **Observations and questions:** The main difference was between information the models could answer from general knowledge and course-specific information that was not included in their context. For the AIE2026 question, the models often filled missing information with plausible assumptions instead of reliably identifying the correct course terminology. This illustrates why grounding through external data or tools is important when factual, domain-specific information is required. I also observed differences in response length and level of detail across models, as well as variation between repeated generations from the same model.
