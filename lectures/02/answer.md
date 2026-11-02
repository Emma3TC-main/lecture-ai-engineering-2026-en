# Lecture 02 Answers

## Exercises

- **Model and settings used (from `config` in `result.json`):**
  - Base model: `Qwen/Qwen3.5-0.8B`
  - Revision: `2fc06364715b967f1860aea9cf38778875588b17`
  - Seed: `42`
  - Epochs: `20`
  - Warmup steps: `6`
  - Learning rate: `2e-4`
  - LoRA rank (`r`): `16`
  - LoRA alpha: `32`
  - Maximum sequence length: `512`
  - Maximum new tokens: `96`
  - Generation: deterministic (`do_sample=false`)
  - Method: LoRA without quantization

- **Did you run training yourself (`executed_training_myself`):** Yes. I ran the standard LoRA training and the before/after evaluations myself. I completed the required notebook workflow. Section 8 (4-bit quantization) was optional and was not used for the reported results.

- **Before training:** `format_rate = 0.9524` / `content_rate = 0.3810`

- **After training:** `format_rate = 1.0000` / `content_rate = 0.9048`

- **Questions that regressed:** None in the standard 21-question evaluation (`regressed = []`).

- **Failures remaining after training and their types:**
  - The grounded question about submitting a format other than PDF was treated as unsupported even though the FAQ contained relevant evidence: **evasion through a confirmation request despite evidence being present**.
  - The question about whether office hours require an appointment was answered as though the FAQ supported the answer even though that information was absent: **assertion without evidence**.
  - No broken-format failures remained after training; format compliance reached `1.0`.

- **Results of knowledge questions (section 6; classification before → after training) and observations:**
  - Final assignment deadline, information absent from the FAQ: `asserted → asserted`. The model continued to provide an unsupported deadline after fine-tuning.
  - Instructor name, information absent from the FAQ: `needs_review → needs_review`. The model correctly recognized that confirmation was required.
  - Topic of the next lecture, information absent from the FAQ: `needs_review → needs_review`. The model correctly continued to abstain.
  - Assignment deadline, information present in the FAQ: `asserted → asserted`. The supported answer remained correct.

  Overall, fine-tuning substantially improved task behavior and format compliance, but it did not reliably teach the model the boundary between information contained in the supplied material and information originating from pretrained knowledge.

- **What changed and what stayed the same with training:**
  - Format compliance improved from `0.9524` to `1.0000`.
  - Content accuracy improved from `0.3810` to `0.9048`.
  - Task-specific FAQ behavior improved substantially.
  - Some knowledge-boundary behavior remained unchanged: the model could still assert information that was not contained in the supplied source.
  - Fine-tuning therefore improved behavior but did not provide reliable source provenance or unknown-information detection.

## Homework

- **Level attempted:** Advanced

- **Settings or data changed:**
  - Following the Advanced instructions, I generated the course glossary SFT dataset with:

    ```bash
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    ```

  - The generated dataset contained `116` Q&A examples in `messages` format.
  - I created a separate 12-question evaluation set:
    - 4 direct glossary questions
    - 4 paraphrased glossary questions
    - 4 questions about concepts absent from the glossary
  - The evaluation questions were checked to avoid literal duplication with the training questions.
  - Expected answers and glossary membership were stored in `advanced_gold.json`.
  - The absent concepts were:
    - CRDT
    - BGP route reflection
    - NUMA
    - Kubernetes Horizontal Pod Autoscaler

  Following `problem.md`, I first loaded the generated `messages` records directly with `Dataset.from_list` and trained them with `SFTTrainer`.

  The first run reused the original 20-epoch setting. Because the glossary dataset contained 116 examples instead of the original 14 training examples, this produced `300` optimizer steps. The logged loss decreased to approximately `0.07`, but generation became degenerate, producing repetitive sequences such as `user_question_question...` and repeated digits.

  I then repeated the experiment with only 3 epochs. Reducing the number of optimizer updates alone did not remove the degenerate generation.

  As a diagnostic experiment, I inspected the supervision pipeline and converted the same 116 glossary examples from conversational `messages` into explicit `prompt` and `completion` fields. I enabled completion-only loss so that supervision was restricted to the assistant completion while keeping the underlying glossary Q&A data unchanged.

  Before retraining, I verified the masking on a sample:

  - Masked prompt tokens: `17`
  - Supervised completion tokens: `59`

  The text receiving loss contained the assistant answer but not the user question. After this change, the degenerate generation disappeared.

  The corrected Advanced training configuration was:

  - Base model: `Qwen/Qwen3.5-0.8B`
  - Training examples: `116`
  - LoRA rank: `16`
  - LoRA alpha: `32`
  - Learning rate: `1e-4`
  - Epochs: `3`
  - Optimizer steps: `45`
  - Trainable parameters: `6,389,760`
  - Trainable percentage: `0.8421%`
  - Training time: `93 s`
  - Completion-only loss: enabled

  Logged loss decreased from `3.646895` at step 5 to `1.704818` at step 45, while generation remained stable.

- **Results (comparison tables or numbers):**

  ### Glossary evaluation

  | Metric | Before | After |
  |---|---:|---:|
  | Known-content correct | 7/8 | 6/8 |
  | Source-boundary correct | 0/4 | 0/4 |

  The base model's most notable semantic error was the paraphrased temperature question: it incorrectly stated that increasing sampling temperature decreases output diversity.

  After fine-tuning, this relationship was corrected: the model correctly stated that higher temperature generally produces more varied output.

  However, two regressions appeared in the context-window questions:

  1. The direct answer described the context window only as an input-token limit and introduced an unsupported `4096`-token value.
  2. The paraphrased question was confused with the model's output window.

  The four absent concepts were still answered rather than identified as absent from the AIE2026 glossary. Thus, source-boundary accuracy remained `0/4`.

  This result shows that fine-tuning did not uniformly improve factual behavior. It corrected one semantic association while introducing regressions elsewhere.

  ### Fine-tuning versus glossary retrieval

  I implemented a simple TF-IDF lexical search over the same glossary and evaluated it using the same 12 questions.

  With threshold `0.18`:

  - Correct absence detection: `4/4`
  - Present questions retrieved: `1/8`

  This configuration produced strong abstention but very low recall.

  With threshold `0.05`:

  - Correct absence detection: `4/4`
  - Present questions marked as found: `8/8`
  - Correct top-1 retrievals: `2/8`

  Lowering the threshold increased recall but exposed weak semantic ranking. Examples included:

  - Token question → `Tool namespace`
  - Temperature question → `LLM`
  - Paraphrased temperature question → `Tool description`

  Therefore, naive lexical retrieval was effective at rejecting the four clearly absent terms in this evaluation, but it was not reliable for semantic paraphrases.

- **Discussion of when fine-tuning is and is not suitable:**

  Fine-tuning and retrieval address different requirements.

  **Fine-tuning is suitable when:**
  - the desired behavior is relatively stable;
  - the objective is to change output style, format, task behavior, or domain-specific response patterns;
  - the desired behavior should be available directly during inference without performing an external lookup.

  **Fine-tuning is less suitable when:**
  - information changes frequently;
  - exact source provenance is required;
  - the system must reliably determine whether information was present or absent from a particular document;
  - factual updates need to be incorporated without retraining.

  The experiment showed that LoRA can generalize beyond exact training wording and can correct specific learned relationships, but it does not guarantee source membership or factual provenance and may introduce regressions even when training loss decreases.

  Retrieval has complementary strengths. It provides explicit access to a source and can abstain when no sufficiently similar information is found, while updates can be incorporated without retraining. However, retrieval quality depends strongly on the search mechanism. The TF-IDF implementation used here performed poorly on semantic paraphrases, indicating that semantic embeddings or reranking would be preferable for a more robust system.

  The failed Advanced runs also provided an important engineering lesson. Training loss alone was not sufficient to validate the SFT pipeline. Following the prescribed `messages` workflow first exposed degenerate generation in my run; inspecting the supervised tokens and testing completion-only supervision provided a way to diagnose the behavior. After restricting loss to the assistant completion, the observed generation degeneration disappeared.

  Overall, the experiment suggests that fine-tuning is better suited to stable behavioral adaptation, while retrieval is preferable when source grounding, freshness, or explicit knowledge boundaries are important.