# Dataset Support Matrix

This framework registers **all benchmark families mentioned in the PDF** and
normalizes them into the shared `BenchmarkExample` schema.

## Registered datasets

### Education-specific
- EduBench
- TutorEval
- LM-Science-Tutor (alias path to TutorEval-style loading)
- ScienceQA
- MathTutorBench
- TutorBench

### Transferable
- HotpotQA
- AgentBench
- SCROLLS
- LongBench-v2
- BEIR
- FEVER
- Wizard of Wikipedia

## Loader policy

The registry uses one of three policies per dataset:

1. **Hugging Face default**
   - Used when a stable public dataset identifier is commonly available.
   - Requires the optional `datasets` package.

2. **Local JSONL / JSON adapter**
   - Used for repo-backed, gated, environment-heavy, or normalized-export
     workflows.
   - Recommended for reproducible conference experiments.

3. **Override-driven**
   - The config can replace the default loader and source per dataset.
   - This is the intended path for MathTutorBench, BEIR subset exports,
     AgentBench environment snapshots, and any institution-specific corpora.

## Normalization target

Every adapter emits the same fields:

- `question`
- `gold_answer`
- `context_text`
- `dialogue_history`
- `rubric`
- `images`
- `reference_docs`
- `expected_doc_ids`
- `metadata`

That means all pipelines and evaluators work against a single data contract.

## Recommendation for serious experiments

For paper-ready runs, export each dataset into a normalized local file and pin
it in `configs/datasets.example.yaml`. This removes external dataset drift,
reduces dependency variability, and keeps benchmark splits reproducible.
