# Day 104 — LLM Eval II: RAGAS (Faithfulness, Context Relevance, Answer Correctness)

## WHY

A RAG (Retrieval-Augmented Generation) pipeline has two independent failure surfaces: **bad retrieval** (the wrong chunks come back) and **bad generation** (the model hallucinates even given perfect context, or generates a plausible-sounding answer from irrelevant context). A single end-to-end "is this answer correct" score conflates these — you can't tell which half of the pipeline to fix. RAGAS-style metrics separate them:

- **Faithfulness** — is every claim in the answer actually supported by the retrieved context? A low score means the model is hallucinating regardless of how good retrieval was.
- **Context relevance** — are the retrieved chunks actually relevant to the question? A low score means retrieval is failing, even if the model is faithful to the (bad) context it got.
- **Answer correctness** — does the answer semantically and factually match a known ground truth? This is the end-to-end check, but only meaningful once you know whether faithfulness/relevance are healthy.

---

## HOW

`FaithfulnessScore` and `ContextRelevanceScore` both follow the same pattern: count total units (claims / chunks) and supported/relevant units, with `score()` returning the supported fraction (defaulting to `1.0` when there are zero units — vacuously faithful/relevant). `AnswerCorrectnessScore` blends `semantic_similarity` and `factual_overlap` via a configurable `weight_semantic`.

`RAGASReport` aggregates all three across a dataset and computes `overall_score()` as the mean of the three means. `failure_taxonomy()` buckets failures by type using fixed thresholds (faithfulness < 0.7 → hallucination, context relevance < 0.5 → poor retrieval, correctness < 0.6 → wrong answer) — this is what tells an engineer *which* part of the pipeline to debug first.

---

## Class Diagram

```mermaid
classDiagram
    class RAGEvalExample {
        +str question
        +str answer
        +list~str~ contexts
        +str ground_truth
        +__post_init__()
        +to_dict() dict
    }

    class FaithfulnessScore {
        +str example_question
        +int num_claims
        +int num_supported_claims
        +__post_init__()
        +score() float
        +to_dict() dict
    }

    class ContextRelevanceScore {
        +str example_question
        +int num_chunks
        +int num_relevant_chunks
        +__post_init__()
        +score() float
        +to_dict() dict
    }

    class AnswerCorrectnessScore {
        +str example_question
        +float semantic_similarity
        +float factual_overlap
        +float weight_semantic
        +__post_init__()
        +score() float
        +to_dict() dict
    }

    class RAGASReport {
        +str dataset_name
        +list~FaithfulnessScore~ faithfulness_scores
        +list~ContextRelevanceScore~ context_scores
        +list~AnswerCorrectnessScore~ correctness_scores
        +__post_init__()
        +mean_faithfulness() float
        +mean_context_relevance() float
        +mean_correctness() float
        +overall_score() float
        +failure_taxonomy() dict
        +to_dict() dict
    }

    RAGASReport --> FaithfulnessScore
    RAGASReport --> ContextRelevanceScore
    RAGASReport --> AnswerCorrectnessScore
```

---

## Sequence Diagram — RAGAS Evaluation of a RAG Pipeline

```mermaid
sequenceDiagram
    participant Eval as Eval Harness
    participant RAG as RAG Pipeline
    participant Retr as Retriever
    participant Gen as Generator (LLM)
    participant Rep as RAGASReport

    Eval->>RAG: RAGEvalExample(question, ground_truth)
    RAG->>Retr: retrieve(question)
    Retr-->>RAG: contexts[]
    RAG->>Gen: generate(question, contexts)
    Gen-->>RAG: answer

    Eval->>Eval: extract claims from answer
    Eval->>Eval: check claims against contexts -> FaithfulnessScore
    Eval->>Eval: check each context vs question -> ContextRelevanceScore
    Eval->>Eval: compare answer vs ground_truth -> AnswerCorrectnessScore

    Eval->>Rep: append all three scores
    Rep->>Rep: overall_score(), failure_taxonomy()
    alt failure_taxonomy shows poor_retrieval
        Rep-->>Eval: fix retriever / chunking / embeddings
    else failure_taxonomy shows hallucination
        Rep-->>Eval: fix generation prompt / grounding instructions
    else failure_taxonomy shows wrong_answer
        Rep-->>Eval: investigate both retrieval and generation
    end
```

---

## Key Takeaways

1. RAGAS separates retrieval failures from generation failures — `failure_taxonomy()` tells you exactly which subsystem to fix.
2. `score()` for `FaithfulnessScore`/`ContextRelevanceScore` defaults to `1.0` with zero units — avoid divide-by-zero while staying conservative (no claims means nothing unsupported).
3. `AnswerCorrectnessScore.weight_semantic` lets you tune how much you trust embedding similarity vs literal factual overlap — useful when ground truths are short factual statements vs long narrative answers.
4. A model can be perfectly faithful to irrelevant context and still be wrong — faithfulness alone is not sufficient, you need all three metrics together.
