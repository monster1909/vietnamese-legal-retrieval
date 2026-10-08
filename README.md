# Vietnamese Legal Retrieval

Hybrid retrieval system that finds the Vietnamese legal documents (laws, decrees, circulars) answering a natural-language question. It fuses two fine-tuned bi-encoders with BM25 and blends in a cross-encoder reranker.

Built for **UIT Data Science Challenge 2026 – Task 1: Legal Information Retrieval** (8.5k legal documents, 7k training questions, up to 5 answers per question).

| Leaderboard | Recall@5 | Precision@5 | Rank |
|---|---|---|---|
| Public | **0.9547** | 0.2058 | 5 / 100 |
| Private | **0.9504** | 0.2041 | 9 / 100 |

## How it works

```
question
   ├─► encoder (fine-tuned, round 2) ─┐
   ├─► encoder (fine-tuned, round 1) ─┼─► weighted RRF ─► chunk → document pooling ─► blend with reranker ─► top-5
   └─► BM25 (accent-free, bigrams)  ──┘                                               (AITeamVN/Vietnamese_Reranker)
```

- **Legal-aware chunking.** Documents are split along *Điều → khoản → điểm → paragraph → sentence* boundaries into chunks of about 1,000 tokens. Each chunk is prefixed with its document number, issuer, title and chapter breadcrumb, so an article from the middle of a document still says which document it belongs to.
- **BM25 with a custom tokenizer.** Document numbers such as `75/2017/NĐ-CP` stay whole tokens, accents are stripped, and syllable bigrams approximate Vietnamese word segmentation.
- **Bi-encoder fine-tuning** of [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) (BGE-M3) in two rounds:
  - Round 1 mines hard negatives with BM25. Round 2 re-mines them with the round-1 model and trains on both rounds.
  - Labels exist only at document level, so the positive for each question is the best-scoring chunk inside its gold document.
  - Negatives that score too close to the positive are dropped as likely unlabeled positives.
  - Training uses `CachedMultipleNegativesRankingLoss` (GradCache) for an effective batch of 128 on one GPU, with document-level early stopping.
- **Fusion tuned offline.** Retriever rankings and reranker logits for a wide candidate pool are computed once and cached. Optuna then searches the fusion weights, RRF constant, pooling and reranker blend on CPU, scoring candidates against a tune/verify split of validation.

## Quickstart

```bash
git clone https://github.com/monster1909/vietnamese-legal-retrieval.git
cd vietnamese-legal-retrieval
pip install -r requirements.txt
```

### Train

Needs a GPU (A100 recommended). Each step resumes from whatever already exists in the artifacts folder.

```bash
bash scripts/train.sh --data data --artifacts artifacts      # train → tune → evaluate
bash scripts/train.sh --steps tune,evaluate --trials 500     # re-tune on existing models
bash scripts/train.sh -h
```

### Inference

```bash
bash scripts/infer.sh predict examples/data/questions.json -o output   # → output/predictions.json
bash scripts/infer.sh search "Thủ tục đăng ký kết hôn với người nước ngoài?" -k 5
bash scripts/infer.sh -h
```

No local GPU? [`notebooks/train.ipynb`](notebooks/train.ipynb) and [`notebooks/infer.ipynb`](notebooks/infer.ipynb) run the same commands on Google Colab. Both shell scripts and both notebooks are thin wrappers around `python -m src.pipeline {train,tune,evaluate,predict,search}`.

## Data

The competition data is **not redistributed** here. [`examples/data/`](examples/data) contains a few synthetic documents and questions in the exact input format:

```
data/
├── train.json              {"<qid>": {"question": "...", "answer": ["<doc_id>", ...]}}
├── questions.json          same format with "answer": null (input to predict)
└── selected-contexts.zip   selected-contexts/context_<id>.json = {"id", "name", "link", "passage"}
```

To train on your own corpus, provide the same three files. `passage` is the full document text, and `answer` lists the ids of the relevant documents.

## Artifacts

Everything the pipeline produces lives in one folder:

```
artifacts/
├── models/encoder/         fine-tuned encoder, round 2
├── models/encoder_ft1/     fine-tuned encoder, round 1
├── emb_encoder.npy         corpus embeddings (fp16), one per encoder
├── emb_encoder_ft1.npy
├── chunks.parquet          chunked corpus
├── bm25_index.npz
├── cache_val.npz           cached validation candidates + reranker logits (for re-tuning)
└── best_config.json        fusion parameters selected by Optuna
```

## Project structure

```
src/
├── pipeline.py        CLI: train, tune, evaluate, predict, search
├── config.py          all hyperparameters
├── data.py            corpus and question loading
├── doc_header.py      parses issuer / number / type / title from legal document headers
├── chunking.py        legal-structure-aware chunking
├── mine_negatives.py  positive selection and hard-negative mining
├── train.py           bi-encoder fine-tuning
├── tune.py            Optuna search over fusion parameters
├── evaluate.py        competition metrics
└── retrieval/         bm25, dense, rerank, fusion, scoring, cache
scripts/               train.sh, infer.sh
notebooks/             train.ipynb, infer.ipynb (Colab)
examples/data/         synthetic sample data
```

## Limitations

- **Questions with several relevant documents.** Recall drops from about 0.98 to about 0.8 when a question needs more than one document, because documents are ranked independently rather than as a set of related regulations.
- **Heavy inference.** Three large transformer models run per query. The full setup needs a GPU; on CPU the reranking step has to be cut down.
- **Zero-shot reranker.** The cross-encoder is used off the shelf. A listwise fine-tuned version did not beat it reliably on validation.
- **Static corpus.** Adding or amending documents requires re-chunking and re-encoding the corpus, and there is no notion of a regulation's effective date or whether it has been superseded.
- **Heuristic parsing.** Chunking and header extraction rely on regular expressions over the usual layout of Vietnamese legal documents. Irregularly formatted documents get coarser chunks and weaker prefixes.
- **Low precision by design.** The metric rewards recall, so the system always returns 5 documents.

## Future work

- **Public demo.** A Gradio app on Hugging Face Spaces with a fast mode (dense + BM25) and an optional rerank mode.
- **Faster inference.** ONNX/int8 export, a distilled reranker, and an adaptive reranking budget.
- **Multi-document recall.** A dynamic cutoff on the number of returned documents and better modelling of related regulations, e.g. amendments and implementing decrees.
- **Retrieval-augmented answers.** Generate an answer that cites the retrieved articles, with conversational follow-up questions.
- **Ablation study.** Measure the contribution of each component: second encoder, BM25, pooling and reranker.

## Acknowledgements

- Base models: [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) and [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker).
- Task and data: UIT Data Science Challenge 2026.
