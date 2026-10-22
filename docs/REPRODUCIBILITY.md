# Reproducibility Checklist

The experiment manifest records:
- configuration fingerprint and normalized experiment configuration;
- upstream NCM Git commit and checkpoint repository revision;
- NLI model ID and resolved commit/revision when available;
- Python, PyTorch, Transformers, datasets, Hugging Face Hub, CUDA and GPU details;
- aligned data row count and reference-alignment quality;
- creation timestamps, stage completion status and hashes of important outputs.

Before analysis, verify:

- [ ] inference contains exactly the expected number of hierarchy levels per aligned article;
- [ ] no duplicate `(article_id, level_index)` rows;
- [ ] all public claim IDs point to an output at the corresponding level;
- [ ] every claim label comes from an NLI scoring record or is explicitly unresolved;
- [ ] transitions refer only to adjacent configured levels;
- [ ] all tables and figures can be recreated from saved JSONL/SQLite records;
- [ ] the public export contains no `source_articles.jsonl`, token, absolute local path or credential;
- [ ] checksums validate before remote publication.

The included local tests use synthetic fixtures and do not download checkpoints, load NCM weights or consume a GPU.
