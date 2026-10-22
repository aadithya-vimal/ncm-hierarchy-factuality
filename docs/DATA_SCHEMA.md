# Data Schema

## Private working records (Google Drive only by default)

`private/source_articles.jsonl` has one row per aligned XSum test article: `article_id`, `ncm_index`, `source_text`, `reference_summary`, `source_sha256`, `reference_sha256`, `alignment_method`. It contains source text and must not be uploaded to a public repository.

## Hierarchy output

`hierarchy_outputs.jsonl` / `hierarchy_outputs.parquet`:
- `article_id`: stable XSum identifier
- `ncm_index`: position in the official NCM dataset split
- `level_index`: one-based hierarchy level
- `level_code_length`: configured level length
- `summary_text`: decoded output at that cumulative level
- `model_id`, `checkpoint_revision`, `upstream_commit`, `run_fingerprint`

## Claim records

`claims.parquet` contains `claim_id`, `article_id`, `level_index`, `claim_index`, `claim_text`, `claim_granularity`, `source_sha256`, `label`, `max_entailment_probability`, `max_contradiction_probability`, `evidence_chunk_indices`, `annotation_model_id`, `annotation_method`.

The public schema does not include full source articles or the evidence-window contents. Evidence chunk indexes are local-relative references to the private source document; they are not sufficient to reconstruct or redistribute the source text.

## Transition records

`claim_transitions.parquet` contains `article_id`, `from_level`, `to_level`, `previous_claim_id`, `current_claim_id`, `similarity`, `previous_label`, `current_label`, `lifecycle_transition`, `matcher`, and `matching_threshold`. One side of an introduced/removed transition is null.

## Analysis

- `level_metrics.csv`: article count, claim counts/proportions, mean ROUGE-1/2/L, cluster-bootstrap intervals.
- `correlations.csv`: Spearman association estimates between ROUGE-L and factuality proportions at each level.
- `transition_metrics.csv`: transition counts/rates by adjacent level.
- `analysis_summary.json`: dataset counts, model IDs, configuration and key statistics (no fabricated values).

Parquet requires `pyarrow`. CSV/JSONL remain available for inspection. Checksums refer to the exact exported files in the local Hugging Face package.
