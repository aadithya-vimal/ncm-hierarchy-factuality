# Research Protocol

## Research question

How do source-grounded factual claims appear, persist, disappear, or change factuality status across successive decoded hierarchy levels of the released Next-Concept Modeling (NCM) XSum summarization checkpoint?

## Primary quantities

1. **Claim status by hierarchy level:** shares of candidate claims labeled supported, contradicted or unresolved by the automated NLI evaluator.
2. **Observable claim transitions:** new sentence-level candidates, removed candidates, matched candidates whose surface form persists, and matched candidates whose automated support label changes between adjacent decoded levels.
3. **Factuality–ROUGE association:** association between level-wise ROUGE-L and support/contradiction proportions. This is associational, not causal.
4. **Uncertainty:** article-cluster bootstrap intervals for level-level means and per-article correlations, where appropriate.

## Data and model

- Upstream implementation: `HazyResearch/ncm`, pinned to commit `c70274273f489abc73832b2b98584a8483b6b203`.
- XSum checkpoint: `hazyresearch/ncm-checkpoints`, pinned to release commit `7100573f83178b6e9f45ebffe0e9aa6df384e712`, `xsum/` assets.
- Input and reference artifacts: `hazyresearch/ncm-datasets`, XSum test split.
- Full source documents: `EdinburghNLP/xsum`, test split, aligned to NCM reference summaries; alignment is checked before generation.
- Entailment model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, pinned to revision `eb8b17b1983bca679126ea69b12b5d28c5fe9b9a`. This classifier provides an automated, imperfect factuality signal.

## Preprocessing

1. Download only the necessary XSum checkpoint and data files.
2. Load NCM input/reference pickles and the full XSum test split.
3. Normalize reference summaries conservatively (case-folding, alphanumeric token normalization) for alignment. Use exact normalized reference matches and a strict positional fallback only when the normalized strings remain sufficiently similar. Abort before model loading if any NCM reference remains unmatched (the configured minimum is 99%, but the run additionally requires a one-to-one aligned record for every NCM input).
4. Store private source text separately from shareable records. Do not upload the source corpus.
5. Run deterministic NCM decoding once for each incomplete article. Decode cumulative VQ levels using the actual upstream `idx_to_zhats` and decoder interfaces.

## Candidate claims and evidence

Each decoded text is split into sentence-level candidate units using deterministic rule-based segmentation. A sentence is not guaranteed to be atomic; report the unit as a *candidate claim* and note this limitation.

For each candidate, retrieval ranks overlapping source windows by TF-IDF similarity. The top configured windows are paired with the claim for NLI inference. Evidence-pair scores are checkpointed. The label thresholds are fixed in configuration: supported when the maximum entailment probability is at least 0.70 and exceeds maximum contradiction by at least 0.10; contradicted symmetrically; unresolved otherwise. These thresholds are operational criteria, not calibrated probabilities of truth.

## Cross-level matching

Between each adjacent pair of levels, build a TF-IDF cosine similarity matrix over candidate claims, then apply one-to-one maximum-weight matching. Only matches at or above the predeclared threshold are accepted. This matching is a transparent lexical correspondence heuristic, not guaranteed semantic identity. A matched claim with different automated labels is recorded as a label change; unmatched later/earlier candidates are recorded as introduced/removed in the *decoded output sequence*, not as causal claims about the model's internal mechanism.

## Analysis and uncertainty

- Calculate claim label rates and ROUGE-1/2/L by level.
- Calculate transitions per adjacent level.
- Calculate Spearman correlations between per-article ROUGE-L and claim-status proportions; include the sample size and interpretation limits.
- Calculate percentile bootstrap intervals by resampling articles as clusters, not individual claims. Multiple claims from one article are not independent.
- No results are supplied in advance. Values are generated only after completed inference and analysis.

## Predeclared limitations

- Sentence candidates may contain multiple facts or pronouns requiring wider context.
- NLI probabilities may be miscalibrated and sensitive to retrieval quality, truncation and source wording.
- TF-IDF cross-level matching can miss paraphrases or match similar but distinct claims.
- A change in automated support label is not itself proof that NCM introduced or corrected a factual error.
- XSum-specific results do not establish generality across tasks or summarization datasets.
- Factuality label changes can reflect text reconstruction differences rather than a simple monotonic refinement in length or quality.
