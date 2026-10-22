# Methods and Operational Definitions

| Term | Operational definition |
|---|---|
| Decoded hierarchy level | A text reconstructed from the cumulative NCM code representation up to a released VQ level, decoded through the official tokenizer decoder. |
| Candidate claim | One sentence yielded by deterministic segmentation of a decoded output. |
| Supported | The NLI model assigns maximum retrieved-window entailment probability >= 0.70 and it exceeds maximum contradiction by >= 0.10. |
| Contradicted | The NLI model assigns maximum retrieved-window contradiction probability >= 0.70 and it exceeds maximum entailment by >= 0.10. |
| Unresolved | Neither threshold condition is met. This is epistemic uncertainty, not a claim of falsity. |
| Introduced | A candidate at the next level not matched to any candidate at the previous level above the lexical threshold. |
| Removed | A prior-level candidate not matched to a next-level candidate above the lexical threshold. |
| Matched label change | A matched pair receives different automated NLI labels. |
| Retained same label | A matched pair receives the same automated NLI label. |

The model output, support label and match status must be retained as separate variables. Do not collapse `unresolved` into `contradicted`, and do not name an automatically introduced candidate a hallucination without further evidence.

## Baselines and reporting

Report all levels, including counts and denominators. Include ROUGE as a reference-overlap metric only; do not use it as a factuality ground truth. For correlations, show magnitude/direction, sample size and cluster-aware uncertainty when practical. Include sensitivity analyses for NLI thresholds and matching threshold as predeclared secondary analyses if compute and time permit; these reuse saved model outputs and do not regenerate NCM generations.
