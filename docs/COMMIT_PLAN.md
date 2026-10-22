# Meaningful Atomic Commit Plan

Use commits to preserve coherent implementation changes, not to inflate the count with empty or arbitrary micro-commits. If the ZIP is your starting point, the staged commit plan below lets you establish a readable history from the already-present files.

Run commands from `NCM Research/github/`. Set your remote URL only after creating an empty GitHub repository without an auto-generated README/license.

```bash
git init -b main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ncm-hierarchy-factuality.git
```

Suggested atomic sequence (staging path groups by concern):

```bash
# 1. Repository contract, license, package metadata and ignore rules
git add README.md .gitignore LICENSE CITATION.cff pyproject.toml requirements-colab.txt
git commit -m "Initialize research repository and licensing metadata"

# 2. Scientific diagrams
git add assets/
git commit -m "Add research pipeline and hierarchy diagrams"

# 3. Research and data governance docs
git add docs/
git commit -m "Document protocol, schema, reproducibility, and licensing"

# 4. Predeclared experiment configuration
git add configs/
git commit -m "Define full XSum experiment and evaluation thresholds"

# 5. General checkpoint/provenance/data helpers
git add src/ncm_factuality/__init__.py src/ncm_factuality/config.py src/ncm_factuality/checkpoint.py src/ncm_factuality/provenance.py src/ncm_factuality/io_utils.py
git commit -m "Add resumable storage and experiment provenance primitives"

# 6. Official NCM integration and hierarchy generation
git add src/ncm_factuality/ncm_adapter.py src/ncm_factuality/input_data.py src/ncm_factuality/inference.py
git commit -m "Integrate pinned NCM checkpoint and persist every hierarchy level"

# 7. Claim extraction and NLI evidence scoring
git add src/ncm_factuality/claim_extraction.py src/ncm_factuality/factuality_evaluation.py src/ncm_factuality/annotation.py
git commit -m "Implement cached source-evidence factuality annotation"

# 8. Cross-level claim matching and statistics
git add src/ncm_factuality/claim_tracking.py src/ncm_factuality/lifecycle.py src/ncm_factuality/statistics.py src/ncm_factuality/analysis.py src/ncm_factuality/visualization.py
git commit -m "Add claim lifecycle matching and article-cluster analysis"

# 9. Reusable dataset export and explicit Hub publishing
git add src/ncm_factuality/dataset_export.py src/ncm_factuality/huggingface_publish.py src/ncm_factuality/cli.py
git commit -m "Add validated dataset export and opt-in Hub publishing"

# 10. Synthetic CPU-only regression tests
git add tests/
git commit -m "Test lifecycle, recovery, schema, and export invariants"

# 11. Colab execution interface
git add notebooks/
git commit -m "Add ordered Colab notebooks for the complete experiment"

# 12. Command-line scripts and CI
git add scripts/ .github/
git commit -m "Add validation utilities and CPU-only CI workflow"

# Review the history, then push
git log --oneline --decorate
git push -u origin main
```

If a path named above is not present in your particular ZIP version, inspect with `git status` before staging. Do not use `git add .` in `local_outputs/` or from the parent `NCM Research/` folder.

## Hugging Face repository history

Initialize and push `huggingface/` separately, after the dataset export and licensing review. First commit documentation/schema, then the validated data in a separate commit, then versioned follow-up updates. Dataset data should not be split into artificial commits just to increase counts.
