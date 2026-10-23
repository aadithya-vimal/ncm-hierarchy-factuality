# Licensing and Data Governance

## Software license

`github/LICENSE` is MIT for original project software and documentation. It does not relicense the external NCM source repository, checkpoint weights, XSum data, NLI checkpoint or dependencies. The upstream NCM source repository and checkpoint release present MIT license metadata; retain attribution and verify the pinned release before use.

## Research dataset

Do not redistribute the original XSum article text. The local `private/source_articles.jsonl` is required for evaluation but is excluded from public export and artifact ZIPs. The export includes model-generated summaries, claim text, automated labels, metadata, transitions and metrics; generated text and derived artifacts may still have terms or rights considerations. Review upstream model/data terms and your intended research/publication use before pushing the dataset publicly. The package's publication script requires `--confirm-license-review` to make this decision explicit.

The dataset README deliberately describes its license scope separately from the software license. Do not set a blanket MIT label on third-party data, checkpoint weights or source articles. Record any upstream constraints and attributions in the dataset card before publication.

## Secrets

Use `hf auth login` or Colab Secrets for a Hugging Face token with write permission to the target dataset repository. Never commit `.env`, tokens, private data, output databases or cached model weights. The publication script reads credentials through `huggingface_hub` authentication and never prints them.

## Upstream NCM dependency

This project uses Next-Concept Modeling (NCM):
- Repository: https://github.com/HazyResearch/ncm
- Pinned commit: `c70274273f489abc73832b2b98584a8483b6b203`
- Upstream license: MIT

NCM is third-party software, not original code authored by this project. Its applicable license and copyright notices remain in force.

This project's MIT license applies to its original code only. It does not automatically cover third-party model weights, datasets, or dependencies.
