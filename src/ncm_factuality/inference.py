"""Resume-safe full-split generation and JSONL export."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .checkpoint import connect_db, completed_article_ids, export_hierarchy_jsonl, save_article
from .io_utils import atomic_write_json, read_jsonl_recover, sha256_file
from .provenance import write_manifest


def _level_lengths(checkpoint_dir: Path) -> list[int]:
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(checkpoint_dir / "config.yaml"))
        levels = list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
        if not levels:
            levels = list(range(1, len(cfg.tokenizer.vqvae.vqvae_vocab_size) + 1))
        return [int(x) for x in levels]
    except Exception as exc:
        raise RuntimeError(f"Could not read the official XSum checkpoint's hierarchy levels from {checkpoint_dir / 'config.yaml'}") from exc


def _generate_with_oom_splitting(adapter, batch_items: list[Any]) -> list[list[dict[str, Any]]]:
    """Only subdivide a batch on CUDA OOM; never silently skip a failing example."""
    try:
        return adapter.generate_batch(batch_items)
    except Exception as exc:
        message = str(exc).casefold()
        is_oom = "out of memory" in message or isinstance(exc, getattr(adapter.torch.cuda, "OutOfMemoryError", ()))
        if not is_oom or len(batch_items) <= 1:
            raise
        adapter.torch.cuda.empty_cache()
        midpoint = len(batch_items) // 2
        return _generate_with_oom_splitting(adapter, batch_items[:midpoint]) + _generate_with_oom_splitting(adapter, batch_items[midpoint:])


def run_full_inference(
    cfg: dict[str, Any], paths: dict[str, Path], upstream_root: str | Path, inputs: dict[str, Any]
) -> dict[str, Any]:
    run_dir = paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(inputs["checkpoint_dir"])
    data_dir = Path(inputs["data_dir"])
    passages = inputs["passages"]
    source_rows = sorted(inputs["source_records"], key=lambda r: int(r["ncm_index"]))
    if len(passages) != len(source_rows):
        raise ValueError(f"Alignment mismatch: {len(passages)} NCM passages vs {len(source_rows)} aligned source rows")
    level_lengths = _level_lengths(checkpoint_dir)
    run_fingerprint = str(paths["fingerprint"].name)
    output_db = run_dir / "research.sqlite3"
    connection = connect_db(output_db)
    expected_ids = {str(row["article_id"]) for row in source_rows}
    completed = completed_article_ids(connection, len(level_lengths)) & expected_ids
    pending_indices = [i for i, row in enumerate(source_rows) if str(row["article_id"]) not in completed]
    print(f"NCM inference plan: {len(source_rows):,} aligned articles; {len(level_lengths)} levels; {len(completed):,} fully checkpointed; {len(pending_indices):,} remaining.")

    adapter = None
    started = time.time()
    batch_size = int(cfg["inference"]["batch_size"])
    output_rows_total = 0
    model_id = f"{cfg['upstream']['checkpoint_repo_id']}/{cfg['upstream'].get('checkpoint_subdir','xsum')}"
    try:
        for start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[start : start + batch_size]
            batch_items = [passages[i] for i in batch_indices]
            if adapter is None:
                from .ncm_adapter import NCMAdapter
                adapter = NCMAdapter(
                    upstream_root=upstream_root,
                    checkpoint_dir=checkpoint_dir,
                    settings=cfg["inference"],
                )
                if adapter.level_lengths != level_lengths:
                    raise RuntimeError(f"Checkpoint levels disagree: YAML={level_lengths}, model={adapter.level_lengths}")
            generated_batch = _generate_with_oom_splitting(adapter, batch_items)
            if len(generated_batch) != len(batch_indices):
                raise RuntimeError("NCM adapter returned a different number of article outputs than inputs")
            for local_idx, per_level in zip(batch_indices, generated_batch):
                source_record = source_rows[local_idx]
                if len(per_level) != len(level_lengths):
                    raise RuntimeError(f"Article {source_record['article_id']} has {len(per_level)} levels; expected {len(level_lengths)}")
                records = []
                for level in per_level:
                    records.append({
                        "article_id": str(source_record["article_id"]),
                        "ncm_index": int(local_idx),
                        "level_index": int(level["level_index"]),
                        "level_code_length": int(level["level_code_length"]),
                        "summary_text": str(level["summary_text"]),
                        "run_fingerprint": run_fingerprint,
                        "model_id": model_id,
                        "checkpoint_revision": str(cfg["upstream"]["checkpoint_revision"]),
                        "upstream_commit": str(cfg["upstream"]["commit"]),
                    })
                save_article(connection, records)  # atomic per-article transaction
                output_rows_total += len(records)
            done_now = len(completed_article_ids(connection, len(level_lengths)) & expected_ids)
            if done_now % max(batch_size * 10, 1) == 0 or start + len(batch_indices) >= len(pending_indices):
                print(f"Checkpointed {done_now:,}/{len(source_rows):,} articles ({done_now/len(source_rows):.1%}).", flush=True)
    except Exception as exc:
        atomic_write_json(run_dir / "inference_failure.json", {
            "stage": "ncm_inference",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_articles_before_failure": len(completed_article_ids(connection, len(level_lengths))),
            "next_step": "Rerun notebook 01. Completed articles will be skipped; inspect this error before retrying if it persists.",
        })
        raise
    finally:
        if adapter is not None:
            adapter.close()
        connection.close()

    connection = connect_db(output_db)
    final_complete = completed_article_ids(connection, len(level_lengths)) & expected_ids
    output_path = run_dir / "hierarchy_outputs.jsonl"
    export_hierarchy_jsonl(connection, output_path)
    total = connection.execute("SELECT COUNT(*) FROM hierarchy_outputs").fetchone()[0]
    connection.close()
    if len(final_complete) != len(source_rows):
        raise RuntimeError(f"Inference stage incomplete: {len(final_complete)}/{len(source_rows)} articles have all hierarchy levels")
    manifest = {
        "experiment_name": cfg["project"]["name"],
        "run_name": cfg["project"]["run_name"],
        "run_fingerprint": run_fingerprint,
        "stage": "inference_complete",
        "n_articles": len(source_rows),
        "n_levels": len(level_lengths),
        "level_code_lengths": level_lengths,
        "n_hierarchy_rows": int(total),
        "model_id": model_id,
        "checkpoint_revision": cfg["upstream"]["checkpoint_revision"],
        "upstream_commit": cfg["upstream"]["commit"],
        "ncm_data_repo_id": cfg["upstream"]["data_repo_id"],
        "resolved_data_revisions": cfg.get("_resolved_revisions", {}),
        "local_project_commit": __import__("ncm_factuality.provenance", fromlist=["git_commit"]).git_commit(paths["project_root"]),
        "source_alignment_report": str((run_dir / "input_alignment.json").name),
        "output_jsonl": output_path.name,
        "output_sha256": sha256_file(output_path),
        "elapsed_seconds_this_invocation": round(time.time() - started, 2),
        "completed_previously": len(completed),
        "new_level_rows_written_this_invocation": output_rows_total,
        "note": "Elapsed time is for this invocation only; no prior runtime is inferred.",
    }
    write_manifest(run_dir / "experiment_manifest.json", manifest)
    print(f"Inference complete: {total:,} level rows in {output_path}; NCM generation is not needed by later notebooks.")
    return manifest
