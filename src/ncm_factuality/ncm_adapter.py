"""Thin adapter for the *official* HazyResearch/NCM implementation.

This module intentionally does not reimplement the model. The exact upstream
source revision is checked out by notebook 00. Intermediate outputs are decoded
from the cumulative code levels produced by the released checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class NCMAdapter:
    def __init__(self, upstream_root: str | Path, checkpoint_dir: str | Path, settings: dict[str, Any]):
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - runtime installation path
            raise RuntimeError("PyTorch/OmegaConf not available; run notebook 00 first.") from exc
        self.torch = torch
        self.upstream_root = Path(upstream_root).resolve()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.settings = settings
        if not self.upstream_root.exists() or not (self.upstream_root / "models" / "next_concept.py").exists():
            raise FileNotFoundError(f"Pinned NCM source was not found under {self.upstream_root}; run notebook 00.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime before loading NCM weights.")
        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))

        # Import the same model/tokenizer classes used by upstream scripts/eval_shard.py.
        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        self.cfg = OmegaConf.load(str(self.checkpoint_dir / "config.yaml"))
        self.device = torch.device("cuda:0")
        self.eos_id = None
        self.pad_id = None
        self.allowance = None
        self.adjacency = None

        self.cfg.tokenizer.vqvae.checkpoint_path = str(self.checkpoint_dir / str(self.cfg.checkpoints.tokenizer))
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if self.cfg.checkpoints.get(name):
                self.cfg.tokenizer.pre_tokenizer[key] = str(self.checkpoint_dir / str(self.cfg.checkpoints[name]))

        self.tokenizer = VQVAETokenizer.from_config(self.cfg.tokenizer)
        self.tokenizer.vqvae = self.tokenizer.vqvae.to(self.device).eval()
        self.model = NextConceptTransformer(vqvae_local=self.tokenizer.vqvae, config=self.cfg.model).to(self.device)
        state_path = self.checkpoint_dir / str(self.cfg.checkpoints.generator)
        state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(f"Unexpected NCM generator checkpoint layout in {state_path}; expected key 'model'.")
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

        # Match the official eval_shard.py defaults and token-picking behavior.
        gen = self.cfg.generation
        for key in ("temperature", "top_k", "top_p", "cfg_scale", "frequency_penalty", "frequency_deficit", "adjacency_penalty", "frequency_allowance"):
            if key in settings:
                gen[key] = settings[key]
        self.temperature = float(gen.get("temperature", 1.0))
        self.top_k = int(gen.get("top_k", 1))
        self.top_p = float(gen.get("top_p", 0.0))
        self.cfg_scale = float(gen.get("cfg_scale", 0.0))
        self.frequency_penalty = float(gen.get("frequency_penalty", 0.0))
        self.frequency_deficit = float(gen.get("frequency_deficit", 0.0))
        adjacency = str(gen.get("adjacency_penalty", "") or "")
        self.adjacency = [float(v) for v in adjacency.split(",")] if adjacency else None
        allowance_spec = str(gen.get("frequency_allowance", "") or "")
        if allowance_spec:
            if ":" in allowance_spec:
                path, ref = allowance_spec.split(":", 1)
                obj = state if path == str(self.cfg.checkpoints.generator) else torch.load(str(self.checkpoint_dir / path), map_location="cpu", weights_only=False)
                for key in ref.split("."):
                    obj = obj[key]
                self.allowance = obj.to(device=self.device, dtype=torch.float64)
            else:
                self.allowance = torch.load(str(self.checkpoint_dir / allowance_spec), map_location=self.device, weights_only=False).double()
        del state
        self.eos_id = int(self.tokenizer.pre_tokenizer.eos_token_id)
        self.pad_id = int(self.tokenizer.pre_tokenizer.pad_token_id)
        self.prefix_len = int(self.cfg.model.prefix_len)
        self.level_lengths = [int(v) for v in self.tokenizer.vqvae.vq.levels]
        if len(self.level_lengths) < 2:
            raise RuntimeError(f"Expected multiple NCM hierarchy levels, got {self.level_lengths}")

    def _prefix(self, item: Any) -> list[int]:
        if isinstance(item, str):
            ids = self.tokenizer.pre_tokenizer.encode(item)
        else:
            ids = item.tolist() if hasattr(item, "tolist") else list(item)
            ids = [int(x) for x in ids]
        return (ids[: self.prefix_len] + [self.pad_id] * self.prefix_len)[: self.prefix_len]

    def _pick_tokens(self, logits):
        """Port of official scripts/eval_shard.py's deterministic frequency-aware token picker."""
        torch = self.torch
        freq = self.frequency_penalty
        mu = self.frequency_deficit
        if freq <= 0 and not self.adjacency:
            return logits.argmax(-1)
        batch, length, vocab = logits.shape
        scores = logits.double()
        allowance = (self.allowance[:vocab] * length)[None, :] if self.allowance is not None else None
        adj_value = self.adjacency[0] if self.adjacency else 0.0
        adj_window = int(self.adjacency[1]) if self.adjacency and len(self.adjacency) > 1 else 1
        adj_decay = self.adjacency[2] if self.adjacency and len(self.adjacency) > 2 else 1.0
        counts = torch.zeros(batch, vocab, dtype=torch.float64, device=scores.device)
        output = torch.empty(batch, length, dtype=torch.long, device=scores.device)
        history = []
        for position in range(length):
            step = scores[:, position, :].clone()
            if freq > 0:
                excess = (counts - allowance).clamp_min(0.0) if allowance is not None else counts
                step -= freq * excess
                if mu > 0 and allowance is not None:
                    step += mu * ((length - position) / length) * (allowance - counts).clamp_min(0.0)
            for j, ids in enumerate(reversed(history[-adj_window:])):
                step.scatter_add_(1, ids[:, None], torch.full((batch, 1), -adj_value * adj_decay**j, dtype=torch.float64, device=scores.device))
            picked = step.argmax(-1)
            output[:, position] = picked
            history.append(picked)
            counts.scatter_add_(1, picked[:, None], torch.ones(batch, 1, dtype=torch.float64, device=scores.device))
        return output

    def _decode_level(self, z_level) -> list[str]:
        torch = self.torch
        with torch.inference_mode():
            logits = self.tokenizer.vqvae.decoder(z_level)
            ids_batch = self._pick_tokens(logits).tolist()
        decoded = []
        cut_at_eos = bool(self.cfg.generation.get("cut_at_eos", True))
        for token_ids in ids_batch:
            token_ids = [int(v) for v in token_ids]
            if cut_at_eos and self.eos_id in token_ids:
                token_ids = token_ids[: token_ids.index(self.eos_id)]
            token_ids = [v for v in token_ids if v not in (self.eos_id, self.pad_id)]
            text = self.tokenizer.pre_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            decoded.append(text)
        return decoded

    def generate_batch(self, items: list[Any]) -> list[list[dict[str, Any]]]:
        """Generate a list of cumulative decoded texts per input (one list per article)."""
        if not items:
            return []
        torch = self.torch
        prefix = torch.tensor([self._prefix(item) for item in items], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            # Exact call shape used by the official script for query_level0 XSum checkpoints.
            codes = self.model.generate(
                None,
                prefix,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                cfg_scale=self.cfg_scale,
                cfg_null_eos=self.eos_id,
            )
            _, z_levels = self.tokenizer.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            by_level = [self._decode_level(z_level) for z_level in z_levels]
        if len(by_level) != len(self.level_lengths):
            raise RuntimeError(f"Decoded {len(by_level)} levels but checkpoint describes {len(self.level_lengths)} levels")
        outputs: list[list[dict[str, Any]]] = [[] for _ in items]
        for level_index, (level_length, texts) in enumerate(zip(self.level_lengths, by_level), start=1):
            if len(texts) != len(items):
                raise RuntimeError("NCM batch output size does not match input batch size")
            for row_index, text in enumerate(texts):
                outputs[row_index].append({
                    "level_index": level_index,
                    "level_code_length": int(level_length),
                    "summary_text": text,
                })
        return outputs

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
        finally:
            self.torch.cuda.empty_cache()
