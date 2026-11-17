"""Restricted next-token scoring of the conclusion label (yes / no / maybe).

For each rendered prompt we apply the model's chat template, append the assistant
generation prefix and read the next-token distribution. The probability of each
label is the summed probability of its single-token surface forms
(e.g. "yes", "Yes", "YES"). We return:
  p_yes, p_no, p_maybe : renormalised over the three labels (sum to 1)
  label_mass           : total probability on any label token before renormalisation
This avoids free-text parsing (a known evaluation artifact, Hua et al. 2025) and is
deterministic for a fixed model, dtype and device.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import LABELS

# Fixed date for Llama-3.x chat templates, which otherwise inject today's date into the
# system header and silently change the prompt from day to day.
FIXED_DATE = "17 Nov 2026"
SURFACE = {"yes": ["yes", "Yes", "YES"], "no": ["no", "No", "NO"], "maybe": ["maybe", "Maybe", "MAYBE"]}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_local(model_id: str, revision: str | None) -> str:
    """Return the local snapshot directory for a pinned revision (downloads it if absent)."""
    from huggingface_hub import snapshot_download
    try:  # prefer the cached snapshot; avoids network calls and gated-repo re-checks
        return snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception:
        return snapshot_download(model_id, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


class LabelScorer:
    def __init__(self, model_id: str, revision: str | None = None, device: str | None = None,
                 dtype: torch.dtype = torch.float16):
        self.model_id = model_id
        self.device = device or pick_device()
        if self.device == "cpu":
            dtype = torch.float32
        path = resolve_local(model_id, revision)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        self.model.to(self.device).eval()
        self.label_token_ids = self._label_tokens()

    def _label_tokens(self) -> dict[str, list[int]]:
        out = {}
        for lab, forms in SURFACE.items():
            ids = []
            for f in forms:
                enc = self.tok.encode(f, add_special_tokens=False)
                if len(enc) == 1:
                    ids.append(enc[0])
            assert ids, f"no single-token surface form for {lab} in {self.model_id}"
            out[lab] = sorted(set(ids))
        # no token may be shared across labels
        flat = [i for v in out.values() for i in v]
        assert len(flat) == len(set(flat))
        return out

    def chat_text(self, user_text: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True, date_string=FIXED_DATE)

    @torch.inference_mode()
    def score(self, user_texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Return array (n, 4): p_yes, p_no, p_maybe (renormalised), label_mass."""
        res = np.zeros((len(user_texts), 4), dtype=np.float64)
        # sort by length to minimise padding; restore order afterwards
        texts = [self.chat_text(t) for t in user_texts]
        order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            lab = torch.stack([probs[:, self.label_token_ids[l]].sum(-1) for l in LABELS], dim=-1)
            mass = lab.sum(-1, keepdim=True)
            out = torch.cat([lab / mass, mass], dim=-1).cpu().numpy()
            res[idx] = out
        return res


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t
