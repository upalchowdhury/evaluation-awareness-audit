"""Model loading and low-level forward/generate helpers (TransformerLens).

Design notes
------------
* All prompts go through the model's own chat template (`chat_wrap`), so cue
  text lands in the user turn exactly as a deployed chat request would.
* Batches are LEFT-padded so that position -1 is always the final content token
  and "positions counted from the end" align across a batch. TransformerLens
  receives the attention mask; if the installed version cannot accept one, we
  transparently fall back to per-item (batch=1) execution.
* Apple Silicon: prefers bfloat16 on MPS when available, else float16. Run with
  PYTORCH_ENABLE_MPS_FALLBACK=1 so rare unsupported ops fall back to CPU.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Sequence

import torch
from tqdm import tqdm

from .utils import batched, log

Hook = tuple[str, Callable]


# --------------------------------------------------------------------------- #
# Device / dtype selection
# --------------------------------------------------------------------------- #
def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str, pref: str = "auto") -> torch.dtype:
    if pref != "auto":
        return getattr(torch, pref)
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        try:  # bf16 needs macOS >= 14 / recent torch
            torch.zeros(2, dtype=torch.bfloat16, device="mps")
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32


def load_model(model_cfg):
    """Load a HookedTransformer according to the `model:` config block."""
    from transformer_lens import HookedTransformer

    device = pick_device(model_cfg.get("device", "auto"))
    dtype = pick_dtype(device, model_cfg.get("dtype", "auto"))
    log.info(f"loading {model_cfg.name} on {device} ({dtype}) ...")
    model = HookedTransformer.from_pretrained(model_cfg.name, device=device, dtype=dtype)
    model.eval()
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    log.info(
        f"loaded {model_cfg.name}: n_layers={model.cfg.n_layers}, "
        f"d_model={model.cfg.d_model}, n_heads={model.cfg.n_heads}, "
        f"device={device}, dtype={dtype}"
    )
    model._evalaware_supports_mask = None  # probed lazily
    return model


# --------------------------------------------------------------------------- #
# Chat formatting & tokenization
# --------------------------------------------------------------------------- #
def chat_wrap(model, user_msg: str, system_msg: str | None = None) -> str:
    """Render a (system, user) exchange with the model's chat template.

    Gemma models reject the system role; the system message is folded into the
    user turn there. Models without a chat template get a plain fallback.
    """
    tok = model.tokenizer
    name = getattr(model.cfg, "model_name", "") or ""
    is_gemma = "gemma" in name.lower()

    if getattr(tok, "chat_template", None):
        if system_msg and not is_gemma:
            msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass  # template rejects system role -> fold below
        merged = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
        msgs = [{"role": "user", "content": merged}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    sys_part = f"{system_msg}\n\n" if system_msg else ""
    bos = tok.bos_token or ""
    return f"{bos}{sys_part}User: {user_msg}\nAssistant:"


def chat_wrap_belief(model, user_msg: str, question: str, system_msg: str | None = None,
                     placeholder: str = "(answer omitted)", prefill: str = "Answer:") -> str:
    """Two-turn belief prompt:
         user: task  ->  assistant: `placeholder`  ->  user: `question`
         -> assistant reply prefilled with `prefill` (next token = the answer).
    The placeholder is fixed so eval/deploy renderings of one content share
    their whole suffix (needed for end-aligned patching)."""
    tok = model.tokenizer
    name = getattr(model.cfg, "model_name", "") or ""
    is_gemma = "gemma" in name.lower()
    if getattr(tok, "chat_template", None):
        def _msgs(fold_system: bool):
            first = f"{system_msg}\n\n{user_msg}" if (system_msg and fold_system) else user_msg
            out = []
            if system_msg and not fold_system:
                out.append({"role": "system", "content": system_msg})
            out += [{"role": "user", "content": first},
                    {"role": "assistant", "content": placeholder},
                    {"role": "user", "content": question}]
            return out
        try:
            text = tok.apply_chat_template(_msgs(fold_system=is_gemma), tokenize=False,
                                           add_generation_prompt=True)
        except Exception:  # template rejects the system role -> fold it
            text = tok.apply_chat_template(_msgs(fold_system=True), tokenize=False,
                                           add_generation_prompt=True)
        return text + prefill
    sys_part = f"{system_msg}\n\n" if system_msg else ""
    bos = tok.bos_token or ""
    return (f"{bos}{sys_part}User: {user_msg}\nAssistant: {placeholder}\n"
            f"User: {question}\nAssistant: {prefill}")


def batch_tokens(model, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded token ids + attention mask on the model device.

    add_special_tokens=False because chat templates already spell out their
    special tokens as text.
    """
    tok = model.tokenizer
    enc = tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
    dev = model.cfg.device
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev)


def single_tokens(model, text: str) -> torch.Tensor:
    tok = model.tokenizer
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return ids.to(model.cfg.device)


def _supports_mask(model) -> bool:
    if model._evalaware_supports_mask is None:
        try:
            ids, mask = batch_tokens(model, ["hello", "hello there friend"])
            with torch.no_grad():
                model(ids, attention_mask=mask)
            model._evalaware_supports_mask = True
        except TypeError:
            model._evalaware_supports_mask = False
            log.info("this transformer-lens version has no attention_mask kwarg; "
                     "falling back to per-item batches")
    return model._evalaware_supports_mask


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #
def final_logits(
    model,
    texts: Sequence[str],
    fwd_hooks: list[Hook] | None = None,
    batch_size: int = 8,
    progress: bool = False,
) -> torch.Tensor:
    """Next-token logits at the final position for each text. (n, d_vocab) fp32 cpu.

    Any fwd_hooks are active for every forward pass (used for steering /
    ablation; per-pair patching runs through interventions.py instead).
    """
    use_mask = _supports_mask(model)
    outs = []
    it = batched(list(texts), batch_size if use_mask else 1)
    if progress:
        total = (len(texts) + (batch_size if use_mask else 1) - 1) // (batch_size if use_mask else 1)
        it = tqdm(it, total=total, desc="forward", leave=False)
    for chunk in it:
        ids, mask = batch_tokens(model, chunk)
        hook_ctx = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else nullcontext()
        with torch.no_grad(), hook_ctx:
            if use_mask:
                logits = model(ids, attention_mask=mask)
            else:
                logits = model(ids)
        outs.append(logits[:, -1, :].float().cpu())
        del logits
    return torch.cat(outs, dim=0)


def generate_texts(
    model,
    texts: Sequence[str],
    max_new_tokens: int = 48,
    fwd_hooks: list[Hook] | None = None,
    progress: bool = True,
) -> list[str]:
    """Greedy continuations, one item at a time (robust across TL versions)."""
    tok = model.tokenizer
    outs = []
    it = tqdm(texts, desc="generate", leave=False) if progress else texts
    for t in it:
        ids = single_tokens(model, t)
        hook_ctx = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else nullcontext()
        with torch.no_grad(), hook_ctx:
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                verbose=False,
            )
        outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


# --------------------------------------------------------------------------- #
# Token-id utilities for logit readouts
# --------------------------------------------------------------------------- #
def first_token_variant_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """First-token ids for common surface variants of each word.

    Readouts sum probability over e.g. {"Yes", " Yes", "yes", " yes", "YES"} so
    they are robust to tokenizer space-handling and capitalization.
    """
    ids: set[int] = set()
    for w in words:
        variants = {w, w.lower(), w.upper(), w.capitalize()}
        variants |= {" " + v for v in list(variants)}
        for v in variants:
            t = tokenizer.encode(v, add_special_tokens=False)
            if not t:
                continue
            # SentencePiece tokenizers (Phi-3, Llama) encode " Yes" as a bare
            # space piece followed by "Yes"; the space piece carries no word
            # identity and would land in BOTH the Yes and the No group.
            if not tokenizer.decode([t[0]]).strip():
                continue
            ids.add(t[0])
    return sorted(ids)


def group_logit_score(logits: torch.Tensor, ids_a: list[int], ids_b: list[int]) -> torch.Tensor:
    """logsumexp(logits[A]) - logsumexp(logits[B]) per row. (n,) tensor."""
    a = torch.logsumexp(logits[:, ids_a], dim=-1)
    b = torch.logsumexp(logits[:, ids_b], dim=-1)
    return a - b


def group_prob(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Total softmax probability mass on `ids` per row."""
    return torch.softmax(logits, dim=-1)[:, ids].sum(dim=-1)
