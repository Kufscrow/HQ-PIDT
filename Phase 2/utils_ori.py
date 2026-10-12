import random
import numpy as np
import torch
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def _crop_history(x, max_context):
    if x is None:
        return None
    return x if x.size(1) <= max_context else x[:, -max_context:]


@torch.inference_mode()
def sample(
    model, x, steps, temperature=1.0, sample=False, top_k=None,
    actions=None, rtgs=None, timesteps=None,
):
    """Decision-Transformer one/few-step action sampling.

    Optimization details
    --------------------
    * Synchronously crops states/actions/RTGs/timesteps to the same trajectory window.
    * Preserves per-state timestep history, which makes old token embeddings stable and
      allows the model's quantum-aware KV cache to remain valid until window rollover.
    * Uses torch.inference_mode() rather than no_grad().
    * Avoids softmax entirely for greedy decoding because argmax(softmax(z))=argmax(z).

    In the present power-system evaluator ``steps`` is normally 1.
    """
    block_size = model.get_block_size()
    max_context = block_size // 3

    for _ in range(steps):
        x_cond = _crop_history(x, max_context)
        actions_cond = _crop_history(actions, max_context)
        rtgs_cond = _crop_history(rtgs, max_context)
        timesteps_cond = _crop_history(timesteps, max_context)

        logits, _ = model(
            x_cond, actions=actions_cond, targets=None,
            rtgs=rtgs_cond, timesteps=timesteps_cond,
        )
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        if sample:
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
        else:
            ix = torch.argmax(logits, dim=-1, keepdim=True)

        # The environment supplies the next physical state.  For steps>1 we keep
        # the legacy return convention, but normal Decision-Transformer use here
        # calls sample(..., steps=1).
        x = ix

    return x
