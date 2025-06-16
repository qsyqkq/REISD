import torch
import numpy as np
import ot  # 使用 POT 库替代 pyemd

def batched_cdist_l2(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute batched L2 distance between x1 and x2.
    Args:
        x1, x2: torch.Tensor of shape [1, N, D]
    Returns:
        torch.Tensor of shape [1, N, N]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def compute_moverscore(prev_tokens, prev_embeddings: np.ndarray,
                       curr_tokens, curr_embeddings: np.ndarray,
                       weights_prev: np.ndarray, weights_curr: np.ndarray,
                       device: str = 'cpu') -> float:
    """
    Compute the MoverScore between two token sequences using POT (Python Optimal Transport).

    Args:
        prev_tokens (List[str]): Tokens for the reference sequence.
        prev_embeddings (np.ndarray): Embeddings array of shape (L_prev, D).
        curr_tokens (List[str]): Tokens for the hypothesis sequence.
        curr_embeddings (np.ndarray): Embeddings array of shape (L_curr, D).
        weights_prev (np.ndarray): Weights for prev_tokens, shape (L_prev,).
        weights_curr (np.ndarray): Weights for curr_tokens, shape (L_curr,).
        device (str): Device string for torch ('cpu' or 'cuda:0').

    Returns:
        float: MoverScore between the two sequences.
    """
    # Early exit on empty
    if prev_embeddings is None or curr_embeddings is None:
        return 0.0
    if prev_embeddings.shape[0] == 0 or curr_embeddings.shape[0] == 0:
        return 0.0

    # Convert to tensor and normalize
    e1 = torch.tensor(prev_embeddings, device=device, dtype=torch.float)
    e2 = torch.tensor(curr_embeddings, device=device, dtype=torch.float)
    e1 = e1 / (e1.norm(dim=1, keepdim=True) + 1e-30)
    e2 = e2 / (e2.norm(dim=1, keepdim=True) + 1e-30)

    # Compute pairwise L2 distance matrix between tokens from e1 and e2
    e1_exp = e1.unsqueeze(1)  # [L1, 1, D]
    e2_exp = e2.unsqueeze(0)  # [1, L2, D]
    dist = ((e1_exp - e2_exp) ** 2).sum(-1).sqrt().cpu().numpy()  # [L1, L2]

    # Normalize weights to form valid probability distributions
    w1 = np.asarray(weights_prev, dtype=np.double)
    w2 = np.asarray(weights_curr, dtype=np.double)
    w1 = w1 / (w1.sum() + 1e-30)
    w2 = w2 / (w2.sum() + 1e-30)

    # Compute EMD using POT
    flow = ot.emd(w1, w2, dist)  # [L1, L2]

    # Compute MoverScore (same formula as original)
    score = 1.0 / (1.0 + np.sum(flow * dist))
    return float(score)
