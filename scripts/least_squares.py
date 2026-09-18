"""Auditable final-layer inverse problem: G.T @ H = dW.

Requires per-frame logit gradients G, which a normal FedAvg server does NOT
receive. A low gradient residual does not establish hidden-feature recovery.
"""
import numpy as np


def recover_features(dweight, dlogits, ridge=0.0, rtol=1e-10):
    dw = np.asarray(dweight, dtype=np.float64)
    gradients = np.asarray(dlogits, dtype=np.float64)
    if dw.ndim != 2 or gradients.ndim not in (2, 3) or gradients.shape[-1] != dw.shape[0]:
        raise ValueError("dW must be (vocab, hidden), dlogits (frames, vocab) or (batch, time, vocab)")
    if not dw.size or not gradients.size or not np.isfinite(dw).all() or not np.isfinite(gradients).all():
        raise ValueError("nonempty finite gradients required")
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("ridge >= 0 and 0 <= rtol < 1 required")
    g = gradients.reshape(-1, gradients.shape[-1])
    a = g.T
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    keep = singular > (singular[0] * rtol if len(singular) else 0)
    rank = int(keep.sum())
    factors = np.zeros_like(singular)
    if ridge:
        factors[keep] = singular[keep]/(singular[keep]**2 + ridge)
    else:
        factors[keep] = 1/singular[keep]
    recovered = (vt.T * factors) @ (u.T @ dw)
    projected = a @ recovered
    constant_direction = g.sum(axis=0)
    denominator = constant_direction @ constant_direction
    constant_fit = constant_direction @ dw / denominator if denominator > 0 else np.zeros(dw.shape[1])
    mean_direction = np.ones(len(g))/len(g)
    projected_mean = vt[keep].T @ (vt[keep] @ mean_direction)
    mean_unobserved = float(np.linalg.norm(mean_direction-projected_mean))
    relative_residual = float(np.linalg.norm(projected-dw)/max(np.linalg.norm(dw), 1e-30))
    report = {"frames": len(g), "vocabulary": dw.shape[0], "hidden_size": dw.shape[1],
              "rank": rank, "nullity": len(g)-rank, "ridge": ridge, "rtol": rtol,
              "relative_gradient_residual": relative_residual,
              "mean_identifiable": mean_unobserved < 1e-8,
              "mean_unobserved_component_norm": mean_unobserved,
              "zero_gradient_rows": np.flatnonzero(np.all(g == 0, axis=1)).tolist(),
              "singular_values": singular.tolist(),
              "interpretation": "minimum-norm/ridge estimate; constant-feature fit is NOT the true frame average"}
    return recovered.reshape(*gradients.shape[:-1], dw.shape[1]), constant_fit, report
