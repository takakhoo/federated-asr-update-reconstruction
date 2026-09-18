"""Known synthetic inverse problems, not captured client/private speech data."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.least_squares import recover_features


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows, examples = [], {}
    for label, frames, ill, noise in [("short identifiable", 8, False, 0), ("long underdetermined", 32, False, 0),
                                        ("ill-conditioned noisy", 8, True, 1e-4)]:
        g = rng.normal(size=(frames, 13))
        g -= g.mean(axis=1, keepdims=True)  # Logit gradients sum to zero over classes.
        if ill:
            u, s, vt = np.linalg.svd(g, full_matrices=False)
            g = (u * np.geomspace(1, 1e-8, len(s))) @ vt
        truth = rng.normal(size=(frames, 6))
        dw = g.T @ truth + noise*rng.normal(size=(13, 6))
        for ridge in (0, 1e-4) if ill else (0,):
            recovered, constant, report = recover_features(dw, g, ridge=ridge)
            report.update({"case": label, "relative_feature_error": float(np.linalg.norm(recovered-truth)/np.linalg.norm(truth)),
                           "actual_mean_error_of_constant_fit": float(np.linalg.norm(constant-truth.mean(axis=0)))})
            rows.append(report)
            examples[(label, ridge)] = (truth, recovered)
        if label == "short identifiable":
            torch.save({"dW": torch.from_numpy(dw), "dlogits": torch.from_numpy(g[None])}, output/"synthetic-grads.pt")
    report = {"seed": 7, "assumption": "per-frame dlogits supplied in addition to aggregate dW; not standard FedAvg access",
              "runs": rows}
    (output/"metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    labels = ["Short", "Long", "Ill-conditioned", "Ill + ridge"]
    axes[0].bar(labels, [max(r["relative_feature_error"], 1e-16) for r in rows], color=["#2166ac", "#b35806", "#b35806", "#2166ac"])
    axes[0].set(yscale="log", ylabel="Relative feature error", title="Identifiability matters more than fit")
    axes[0].tick_params(axis="x", rotation=20)
    a, b = examples[("long underdetermined", 0)]
    axes[1].plot(a[:, 0], label="True feature")
    axes[1].plot(b[:, 0], label="Minimum-norm estimate")
    axes[1].set(xlabel="Frame", ylabel="Feature value", title="Same gradients, different features")
    for axis in axes:
        axis.title.set_fontsize(11)
    axes[1].legend()
    fig.savefig(output/"identifiability.png", dpi=170)
    print(json.dumps([{k:r[k] for k in ("case", "rank", "nullity", "ridge", "relative_feature_error", "relative_gradient_residual")} for r in rows], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/ls"))
    run(parser.parse_args().output)
