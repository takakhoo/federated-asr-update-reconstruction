#!/usr/bin/env bash
set -euo pipefail

baseline_root="${FEDWAV2VEC_ROOT:?Set FEDWAV2VEC_ROOT to the fedwav2vec2 baseline root}"
cd "$baseline_root"

echo "Baseline root: $PWD"
echo "Expected files:"
for path in \
  fedwav2vec2/main.py \
  fedwav2vec2/client.py \
  fedwav2vec2/server.py \
  fedwav2vec2/strategy.py \
  fedwav2vec2/sb_recipe.py \
  fedwav2vec2/models.py \
  fedwav2vec2/dataset.py \
  fedwav2vec2/conf/base.yaml \
  fedwav2vec2/conf/sb_config/w2v2.yaml; do
  if [[ -f "$path" ]]; then
    printf '  [ok] %s\n' "$path"
  else
    printf '  [missing] %s\n' "$path"
  fi
done

echo
echo "Capture-hook references:"
rg -n "ctc_lin|save_updates|save_grads|backward\(\)" fedwav2vec2 || true
