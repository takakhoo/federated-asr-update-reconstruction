#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_root="${FEDWAV2VEC_ROOT:?Set FEDWAV2VEC_ROOT to the fedwav2vec2 baseline root}"

python "$script_dir/make_tiny_client.py" --baseline-root "$baseline_root"

cd "$baseline_root"
mkdir -p data/server
for split in train dev test; do
  cp -f "data/client_0/ted_${split}.csv" "data/server/ted_${split}.csv"
done

CUDA_VISIBLE_DEVICES= python -m fedwav2vec2.main \
  rounds=1 \
  local_epochs=1 \
  total_clients=1 \
  strategy.min_fit_clients=1 \
  strategy.fraction_fit=1.0 \
  client_resources.num_gpus=0 \
  client_resources.num_cpus=4 \
  server_device=cpu \
  data_path=data \
  'dataset.extract_subdirectory=audio' \
  hooks.save_updates=true \
  hooks.save_grads=true

echo "Updates directory: $baseline_root/updates"
ls -lh updates
