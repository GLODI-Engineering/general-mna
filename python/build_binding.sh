#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cargo build --manifest-path "${repository_dir}/Cargo.toml" --release --features python
cp "${repository_dir}/target/release/libgeneral_mna.so" \
   "${repository_dir}/python/elspice_mna/_native.so"

PYTHONPATH="${repository_dir}/python" python3 -c \
  "import elspice_mna; print('loaded elspice_mna native binding')"
