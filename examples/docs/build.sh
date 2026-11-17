#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
cargo build -p remixsearch-cli -p docs-rs-builder
"$root/target/debug/docs-rs-builder" \
  --manifest "$root/Cargo.toml" \
  --pages "$root/examples/docs/pages" \
  --target "$root/examples/docs/doc-target"
if command -v wasm-bindgen >/dev/null 2>&1; then
  rustup target add wasm32-unknown-unknown
  cargo build -p remixsearch-wasm --target wasm32-unknown-unknown
  wasm-bindgen --target web --out-dir "$root/examples/docs/doc-target/doc/pkg" \
    "$root/target/wasm32-unknown-unknown/debug/remixsearch_wasm.wasm"
fi
echo "open http://127.0.0.1:8765/ from a server rooted at examples/docs/doc-target/doc"
