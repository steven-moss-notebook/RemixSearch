# remixsearch

Build a semantic search index for a static site. A teacher embeds every passage and image on the build machine. A small query encoder is distilled into that space, packed with the graph into one brotli file (`index.br`), and runs in Rust or in the browser. The browser encodes the typed query. Image vectors are computed at build time.

Ranking is cosine. The packed student is int8 and does not include Burn.

## Workspace

| Crate | Role |
| --- | --- |
| `remixsearch-format` | `index.br` package: manifest, vectors, HNSW graph, text, model, tokenizer |
| `remixsearch-infer` | Search and the int8 query encoder. Compiles to WASM |
| `remixsearch-build` | Ingest HTML, Markdown, JSONL, and images, then pack the index |
| `remixsearch-train` | Byte-level BPE, EmbeddingGemma 2 teacher, Burn distillation |
| `remixsearch-cli` | `remixsearch` binary |
| `remixsearch-wasm` | `RemixSearch.open` / `RemixSearch.search` |
| `docs-rs-builder` | `cargo doc` for this workspace, then a search box on the generated pages |
| `astro-remixsearch` | Astro integration that packs an index after `astro build` |

## Build

```sh
cargo build -p remixsearch-cli
cargo test --workspace --exclude remixsearch-wasm
```

`remixsearch-wasm` targets `wasm32-unknown-unknown`. `examples/docs/build.sh` builds it with `wasm-bindgen` when that CLI is installed.

## Index a site

```sh
remixsearch build --input ./site --output ./search
```

With no `--teacher` and no `--demo`, the build downloads `google/embeddinggemma-2` into the Hugging Face cache (`HF_HUB_CACHE`, or `~/.cache/huggingface`) if `model.safetensors` and `tokenizer.json` are not already there. `--teacher PATH` uses a local checkout. `--demo` uses a hash embedder and does not download the model.

```sh
remixsearch search --index ./search --query "how is the index packed"
remixsearch eval --index ./search --queries ./queries.jsonl
remixsearch compare --output ./compare --concepts 1 --epochs 80
```

`compare` writes a generated corpus, distills the hash embedder and EmbeddingGemma 2, and prints held-out recall.

## Docs example

`examples/docs/queries.json` is the question set. `examples/docs/build.sh` generates rustdoc, copies those questions onto the staged pages, and packs a `--demo` index. Serve `examples/docs/doc-target/doc`.

These paths are build output and are gitignored:

- `examples/docs/doc-target` — `cargo doc` tree
- `examples/docs/pages` — HTML staged for the indexer
- `examples/docs/search` — packed index, when written beside the example
- `examples/docs/pkg` — wasm-bindgen output
- `examples/docs/rustwide` — leftover copy of another crate's rustdoc
- `examples/docs/content` — old sample pages the builder does not read

## Astro

```js
import { remixsearch } from "astro-remixsearch";

export default { integrations: [remixsearch()] };
```

The hook writes `remixsearch.jsonl` into the Astro build directory and runs `remixsearch build --demo`. Load `RemixSearch` from the wasm package and open the `search/` directory it produced. The rustdoc on `remixsearch_wasm::astro` is the setup guide.

## Teacher check

`REMIXSEARCH_TEACHER=/path/to/embeddinggemma-2 cargo test -p remixsearch-train --lib golden_vectors` checks the Rust teacher against `testdata/gemma-query.txt` when that fixture exists. Without the variable the test returns immediately.

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.
