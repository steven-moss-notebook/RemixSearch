//! Browser searcher. [`RemixSearch::open`] fetches one brotli package and decompresses it.
//!
//! An Astro site wires that call in the [`astro`] module.

use remixsearch_format::{Package, PACKAGE_FILE};
use remixsearch_infer::Searcher;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Packed search index opened in the browser.
///
/// See [`astro`] for the build hook that writes the package an Astro site loads.
#[wasm_bindgen]
pub struct RemixSearch {
    searcher: Searcher,
}

#[wasm_bindgen]
impl RemixSearch {
    /// Fetch `index.br` from `base_url` and decompress the package.
    pub async fn open(base_url: String) -> Result<RemixSearch, JsValue> {
        let bytes = fetch_bytes(&base_url, PACKAGE_FILE).await?;
        let package = Package::open(&bytes).map_err(js_err)?;
        let searcher = Searcher::open(&package).map_err(js_err)?;
        Ok(RemixSearch { searcher })
    }

    /// Rank `k` pages for `query` and return the report as JSON.
    pub async fn search(&mut self, query: String, k: usize) -> Result<String, JsValue> {
        let outcome = self.searcher.search(&query, k);
        let hits: Vec<_> = outcome
            .hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "kind": match h.kind {
                        remixsearch_format::NodeKind::Image => "image",
                        remixsearch_format::NodeKind::Passage => "passage",
                    },
                    "title": h.title,
                    "url": h.url,
                    "imageUrl": h.image_url,
                    "alt": h.alt,
                    "score": h.score,
                    "cosine": h.cosine,
                    "matched": h.matched,
                    "snippet": h.snippet,
                })
            })
            .collect();
        let terms: Vec<_> = outcome
            .report
            .terms
            .iter()
            .map(|t| serde_json::json!({
                "term": t.term,
                "pieces": t.pieces,
                "unknown": t.unknown,
            }))
            .collect();
        let body = serde_json::json!({
            "summary": outcome.report.summary,
            "confident": outcome.report.confident,
            "candidates": outcome.report.candidates,
            "bestCosine": outcome.report.best_cosine,
            "secondCosine": outcome.report.second_cosine,
            "terms": terms,
            "hits": hits,
        });
        serde_json::to_string(&body).map_err(js_err)
    }
}

/// Set up a search index in an Astro site.
///
/// Add the `astro-remixsearch` integration. After `astro build`, its
/// `astro:build:done` hook reads each generated HTML page and packs a browser
/// index next to the site.
///
/// In `astro.config`, import `remixsearch` from `astro-remixsearch` and add it
/// to `integrations`. The hook walks the build directory, keeps the text inside
/// `main`, and keeps each image `src` and `alt`. It writes `remixsearch.jsonl`,
/// then runs `remixsearch build` with `--demo` into a `search/` directory.
///
/// On a page, load this crate's wasm package and call [`RemixSearch::open`]
/// with the base URL of that packed index, then [`RemixSearch::search`].
pub mod astro {}

async fn fetch_bytes(base: &str, path: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("window is unavailable"))?;
    let url = format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/'));
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(&url, &opts)?;
    let value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let response: Response = value.dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!("GET {url} returned {}", response.status())));
    }
    let buffer = JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

fn js_err(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}
