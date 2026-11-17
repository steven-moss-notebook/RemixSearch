//! Index rustdoc for the remixsearch crates in this workspace.

use docs_rs_builder::RustwideBuilder;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if let Err(err) = run() {
        eprintln!("docs-rs-builder: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut output = PathBuf::new();
    let mut pages = PathBuf::from("examples/docs/pages");
    let mut target = PathBuf::from("examples/docs/doc-target");
    let mut manifest = PathBuf::from("Cargo.toml");
    let mut inject_only = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => output = PathBuf::from(args.next().ok_or("--output needs a path")?),
            "--pages" => pages = PathBuf::from(args.next().ok_or("--pages needs a path")?),
            "--target" => target = PathBuf::from(args.next().ok_or("--target needs a path")?),
            "--manifest" => manifest = PathBuf::from(args.next().ok_or("--manifest needs a path")?),
            "--inject-only" => inject_only = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if inject_only {
        let docs = target.join("doc");
        let injected = RustwideBuilder::inject_search(&docs)?;
        eprintln!("added search to {injected} generated pages");
        return Ok(());
    }
    eprintln!("documenting {}", manifest.display());
    let docs = RustwideBuilder::document_workspace(&manifest, &target)?;
    // The Astro guide is the `remixsearch_wasm::astro` rustdoc page. Drop a
    // guide directory left behind by a builder that copied HTML in by hand.
    let _ = fs::remove_dir_all(docs.join("guides"));
    if pages.exists() {
        fs::remove_dir_all(&pages).map_err(|e| e.to_string())?;
    }
    let count = RustwideBuilder::stage_pages(&docs, &pages)?;
    install_queries(&pages)?;
    eprintln!("staged {count} remixsearch pages into {}", pages.display());
    if output.as_os_str().is_empty() {
        output = target.join("doc").join("search");
    }
    let remixsearch = env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or("missing exe directory")?
        .join("remixsearch");
    let status = Command::new(remixsearch)
        .args([
            "build",
            "--input",
            &pages.to_string_lossy(),
            "--output",
            &output.to_string_lossy(),
            "--demo",
            "--epochs",
            "80",
            "--max-pairs",
            "4096",
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("remixsearch build failed ({status})"));
    }
    let injected = RustwideBuilder::inject_search(&docs)?;
    eprintln!("added search to {injected} generated pages");
    Ok(())
}

fn install_queries(pages: &std::path::Path) -> Result<(), String> {
    let src = PathBuf::from("examples/docs/queries.json");
    fs::copy(&src, pages.join("queries.json")).map_err(|e| format!("copy {}: {e}", src.display()))?;
    Ok(())
}
