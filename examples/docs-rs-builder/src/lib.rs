//! Generate rustdoc for this workspace, then a search index over those pages.
//!
//! Documentation is produced with `cargo doc` for the remixsearch crates. The
//! generated HTML is what gets the search box. Crate sources are not rewritten.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SKIP_DIRS: &[&str] = &["static.files", "src", "search", "pkg", "rustwide"];

/// Builds rustdoc for the remixsearch workspace and hands those pages to `remixsearch`.
pub struct RustwideBuilder;

impl RustwideBuilder {
    /// `cargo doc --workspace --no-deps` for the remixsearch crates.
    pub fn document_workspace(manifest: &Path, target_dir: &Path) -> Result<PathBuf, String> {
        let mut cmd = cargo_command();
        cmd.arg("doc")
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--workspace")
            .arg("--no-deps")
            .arg("--target-dir")
            .arg(target_dir);
        let status = cmd.status().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("workspace documentation failed ({status})"));
        }
        let docs = target_dir.join("doc");
        if !docs.join("remixsearch_infer").is_dir() {
            return Err(format!("rustdoc did not write {}", docs.display()));
        }
        let _ = fs::remove_dir_all(docs.join("rustwide"));
        Ok(docs)
    }

    /// Copy the item pages into `pages`, keeping module paths and skipping `all.html`.
    pub fn stage_pages(docs: &Path, pages: &Path) -> Result<usize, String> {
        let mut count = 0;
        stage_tree(docs, docs, pages, &mut count)?;
        if count == 0 {
            return Err(format!("no html pages in {}", docs.display()));
        }
        Ok(count)
    }

    /// Insert the search box into generated rustdoc pages. The crate sources are not modified.
    pub fn inject_search(docs: &Path) -> Result<usize, String> {
        let mut count = 0;
        inject_tree(docs, &mut count)?;
        Ok(count)
    }
}

pub fn inject_search_html(html: &str) -> String {
    let html = strip_search_markup(html);
    let at = html
        .find("</rustdoc-topbar>")
        .map(|i| i + "</rustdoc-topbar>".len())
        .or_else(|| html.rfind("</body>"));
    let Some(at) = at else {
        return html;
    };
    let mut out = String::with_capacity(html.len() + SEARCH_MARKUP.len());
    out.push_str(&html[..at]);
    out.push_str(SEARCH_MARKUP);
    out.push_str(&html[at..]);
    out
}

fn strip_search_markup(html: &str) -> String {
    let Some(start) = html.find("<div id=\"remixsearch-box\">") else {
        return html.to_string();
    };
    let Some(rel) = html[start..].find("</script>") else {
        return html.to_string();
    };
    let mut end = start + rel + "</script>".len();
    if html[end..].starts_with('\n') {
        end += 1;
    }
    let mut out = String::with_capacity(html.len());
    out.push_str(&html[..start]);
    out.push_str(&html[end..]);
    out
}

const SEARCH_MARKUP: &str = include_str!("../search-snippet.html");

fn inject_tree(dir: &Path, count: &mut usize) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if SKIP_DIRS.contains(&name.to_string_lossy().as_ref()) {
                continue;
            }
            inject_tree(&path, count)?;
            continue;
        }
        if !name.to_string_lossy().ends_with(".html") {
            continue;
        }
        let html = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let next = inject_search_html(&html);
        if next != html {
            fs::write(&path, next).map_err(|e| e.to_string())?;
            *count += 1;
        }
    }
    Ok(())
}

fn stage_tree(root: &Path, dir: &Path, pages: &Path, count: &mut usize) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if SKIP_DIRS.contains(&name.to_string_lossy().as_ref()) {
                continue;
            }
            stage_tree(root, &path, pages, count)?;
            continue;
        }
        let name = name.to_string_lossy();
        if !name.ends_with(".html") || name == "all.html" {
            continue;
        }
        let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
        let dest = pages.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::copy(&path, &dest).map_err(|e| e.to_string())?;
        *count += 1;
    }
    Ok(())
}

fn cargo_command() -> Command {
    if let Some(path) = std::env::var_os("CARGO") {
        return Command::new(path);
    }
    if let Ok(home) = registry_src() {
        let bin = home.join("..").join("..").join("bin").join("cargo");
        if bin.is_file() {
            return Command::new(bin);
        }
    }
    Command::new("cargo")
}

pub fn registry_src() -> Result<PathBuf, String> {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .ok_or("CARGO_HOME is not set")?;
    Ok(cargo_home.join("registry").join("src"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_is_inserted_into_generated_html_once() {
        let html = "<html><body><p>WorkspaceBuilder</p></body></html>";
        let once = inject_search_html(html);
        assert!(once.contains("id=\"remixsearch-box\""));
        assert!(once.contains("WorkspaceBuilder"));
        assert_eq!(inject_search_html(&once), once);
        let rustdoc = "<rustdoc-topbar><h2>rustwide</h2></rustdoc-topbar><main>body</main>";
        let placed = inject_search_html(rustdoc);
        let top = placed.find("</rustdoc-topbar>").unwrap();
        let box_at = placed.find("id=\"remixsearch-box\"").unwrap();
        assert!(box_at > top);
        assert!(box_at < placed.find("<main>").unwrap());
        assert!(once.contains("color:#1a1814"));
        assert!(!once.contains("rustwide/"));
    }
}
