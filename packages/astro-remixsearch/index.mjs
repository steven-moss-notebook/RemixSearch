import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

// After `astro build`, pull text and images out of the static pages and pack
// a search index next to them. The site's own image files stay where they are.
export function remixsearch(options = {}) {
  const binary = options.binary ?? "remixsearch";
  const outDir = options.outDir ?? "search";
  return {
    name: "remixsearch",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        const root = dir.pathname ?? dir;
        const pages = await walk(root, []);
        const rows = [];
        for (const file of pages) {
          if (!file.endsWith(".html")) continue;
          const html = await readFile(file, "utf8");
          const url = "/" + path.relative(root, file).replace(/index\.html$/, "").replace(/\\/g, "/");
          const title = match(html, /<title>([^<]*)<\/title>/i) || url;
          const main = match(html, /<main[\s\S]*?<\/main>/i) || match(html, /<body[\s\S]*?<\/body>/i) || html;
          const text = main.replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
          const images = [...main.matchAll(/<img\b[^>]*>/gi)].map((tag) => ({
            src: attr(tag[0], "src"),
            alt: attr(tag[0], "alt"),
            caption: "",
          })).filter((image) => image.src);
          if (text || images.length) rows.push({ url, title, headings: [], text, images });
        }
        const jsonl = path.join(root, "remixsearch.jsonl");
        await writeFile(jsonl, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
        const output = path.join(root, outDir);
        await mkdir(output, { recursive: true });
        await run(binary, ["build", "--input", jsonl, "--output", output, "--demo", "--epochs", "1", "--max-pairs", "512"]);
      },
    },
  };
}

function match(text, pattern) {
  const found = text.match(pattern);
  return found ? found[1] ?? found[0] : "";
}

function attr(tag, name) {
  const found = tag.match(new RegExp(name + '="([^"]*)"', "i"));
  return found ? found[1] : "";
}

async function walk(dir, out) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, out);
    else out.push(full);
  }
  return out;
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
  });
}
