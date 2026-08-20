//! Synthetic + on-disk seed snippets used to materialize JSONL when needed.

use std::path::{Path, PathBuf};

use rand::prelude::*;
use serde::Serialize;

const PYTHON_TEMPLATES: &[&str] = &[
    "from pathlib import Path\n\ndef {fn}(path: str) -> Path:\n    p = Path(path)\n    if not p.exists():\n        raise FileNotFoundError(path)\n    return p\n",
    "import subprocess\n\ndef {fn}(cmd: list[str]) -> str:\n    r = subprocess.run(cmd, check=True, capture_output=True, text=True)\n    return r.stdout.strip()\n",
    "def {fn}(items: list[str]) -> list[str]:\n    out = []\n    for x in items:\n        if x.startswith('{pfx}'):\n            out.append(x.strip())\n    return out\n",
    "class {cls}:\n    def __init__(self, root: str):\n        self.root = root\n\n    def {fn}(self, name: str) -> str:\n        return f'{{self.root}}/{{name}}'\n",
    "if __name__ == '__main__':\n    for line in open('{file}'):\n        if '{kw}' in line:\n            print(line.strip())\n",
    "try:\n    data = open('{file}').read()\nexcept OSError as exc:\n    raise SystemExit(exc)\n",
];

const RUST_TEMPLATES: &[&str] = &[
    "fn {fn}(path: &str) -> Result<String, std::io::Error> {{\n    std::fs::read_to_string(path)\n}}\n",
    "fn {fn}(xs: &[i32]) -> i32 {{\n    xs.iter().sum()\n}}\n",
    "fn main() {{\n    match {fn}(\"{file}\") {{\n        Ok(s) => println!(\"{{s}}\"),\n        Err(e) => eprintln!(\"{{e}}\"),\n    }}\n}}\n",
    "pub fn {fn}(cmd: &str) -> Result<(), Box<dyn std::error::Error>> {{\n    let status = std::process::Command::new(\"sh\").arg(\"-c\").arg(cmd).status()?;\n    if status.success() {{ Ok(()) }} else {{ Err(\"fail\".into()) }}\n}}\n",
    "#[derive(Debug)]\nstruct {cls} {{\n    name: String,\n}}\n\nimpl {cls} {{\n    fn new(name: &str) -> Self {{\n        Self {{ name: name.into() }}\n    }}\n}}\n",
    "for item in items.iter() {{\n    if item.starts_with(\"{pfx}\") {{\n        println!(\"{{item}}\");\n    }}\n}}\n",
];

const BASH_TEMPLATES: &[&str] = &[
    "#!/usr/bin/env bash\nset -euo pipefail\n{fn}() {{\n  local f=\"$1\"\n  if [[ -f \"$f\" ]]; then cat \"$f\"; fi\n}}\n",
    "for f in {glob}; do\n  {cmd} \"$f\"\ndone\n",
    "if [[ -d \"{dir}\" ]]; then\n  find \"{dir}\" -name '{glob}' | while read -r p; do echo \"$p\"; done\nfi\n",
    "{cmd} {args} | {cmd2} {args2}\n",
    "git status\ngit diff --stat\n",
    "python3 -m ullis chat --ckpt checkpoints/packed.pt\n",
    "cargo test --offline && cargo clippy --all-targets -- -D warnings\n",
    "export PATH=\"$HOME/.cargo/bin:$PATH\"\ncd \"{dir}\" && {cmd} {args}\n",
];

const NAMES: &[&str] = &[
    "run", "load", "scan", "build", "sync", "clean", "watch", "exec", "probe", "pack",
];
const CLASSES: &[&str] = &["Agent", "Runner", "Store", "Task", "Shell", "KanNet"];
const FILES: &[&str] = &[
    "main.py",
    "lib.rs",
    "run.sh",
    "Cargo.toml",
    "pyproject.toml",
    "notes.md",
];
const PFX: &[&str] = &["TODO", "fn ", "def ", "export ", "use "];
const CMDS: &[&str] = &[
    "ls", "cat", "rg", "git", "cargo", "python3", "chmod", "head",
];
const GLOBS: &[&str] = &["*.py", "*.rs", "*.sh", "*.md", "src/*"];
const DIRS: &[&str] = &["src", "scripts", "data", ".", "ullis"];
pub const LANGS: [&str; 3] = ["python", "rust", "bash"];

fn pick<T: Clone>(rng: &mut impl Rng, xs: &[T]) -> T {
    xs[rng.random_range(0..xs.len())].clone()
}

fn fill_python(rng: &mut impl Rng) -> String {
    let t = pick(rng, PYTHON_TEMPLATES);
    t.replace("{fn}", pick(rng, NAMES))
        .replace("{cls}", pick(rng, CLASSES))
        .replace("{file}", pick(rng, FILES))
        .replace("{pfx}", pick(rng, PFX))
        .replace("{kw}", pick(rng, &["def ", "import", "class ", "Path"]))
}

fn fill_rust(rng: &mut impl Rng) -> String {
    let t = pick(rng, RUST_TEMPLATES);
    t.replace("{fn}", pick(rng, NAMES))
        .replace("{cls}", pick(rng, CLASSES))
        .replace("{file}", pick(rng, FILES))
        .replace("{pfx}", pick(rng, PFX))
}

fn fill_bash(rng: &mut impl Rng) -> String {
    let t = pick(rng, BASH_TEMPLATES);
    t.replace("{fn}", pick(rng, NAMES))
        .replace("{glob}", pick(rng, GLOBS))
        .replace("{cmd2}", pick(rng, CMDS))
        .replace("{cmd}", pick(rng, CMDS))
        .replace("{args2}", pick(rng, &["-n", "head", "wc -l", "sort"]))
        .replace(
            "{args}",
            pick(rng, &["-l", "--help", "-n 20", "status", "test"]),
        )
        .replace("{dir}", pick(rng, DIRS))
}

pub fn fill_lang(lang: &str, rng: &mut impl Rng) -> String {
    match lang {
        "rust" => fill_rust(rng),
        "bash" => fill_bash(rng),
        _ => fill_python(rng),
    }
}

pub fn seed_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("data/seed"),
        PathBuf::from("../data/seed"),
        PathBuf::from("ullis-core/../data/seed"),
    ];
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("data/seed")
}

pub fn load_seed_chunks() -> std::collections::HashMap<String, Vec<String>> {
    let mut out = std::collections::HashMap::new();
    for lang in LANGS {
        out.insert(lang.to_string(), Vec::new());
    }
    let dir = seed_dir();
    if !dir.exists() {
        return out;
    }
    for lang in LANGS {
        let path = dir.join(format!("{lang}.txt"));
        if !path.exists() {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&path) {
            let mut chunks: Vec<String> = text
                .split("\n\n---\n\n")
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect();
            if chunks.len() <= 1 {
                chunks = text
                    .split("\n\n")
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty())
                    .collect();
            }
            out.insert(lang.to_string(), chunks);
        }
    }
    out
}

pub fn snippet_text(
    lang: &str,
    rng: &mut impl Rng,
    seeds: &std::collections::HashMap<String, Vec<String>>,
) -> String {
    let body = if let Some(chunks) = seeds.get(lang) {
        if !chunks.is_empty() && rng.random::<f32>() < 0.45 {
            pick(rng, chunks)
        } else {
            fill_lang(lang, rng)
        }
    } else {
        fill_lang(lang, rng)
    };
    format!("{}\n", body.trim_end())
}

pub fn corpus_texts(n: usize, seed: u64) -> Vec<String> {
    let mut rng = crate::device::rng_from_seed(seed);
    let seeds = load_seed_chunks();
    let mut texts = Vec::new();
    for chunks in seeds.values() {
        texts.extend(chunks.clone());
    }
    for i in 0..n {
        texts.push(snippet_text(LANGS[i % LANGS.len()], &mut rng, &seeds));
    }
    texts
}

#[derive(Serialize)]
struct Line<'a> {
    system: &'a str,
    user: &'a str,
    thinking: &'a str,
    output: &'a str,
}

fn thinking_trace(lang: &str) -> String {
    format!(
        "1. Identify the language as {lang} from keywords and delimiters.\n2. Reconstruct the requested snippet from the user turn.\n3. Check brackets, names and return types before emitting the final block."
    )
}

fn user_for(lang: &str) -> &'static str {
    match lang {
        "rust" => "Write a small, complete Rust snippet.",
        "bash" => "Write a small, complete bash snippet.",
        _ => "Write a small, complete Python snippet.",
    }
}

pub fn write_jsonl(path: impl AsRef<Path>, n: usize, seed: u64) -> anyhow::Result<u64> {
    use std::io::Write;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rng = crate::device::rng_from_seed(seed);
    let seeds = load_seed_chunks();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut lines = 0u64;
    for lang in LANGS {
        if let Some(chunks) = seeds.get(lang) {
            for c in chunks {
                let sys = crate::data::ChatRecord::system_for_lang(lang);
                let think = thinking_trace(lang);
                let rec = Line {
                    system: sys,
                    user: user_for(lang),
                    thinking: &think,
                    output: c,
                };
                serde_json::to_writer(&mut f, &rec)?;
                f.write_all(b"\n")?;
                lines += 1;
            }
        }
    }
    for i in 0..n {
        let lang = LANGS[i % LANGS.len()];
        let text = snippet_text(lang, &mut rng, &seeds);
        let sys = crate::data::ChatRecord::system_for_lang(lang);
        let think = thinking_trace(lang);
        let rec = Line {
            system: sys,
            user: user_for(lang),
            thinking: &think,
            output: &text,
        };
        serde_json::to_writer(&mut f, &rec)?;
        f.write_all(b"\n")?;
        lines += 1;
    }
    f.flush()?;
    Ok(lines)
}

pub fn ensure_jsonl(path: impl AsRef<Path>, seed: u64) -> anyhow::Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    if path.exists() {
        return Ok(path);
    }
    let n = 800;
    eprintln!(
        "ullis: {} missing — writing {n} synthetic JSONL records",
        path.display()
    );
    write_jsonl(&path, n, seed)?;
    Ok(path)
}
