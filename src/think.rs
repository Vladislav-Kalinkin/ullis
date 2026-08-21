//! Multi-tier thinking budgets and ephemeral reasoning garbage collection.
//!
//! Persistent dialogue never stores `thinking` tokens. The scratch ring is
//! a `SovereignFlashBuffer` that is zeroed the instant a turn finishes.

use std::str::FromStr;

use clap::ValueEnum;

use crate::data::SovereignFlashBuffer;
use crate::kan::KanEvalMode;

/// Inference computational budget (`--thinking`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ThinkingMode {
    /// Bypass reasoning: mask routed (thinking) projections, coarse G=4 base.
    Low,
    /// Dim/italic think stream, then bold-green output. Default v0.5 path.
    #[default]
    Medium,
    /// Longer visual reasoning, still a single MoE-KAN pass per token.
    High,
    /// Deep recurrent resonance over the G=12 MoE-KAN stack, streamed live.
    Xhigh,
}

impl ThinkingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }

    /// Hidden reasoning token budget, always `< seq_len` so the ring cannot grow.
    pub fn think_budget(self, seq_len: usize) -> usize {
        let cap = seq_len.saturating_sub(8);
        match self {
            Self::Low => 0,
            Self::Medium => (seq_len / 4).max(8).min(cap),
            Self::High => (seq_len / 2).max(16).min(cap),
            Self::Xhigh => (seq_len * 3 / 4).max(24).min(cap),
        }
    }

    pub fn kan_mode(self) -> KanEvalMode {
        match self {
            Self::Low => KanEvalMode::Coarse,
            Self::Medium | Self::High => KanEvalMode::Full,
            Self::Xhigh => KanEvalMode::Resonant {
                loops: XHIGH_RESONANCE_LOOPS,
            },
        }
    }
}

impl FromStr for ThinkingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" | "l" | "off" | "0" => Ok(Self::Low),
            "medium" | "med" | "m" | "1" => Ok(Self::Medium),
            "high" | "h" | "2" => Ok(Self::High),
            "xhigh" | "xh" | "max" | "3" => Ok(Self::Xhigh),
            other => Err(format!(
                "unknown thinking mode `{other}` (low|medium|high|xhigh)"
            )),
        }
    }
}

/// Extra residual KAN passes per block at `xhigh` (mixer runs once).
pub const XHIGH_RESONANCE_LOOPS: u8 = 3;

/// Hard cap on persisted dialogue characters — independent of thinking depth.
pub const DIALOGUE_CHAR_CAP: usize = 3_072;
pub const DIALOGUE_TURN_CAP: usize = 6;

const DEFAULT_SYSTEM: &str = "You are a compact ternary KAN code engine. Infer the language from the prompt tokens and emit well-formed source.";

/// Persistent cache: `system` + `(user, output)` turns. No thinking tokens.
#[derive(Debug)]
pub struct DialogueCache {
    system: String,
    turns: Vec<(String, String)>,
}

impl DialogueCache {
    pub fn new(system: impl Into<String>) -> Self {
        let mut system = system.into();
        if system.trim().is_empty() {
            system = DEFAULT_SYSTEM.to_string();
        }
        Self {
            system,
            turns: Vec::with_capacity(DIALOGUE_TURN_CAP),
        }
    }

    pub fn system(&self) -> &str {
        &self.system
    }

    pub fn set_system(&mut self, system: impl Into<String>) {
        self.system = system.into();
        if self.system.trim().is_empty() {
            self.system = DEFAULT_SYSTEM.to_string();
        }
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Store a finished turn. Thinking is never accepted here.
    pub fn persist_turn(&mut self, user: String, output: String) {
        self.turns.push((user, output));
        while self.turns.len() > DIALOGUE_TURN_CAP {
            self.turns.remove(0);
        }
        self.trim_chars();
    }

    fn trim_chars(&mut self) {
        loop {
            let n: usize = self.system.len()
                + self
                    .turns
                    .iter()
                    .map(|(u, o)| u.len() + o.len())
                    .sum::<usize>();
            if n <= DIALOGUE_CHAR_CAP || self.turns.len() <= 1 {
                break;
            }
            self.turns.remove(0);
        }
    }

    /// Packed prompt for the next user turn (no thinking, no live output).
    pub fn pack_user(&self, user: &str) -> (String, String) {
        let mut user_block = String::new();
        for (u, o) in &self.turns {
            user_block.push_str(u.trim());
            user_block.push('\n');
            user_block.push_str(o.trim());
            user_block.push('\n');
        }
        user_block.push_str(user.trim());
        (self.system.clone(), user_block)
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn turns(&self) -> &[(String, String)] {
        &self.turns
    }

    pub fn restore(&mut self, system: String, turns: Vec<(String, String)>) {
        self.set_system(system);
        self.turns = turns;
        while self.turns.len() > DIALOGUE_TURN_CAP {
            self.turns.remove(0);
        }
        self.trim_chars();
    }
}

/// Ring of hidden reasoning tokens. Wiped after every turn.
#[derive(Debug)]
pub struct ReasoningScratch {
    tokens: SovereignFlashBuffer,
    text: String,
}

impl Default for ReasoningScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningScratch {
    pub fn new() -> Self {
        Self::with_cap(crate::data::MAX_TOKEN_BUF)
    }

    pub fn with_cap(cap: usize) -> Self {
        Self {
            tokens: SovereignFlashBuffer::new(cap.max(64))
                .unwrap_or_else(|_| SovereignFlashBuffer::new(64).expect("tiny flash")),
            text: String::new(),
        }
    }

    pub fn push_token(&mut self, id: u32) {
        self.tokens.push(id, 1);
    }

    pub fn push_text(&mut self, s: &str) {
        self.text.push_str(s);
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.text.is_empty()
    }

    /// Ephemeral GC used by the visual streamer after the output lane ends.
    pub fn clear(&mut self) {
        self.wipe();
    }

    /// Zero the flash plane and drop the text scratch.
    pub fn wipe(&mut self) {
        self.tokens.clear();
        if !self.text.is_empty() {
            let n = self.text.len();
            self.text.clear();
            self.text.reserve(n);
            for _ in 0..n {
                self.text.push('\0');
            }
            self.text.clear();
        }
        self.text.shrink_to_fit();
    }
}

impl Drop for ReasoningScratch {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Cheap structural check used by `xhigh` before the visible stream starts.
pub fn grammar_unbalanced(s: &str) -> bool {
    let mut paren: i32 = 0;
    let mut brace: i32 = 0;
    let mut bracket: i32 = 0;
    let mut quote: Option<char> = None;
    let mut escape = false;
    for c in s.chars() {
        if let Some(q) = quote {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => quote = Some(c),
            '(' => paren += 1,
            ')' => paren -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            _ => {}
        }
        if paren < 0 || brace < 0 || bracket < 0 {
            return true;
        }
    }
    quote.is_some() || paren != 0 || brace != 0 || bracket != 0
}

/// True when decoded text has closed the thinking span.
pub fn thinking_closed(decoded: &str) -> bool {
    decoded.contains(crate::data::TAG_THINK_END) || decoded.contains(crate::data::TAG_OUTPUT)
}

/// Strip structural tags from a visible completion.
pub fn strip_tags(s: &str) -> String {
    let mut out = s.to_string();
    for tag in [
        crate::data::TAG_SYSTEM,
        crate::data::TAG_USER,
        crate::data::TAG_THINKING,
        crate::data::TAG_THINK_END,
        crate::data::TAG_OUTPUT,
    ] {
        out = out.replace(tag, "");
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wipe_releases_capacity() {
        let mut s = ReasoningScratch::new();
        for i in 0..64u32 {
            s.push_token(i + 1);
        }
        s.push_text("fn main() { let x = 1; }");
        assert_eq!(s.len(), 64);
        assert!(!s.is_empty());
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn dialogue_never_keeps_thinking() {
        let mut c = DialogueCache::new("sys");
        c.persist_turn("user".into(), "fn f() {}".into());
        let (sys, user) = c.pack_user("next");
        assert_eq!(sys, "sys");
        assert!(!user.contains(crate::data::TAG_THINKING));
        assert!(user.contains("fn f() {}"));
    }

    #[test]
    fn grammar_detects_unbalanced() {
        assert!(grammar_unbalanced("fn f( { }"));
        assert!(!grammar_unbalanced("fn f() { let s = \"{\"; }"));
    }

    #[test]
    fn budgets_fit_seq_len() {
        for seq in [32usize, 96, 256] {
            for m in [
                ThinkingMode::Low,
                ThinkingMode::Medium,
                ThinkingMode::High,
                ThinkingMode::Xhigh,
            ] {
                assert!(m.think_budget(seq) < seq);
            }
            assert_eq!(ThinkingMode::Low.think_budget(seq), 0);
        }
    }
}
