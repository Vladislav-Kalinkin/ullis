//! Conversation rendering shared by train, generate, and chat.
//!
//! Train and decode must emit the same markup. Next-token loss is restricted to
//! assistant tokens (including `<assistant>` … `</assistant>`) and EOS so
//! repeated system prompts cannot dominate the gradient.

use crate::tokenizer::BpeTokenizer;
use anyhow::{Result, bail};

pub const SYSTEM_OPEN: &str = "<system>";
pub const SYSTEM_CLOSE: &str = "</system>";
pub const USER_OPEN: &str = "<user>";
pub const USER_CLOSE: &str = "</user>";
pub const ASSISTANT_OPEN: &str = "<assistant>";
pub const ASSISTANT_CLOSE: &str = "</assistant>";
pub const THINKING_OPEN: &str = "<thinking>";
pub const THINKING_CLOSE: &str = "</thinking>";
pub const TOOL_OPEN: &str = "<tool>";
pub const TOOL_CLOSE: &str = "</tool>";

/// Inclusive-exclusive byte range in a rendered conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
    pub thinking: Option<&'a str>,
}

/// Markup used both to train and to seed generate/chat.
pub fn render_messages(messages: &[ChatMessage<'_>]) -> (String, Vec<ByteSpan>) {
    let mut out = String::new();
    let mut supervised = Vec::new();
    for message in messages {
        match message.role {
            "system" => {
                out.push_str(SYSTEM_OPEN);
                out.push_str(message.content);
                out.push_str(SYSTEM_CLOSE);
            }
            "user" => {
                out.push_str(USER_OPEN);
                out.push_str(message.content);
                out.push_str(USER_CLOSE);
            }
            "assistant" => {
                let start = out.len();
                out.push_str(ASSISTANT_OPEN);
                out.push_str(THINKING_OPEN);
                out.push_str(message.thinking.unwrap_or(""));
                out.push_str(THINKING_CLOSE);
                out.push_str(message.content);
                out.push_str(ASSISTANT_CLOSE);
                supervised.push(ByteSpan {
                    start,
                    end: out.len(),
                });
            }
            "tool" => {
                out.push_str(TOOL_OPEN);
                out.push_str(message.content);
                out.push_str(TOOL_CLOSE);
            }
            _ => {}
        }
    }
    (out, supervised)
}

/// True when the caller already supplied train markup.
pub fn has_conversation_markup(text: &str) -> bool {
    text.contains(SYSTEM_OPEN) || text.contains(USER_OPEN) || text.contains(ASSISTANT_OPEN)
}

/// Open assistant turn used by `generate` / `chat` for a raw user string.
pub fn generation_prefix(user_text: &str) -> String {
    if has_conversation_markup(user_text) {
        return user_text.to_string();
    }
    let (text, _) = render_messages(&[ChatMessage {
        role: "user",
        content: user_text,
        thinking: None,
    }]);
    let mut prefix = text;
    prefix.push_str(ASSISTANT_OPEN);
    prefix.push_str(THINKING_OPEN);
    prefix
}

/// Truncate a decoded completion at the first trained end-of-turn tag.
pub fn truncate_at_assistant_end(text: &str) -> Option<&str> {
    text.find(ASSISTANT_CLOSE).map(|index| &text[..index])
}

/// `labels[i] = ids[i]` when token `i` is a supervised target, otherwise `pad`.
///
/// EOS is always supervised so the model can end a record. BOS and pad are not.
pub fn supervised_labels(
    tokenizer: &BpeTokenizer,
    ids: &[u32],
    text: &str,
    spans: &[ByteSpan],
) -> Vec<u32> {
    let mut labels = vec![tokenizer.pad_id; ids.len()];
    let mut byte_pos = 0_usize;
    for (index, &id) in ids.iter().enumerate() {
        if id == tokenizer.bos_id || id == tokenizer.pad_id || id == tokenizer.unk_id {
            continue;
        }
        if id == tokenizer.eos_id {
            labels[index] = id;
            continue;
        }
        let bytes = tokenizer.token_bytes(id);
        let start = byte_pos;
        let end = byte_pos.saturating_add(bytes.len());
        if spans
            .iter()
            .any(|span| start < span.end && end > span.start)
        {
            labels[index] = id;
        }
        byte_pos = end;
    }
    debug_assert_eq!(
        byte_pos,
        text.len(),
        "BPE tokens must cover the rendered conversation"
    );
    labels
}

/// Dense token stream: documents are concatenated, never pad-filled.
///
/// Pad in the *input* would repeat one embedding for the rest of the window and
/// send 1-bit ROSA SAM into a long identical suffix (seconds per step). Ignore
/// index lives only in `labels`; every forwarded id is a real corpus token.
/// Windows with fewer than [`min_supervised_targets`] next-token labels are
/// skipped: a 2048-wide step on one EOS would otherwise dominate mean-or-sum
/// SGD relative to a dense assistant span.
pub fn pack_document_windows(
    documents: &[(Vec<u32>, Vec<u32>)],
    context_len: usize,
    pad_id: u32,
    needed: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if context_len < 2 {
        bail!("packed windows require context_len >= 2");
    }
    if needed == 0 {
        bail!("encoded corpus is empty");
    }
    let mut tokens = Vec::new();
    let mut labels = Vec::new();
    for (ids, targets) in documents {
        if ids.len() != targets.len() || ids.is_empty() {
            continue;
        }
        tokens.extend_from_slice(ids);
        labels.extend_from_slice(targets);
    }
    if tokens.is_empty() || tokens.iter().all(|&id| id == pad_id) {
        bail!("encoded corpus is empty");
    }
    if labels.iter().all(|&id| id == pad_id) {
        bail!("encoded corpus has no supervised assistant/EOS tokens");
    }
    let n = tokens.len();
    let min_supervised = min_supervised_targets(context_len);
    let mut out_tokens = Vec::with_capacity(needed);
    let mut out_labels = Vec::with_capacity(needed);
    let mut pos = 0_usize;
    let mut scanned = 0_usize;
    let scan_limit = n
        .saturating_mul(2)
        .saturating_add(needed)
        .saturating_add(context_len);
    while out_tokens.len() < needed {
        if scanned > scan_limit {
            bail!("could not pack {needed} windows with supervised next-token targets");
        }
        scanned += 1;
        let mut supervised = 0_usize;
        for t in 0..context_len.saturating_sub(1) {
            if labels[(pos + t + 1) % n] != pad_id {
                supervised += 1;
                if supervised >= min_supervised {
                    break;
                }
            }
        }
        if supervised < min_supervised {
            pos = (pos + 1) % n;
            continue;
        }
        for i in 0..context_len {
            if out_tokens.len() >= needed {
                break;
            }
            let index = (pos + i) % n;
            out_tokens.push(tokens[index]);
            out_labels.push(labels[index]);
        }
        pos = (pos + context_len) % n;
    }
    Ok((out_tokens, out_labels))
}

/// Minimum next-token labels in a packed window: one eighth of `T-1`, at least 1.
///
/// Assistant-only CE leaves long system/user stretches with a handful of
/// supervised ids. Token-sum SGD on those windows is cheap; the skip is so a
/// 2048-token Metal step is not spent on a single EOS.
pub fn min_supervised_targets(context_len: usize) -> usize {
    context_len.saturating_sub(1).saturating_div(8).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::train_bpe;

    #[test]
    fn render_marks_only_assistant_bytes() {
        let (text, spans) = render_messages(&[
            ChatMessage {
                role: "system",
                content: "Be brief.",
                thinking: None,
            },
            ChatMessage {
                role: "user",
                content: "2+2?",
                thinking: None,
            },
            ChatMessage {
                role: "assistant",
                content: "4",
                thinking: Some("add"),
            },
        ]);
        assert!(text.starts_with("<system>Be brief.</system><user>2+2?</user><assistant>"));
        assert_eq!(spans.len(), 1);
        assert_eq!(
            &text[spans[0].start..spans[0].end],
            "<assistant><thinking>add</thinking>4</assistant>"
        );
        assert!(!text[spans[0].start..spans[0].end].contains("<user>"));
    }

    #[test]
    fn generation_prefix_opens_the_trained_assistant_span() {
        assert_eq!(
            generation_prefix("Write Hello"),
            "<user>Write Hello</user><assistant><thinking>"
        );
        assert_eq!(
            generation_prefix("<user>already</user><assistant><thinking>"),
            "<user>already</user><assistant><thinking>"
        );
    }

    #[test]
    fn truncate_strips_the_end_tag_and_whatever_follows() {
        let text = "hello</assistant><system>next";
        assert_eq!(truncate_at_assistant_end(text), Some("hello"));
        assert_eq!(truncate_at_assistant_end("hello"), None);
    }

    #[test]
    fn labels_supervise_assistant_and_eos_only() {
        let mut tokenizer = train_bpe(
            &["<system>Be brief.</system><user>2+2?</user><assistant><thinking>add</thinking>4</assistant>".into()],
            512,
            1,
        )
        .unwrap();
        let (text, spans) = render_messages(&[
            ChatMessage {
                role: "system",
                content: "Be brief.",
                thinking: None,
            },
            ChatMessage {
                role: "user",
                content: "2+2?",
                thinking: None,
            },
            ChatMessage {
                role: "assistant",
                content: "4",
                thinking: Some("add"),
            },
        ]);
        let ids = tokenizer.encode(&text, true, true);
        let labels = supervised_labels(&tokenizer, &ids, &text, &spans);
        assert_eq!(ids[0], tokenizer.bos_id);
        assert_eq!(labels[0], tokenizer.pad_id);
        assert_eq!(*ids.last().unwrap(), tokenizer.eos_id);
        assert_eq!(*labels.last().unwrap(), tokenizer.eos_id);
        let supervised: Vec<u32> = labels
            .iter()
            .copied()
            .filter(|&id| id != tokenizer.pad_id)
            .collect();
        assert!(supervised.contains(&tokenizer.eos_id));
        let decoded_supervised = tokenizer.decode(&supervised);
        assert!(decoded_supervised.contains("add"));
        assert!(decoded_supervised.contains('4'));
        assert!(!decoded_supervised.contains("Be brief"));
        assert!(!decoded_supervised.contains("2+2"));
    }

    #[test]
    fn pack_windows_concatenate_real_tokens_without_pad() {
        let pad = 0_u32;
        let docs = vec![
            (vec![1_u32, 2, 3], vec![0_u32, 2, 3]),
            (vec![4_u32, 5], vec![4_u32, 5]),
        ];
        let (tokens, labels) = pack_document_windows(&docs, 4, pad, 8).unwrap();
        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 1, 2, 3]);
        assert_eq!(labels, vec![0, 2, 3, 4, 5, 0, 2, 3]);
        assert!(!tokens.contains(&pad));
    }

    #[test]
    fn pack_windows_skip_windows_with_no_supervised_target() {
        let pad = 0_u32;
        let docs = vec![(vec![1_u32, 2, 3, 4, 5, 6], vec![0_u32, 0, 0, 0, 5, 6])];
        let (tokens, labels) = pack_document_windows(&docs, 3, pad, 6).unwrap();
        assert_eq!(tokens.len(), 6);
        assert!(!tokens.contains(&pad));
        for window in tokens.chunks(3).zip(labels.chunks(3)) {
            let (_ids, window_labels) = window;
            assert!(
                window_labels[1..].iter().any(|&id| id != pad),
                "every packed window must have a supervised next-token target, got {window_labels:?}"
            );
        }
    }

    #[test]
    fn pack_windows_skip_sparse_assistant_tails() {
        let pad = 0_u32;
        // T=32 requires min_supervised = 31/8 = 3. Two supervised labels in a
        // long unsupervised prefix must not become a packed window.
        let ids = vec![1_u32; 64];
        let mut targets = vec![pad; 64];
        targets[60] = 1;
        targets[61] = 1;
        let err = pack_document_windows(&[(ids.clone(), targets.clone())], 32, pad, 32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("supervised next-token"));
        targets[50] = 1;
        let (tokens, labels) = pack_document_windows(&[(ids, targets)], 32, pad, 32).unwrap();
        assert_eq!(tokens.len(), 32);
        assert!(labels[1..].iter().filter(|&&id| id != pad).count() >= min_supervised_targets(32));
    }
}
