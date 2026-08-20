use ullis::seed::corpus_texts;
use ullis::tokenizer::{train_bpe, StreamDecoder};

#[test]
fn roundtrip_code() {
    let texts = corpus_texts(60, 1);
    let mut tok = train_bpe(&texts, 512, 1).unwrap();
    let samples = [
        "def load(path):\n    return path\n",
        "fn main() {\n    match x {\n        Ok(s) => s,\n    }\n}\n",
        "#!/usr/bin/env bash\nset -euo pipefail\n",
        "impl Agent {\n    fn new(name: &str) -> Self { Self { name: name.into() } }\n}\n",
    ];
    for s in samples {
        let ids = tok.encode(s, false, false);
        assert_eq!(tok.decode(&ids), s);
        assert!(*ids.iter().min().unwrap() < tok.vocab_size);
    }
}

#[test]
fn code_atoms_compress() {
    let texts = corpus_texts(80, 2);
    let mut tok = train_bpe(&texts, 1024, 2).unwrap();
    for atom in ["def ", "fn ", "return", "impl", "match"] {
        let ids = tok.encode(atom, false, false);
        assert!(
            ids.len() <= 3,
            "{atom:?} used {} tokens: {ids:?}",
            ids.len()
        );
    }
}

#[test]
fn stream_decoder_utf8() {
    let mut tok = train_bpe(&["café π".into()], 320, 0).unwrap();
    let text = "café";
    let ids = tok.encode(text, false, false);
    let mut dec = StreamDecoder::new(&tok);
    let mut out = String::new();
    for i in ids {
        out.push_str(&dec.push(i));
    }
    out.push_str(&dec.flush());
    assert_eq!(out, text);
}
