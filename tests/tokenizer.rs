use ullis::tokenizer::{StreamDecoder, train_bpe, train_bpe_from_reader};

fn fixture_texts() -> Vec<String> {
    vec![
        "def load(path):\n    return path\n".into(),
        "fn main() {\n    match x {\n        Ok(s) => s,\n    }\n}\n".into(),
        "#!/usr/bin/env bash\nset -euo pipefail\n".into(),
        "impl Agent {\n    fn new(name: &str) -> Self { Self { name: name.into() } }\n}\n".into(),
        "return match impl def class import".into(),
    ]
}

#[test]
fn roundtrip_code() {
    let texts = fixture_texts();
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
        assert!(*ids.iter().min().unwrap() < tok.vocab_size());
    }
}

#[test]
fn corpus_bpe_compresses_repeated_domain_text() {
    let mut texts = fixture_texts();
    texts.extend((0..64).map(|_| "ullis_engine ullis_engine\n".into()));
    let mut tok = train_bpe(&texts, 1024, 2).unwrap();
    let ids = tok.encode("ullis_engine", false, false);
    assert!(
        ids.len() <= 2,
        "domain term used {} tokens: {ids:?}",
        ids.len()
    );
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

#[test]
fn reader_training_does_not_require_a_text_vector() {
    let mut tok = train_bpe_from_reader(&b"micro micro micro\n"[..], 512, 3).unwrap();
    assert!(tok.vocab_size() < 512);
    let ids = tok.encode("micro", false, false);
    assert_eq!(tok.decode(&ids), "micro");
}
