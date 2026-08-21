use ullis::tokenizer::{train_bpe, StreamDecoder};

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
        assert!(*ids.iter().min().unwrap() < tok.vocab_size);
    }
}

#[test]
fn code_atoms_compress() {
    let texts = fixture_texts();
    let mut tok = train_bpe(&texts, 1024, 2).unwrap();
    for atom in ["def ", "fn ", "return", "impl", "match"] {
        let ids = tok.encode(atom, false, false);
        assert_eq!(
            ids.len(),
            1,
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
