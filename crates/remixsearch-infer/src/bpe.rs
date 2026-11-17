use remixsearch_format::{Error, Result};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct CompactBpe {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub merges: HashMap<(String, String), usize>,
    pub unk_id: u32,
    pub pad_id: u32,
}

impl CompactBpe {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.pieces(text).into_iter().map(|(_, id)| id).collect()
    }

    /// BPE pieces for `text`, each with its vocab id. An id equal to `unk_id` is an unknown piece.
    pub fn pieces(&self, text: &str) -> Vec<(String, u32)> {
        if self.vocab.is_empty() {
            return text.bytes().map(|b| (format!("{b}"), b as u32)).collect();
        }
        let prefixed;
        let text = if text.starts_with(' ') {
            text
        } else {
            prefixed = format!(" {text}");
            &prefixed
        };
        let mut out = Vec::new();
        for piece in split_pieces(text) {
            for token in bpe_merge(&piece, &self.merges) {
                let id = self.vocab.get(&token).copied().unwrap_or(self.unk_id);
                out.push((token, id));
            }
        }
        out
    }

    pub fn byte_fallback_for_tests() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            merges: HashMap::new(),
            unk_id: 0,
            pad_id: 0,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"SBPE".to_vec();
        out.extend_from_slice(&(self.id_to_token.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.pad_id.to_le_bytes());
        out.extend_from_slice(&self.unk_id.to_le_bytes());
        for token in &self.id_to_token {
            let b = token.as_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
        }
        let mut pairs: Vec<(&(String, String), &usize)> = self.merges.iter().collect();
        pairs.sort_by_key(|(_, rank)| *rank);
        out.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
        for ((a, b), _) in pairs {
            for s in [a, b] {
                let bytes = s.as_bytes();
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 16 || &data[0..4] != b"SBPE" {
            return Err(Error::Message("bad tokenizer magic".into()));
        }
        let mut i = 4;
        let read_u32 = |i: &mut usize| -> Result<u32> {
            if *i + 4 > data.len() {
                return Err(Error::Message("truncated tokenizer".into()));
            }
            let v = u32::from_le_bytes(data[*i..*i + 4].try_into().unwrap());
            *i += 4;
            Ok(v)
        };
        let read_str = |i: &mut usize| -> Result<String> {
            let n = read_u32(i)? as usize;
            if *i + n > data.len() {
                return Err(Error::Message("truncated token".into()));
            }
            let s = String::from_utf8_lossy(&data[*i..*i + n]).into_owned();
            *i += n;
            Ok(s)
        };
        let n = read_u32(&mut i)? as usize;
        let pad_id = read_u32(&mut i)?;
        let unk_id = read_u32(&mut i)?;
        let mut id_to_token = Vec::with_capacity(n);
        let mut vocab = HashMap::with_capacity(n);
        for idx in 0..n {
            let token = read_str(&mut i)?;
            vocab.insert(token.clone(), idx as u32);
            id_to_token.push(token);
        }
        let merges_n = read_u32(&mut i)? as usize;
        let mut merges = HashMap::with_capacity(merges_n);
        for rank in 0..merges_n {
            let a = read_str(&mut i)?;
            let b = read_str(&mut i)?;
            merges.insert((a, b), rank);
        }
        Ok(Self { vocab, id_to_token, merges, unk_id, pad_id })
    }
}

pub fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!'..=b'~').map(|b| b as u32).collect();
    bs.extend((0xA1..=0xAC).map(|b| b as u32));
    bs.extend((0xAE..=0xFF).map(|b| b as u32));
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = vec!['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        map[b as usize] = char::from_u32(c).unwrap_or('\0');
    }
    map
}

fn byte_map() -> &'static Vec<char> {
    static MAP: OnceLock<Vec<char>> = OnceLock::new();
    MAP.get_or_init(bytes_to_unicode)
}

pub fn split_pieces(text: &str) -> Vec<String> {
    let map = byte_map();
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = next_piece(&chars, i);
        let piece: String = chars[i..end].iter().collect();
        out.push(piece.bytes().map(|b| map[b as usize]).collect());
        i = end;
    }
    out
}

fn next_piece(chars: &[char], i: usize) -> usize {
    let rest: String = chars[i..].iter().collect();
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(contraction) {
            return i + contraction.chars().count();
        }
    }
    let mut j = i;
    if chars.get(j) == Some(&' ') {
        j += 1;
    }
    if let Some(kind) = chars.get(j).copied().and_then(word_kind) {
        j += 1;
        while chars.get(j).copied().and_then(word_kind) == Some(kind) {
            j += 1;
        }
        return j;
    }
    if chars[i].is_whitespace() {
        let mut j = i + 1;
        while chars.get(j).is_some_and(|c| c.is_whitespace()) {
            j += 1;
        }
        return j;
    }
    i + 1
}

fn word_kind(c: char) -> Option<u8> {
    if c.is_alphabetic() {
        Some(1)
    } else if c.is_numeric() {
        Some(2)
    } else if !c.is_whitespace() {
        Some(3)
    } else {
        None
    }
}

fn bpe_merge(token: &str, merges: &HashMap<(String, String), usize>) -> Vec<String> {
    let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
    if word.len() < 2 {
        return word;
    }
    loop {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..word.len() - 1 {
            if let Some(&rank) = merges.get(&(word[i].clone(), word[i + 1].clone())) {
                if best.map(|(r, _)| rank < r).unwrap_or(true) {
                    best = Some((rank, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let merged = format!("{}{}", word[i], word[i + 1]);
        word[i] = merged;
        word.remove(i + 1);
    }
    word
}
