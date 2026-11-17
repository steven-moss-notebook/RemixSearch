use crate::{Error, NodeMeta, QuantVector, Result};

pub fn brotli_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut params = brotli::enc::BrotliEncoderParams::default();
    params.quality = 5;
    brotli::BrotliCompress(&mut &data[..], &mut out, &params)
        .map_err(|e| Error::Brotli(e.to_string()))?;
    Ok(out)
}

pub fn brotli_decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    brotli::BrotliDecompress(&mut &data[..], &mut out).map_err(|e| Error::Brotli(e.to_string()))?;
    Ok(out)
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn read_str(data: &[u8], i: &mut usize) -> Result<String> {
    if *i + 4 > data.len() {
        return Err(Error::Message("truncated string".into()));
    }
    let n = u32::from_le_bytes(data[*i..*i + 4].try_into().unwrap()) as usize;
    *i += 4;
    if *i + n > data.len() {
        return Err(Error::Message("truncated string bytes".into()));
    }
    let s = String::from_utf8_lossy(&data[*i..*i + n]).into_owned();
    *i += n;
    Ok(s)
}

pub fn encode_meta(nodes: &[NodeMeta]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    for n in nodes {
        out.push(n.kind.as_u8());
        push_str(&mut out, &n.url);
        push_str(&mut out, &n.title);
        push_str(&mut out, &n.image_url);
        push_str(&mut out, &n.alt);
        push_str(&mut out, &n.caption);
        push_str(&mut out, &n.snippet);
    }
    out
}

pub fn decode_meta(data: &[u8]) -> Result<Vec<NodeMeta>> {
    if data.len() < 4 {
        return Err(Error::Message("empty meta".into()));
    }
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut i = 4;
    let mut nodes = Vec::with_capacity(n);
    for _ in 0..n {
        if i >= data.len() {
            return Err(Error::Message("truncated meta".into()));
        }
        let kind = crate::NodeKind::from_u8(data[i]);
        i += 1;
        nodes.push(NodeMeta {
            kind,
            url: read_str(data, &mut i)?,
            title: read_str(data, &mut i)?,
            image_url: read_str(data, &mut i)?,
            alt: read_str(data, &mut i)?,
            caption: read_str(data, &mut i)?,
            snippet: read_str(data, &mut i)?,
        });
    }
    Ok(nodes)
}

pub fn encode_vectors(vecs: &[QuantVector]) -> Vec<u8> {
    let mut out = Vec::new();
    let dims = vecs.first().map(|v| v.values.len()).unwrap_or(0);
    out.extend_from_slice(&(vecs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(dims as u32).to_le_bytes());
    for v in vecs {
        out.extend_from_slice(&v.scale.to_le_bytes());
        for x in &v.values {
            out.push(*x as u8);
        }
    }
    out
}

pub fn decode_vectors(data: &[u8]) -> Result<Vec<QuantVector>> {
    if data.len() < 8 {
        return Err(Error::Message("empty vectors".into()));
    }
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let dims = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let mut i = 8;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if i + 4 + dims > data.len() {
            return Err(Error::Message("truncated vectors".into()));
        }
        let scale = f32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        i += 4;
        let values = data[i..i + dims].iter().map(|b| *b as i8).collect();
        i += dims;
        out.push(QuantVector { scale, values });
    }
    Ok(out)
}

pub fn encode_graph(neighbors: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
    for n in neighbors {
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        for id in n {
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
    out
}

pub fn decode_graph(data: &[u8]) -> Result<Vec<Vec<u32>>> {
    if data.len() < 4 {
        return Err(Error::Message("empty graph".into()));
    }
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut i = 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if i + 2 > data.len() {
            return Err(Error::Message("truncated graph".into()));
        }
        let c = u16::from_le_bytes(data[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let mut neigh = Vec::with_capacity(c);
        for _ in 0..c {
            if i + 4 > data.len() {
                return Err(Error::Message("truncated neighbor".into()));
            }
            neigh.push(u32::from_le_bytes(data[i..i + 4].try_into().unwrap()));
            i += 4;
        }
        out.push(neigh);
    }
    Ok(out)
}

/// One brotli stream holding every index member, with a directory in front of the bytes.
pub struct Package {
    members: Vec<(String, Vec<u8>)>,
}

impl Package {
    pub fn new() -> Self {
        Self { members: Vec::new() }
    }

    pub fn insert(&mut self, name: impl Into<String>, bytes: Vec<u8>) {
        let name = name.into();
        if let Some(existing) = self.members.iter_mut().find(|(n, _)| n == &name) {
            existing.1 = bytes;
        } else {
            self.members.push((name, bytes));
        }
    }

    pub fn get(&self, name: &str) -> Result<&[u8]> {
        self.members
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| Error::Message(format!("package is missing {name}")))
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SIDX");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(self.members.len() as u32).to_le_bytes());
        for (name, bytes) in &self.members {
            let name = name.as_bytes();
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        out
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 12 || &data[0..4] != b"SIDX" {
            return Err(Error::Message("not a search index package".into()));
        }
        let count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let mut i = 12;
        let mut members = Vec::with_capacity(count);
        for _ in 0..count {
            if i + 2 > data.len() {
                return Err(Error::Message("truncated package name".into()));
            }
            let name_len = u16::from_le_bytes(data[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            if i + name_len + 4 > data.len() {
                return Err(Error::Message("truncated package member".into()));
            }
            let name = String::from_utf8_lossy(&data[i..i + name_len]).into_owned();
            i += name_len;
            let len = u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
            i += 4;
            if i + len > data.len() {
                return Err(Error::Message("truncated package bytes".into()));
            }
            members.push((name, data[i..i + len].to_vec()));
            i += len;
        }
        Ok(Self { members })
    }

    pub fn compress(&self) -> Result<Vec<u8>> {
        brotli_compress(&self.encode())
    }

    pub fn open(compressed: &[u8]) -> Result<Self> {
        Self::decode(&brotli_decompress(compressed)?)
    }
}
