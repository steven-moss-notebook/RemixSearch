use image::{Rgb, RgbImage};
use std::path::Path;

const GLYPHS: &[(char, u64)] = &[
    ('0', 0x3E22222A22223E),
    ('1', 0x0818080808083E),
    ('2', 0x3E02023E20203E),
    ('3', 0x3E02023E02023E),
    ('4', 0x2424243E040404),
    ('5', 0x3E20203E02023E),
    ('6', 0x3E20203E22223E),
    ('7', 0x3E020204040808),
    ('8', 0x3E22223E22223E),
    ('9', 0x3E22223E02023E),
    ('a', 0x003E02023E2222),
    ('b', 0x20203E22223E20),
    ('c', 0x003E2020203E00),
    ('d', 0x02023E22223E02),
    ('e', 0x003E22203E003E),
    ('f', 0x0C101E10101000),
    ('g', 0x003E22223E0202),
    ('i', 0x0800080808081C),
    ('l', 0x1818080808081C),
    ('m', 0x0000667E5A4242),
    ('n', 0x003E2222222200),
    ('o', 0x003E2222223E00),
    ('r', 0x003E2020202000),
    ('s', 0x003E203E023E00),
    ('t', 0x101E1010100C00),
    ('u', 0x00222222223E00),
    (' ', 0),
    ('-', 0x0000003E000000),
];

fn glyph(ch: char) -> u64 {
    let c = ch.to_ascii_lowercase();
    GLYPHS.iter().find(|(g, _)| *g == c).map(|(_, b)| *b).unwrap_or(0x224A0000)
}

pub fn draw_diagram(path: &Path, label: &str, seed: u32) -> std::io::Result<()> {
    let mut img = RgbImage::from_pixel(192, 112, Rgb([20, 24, 32]));
    let color = Rgb([
        40 + ((seed * 47) % 180) as u8,
        50 + ((seed * 29) % 160) as u8,
        80 + ((seed * 13) % 140) as u8,
    ]);
    let x0 = 12 + (seed % 20) as i32;
    let y0 = 16 + ((seed / 3) % 12) as i32;
    let w = 36 + (seed % 9) * 4;
    let h = 28 + (seed % 5) * 3;
    for y in y0..y0 + h as i32 {
        for x in x0..x0 + w as i32 {
            if x >= 0 && y >= 0 && x < 192 && y < 112 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
    // A second bar so nearby seeds stay visually distinct.
    let accent = Rgb([230, 230 - (seed % 40) as u8, 40]);
    let ay = 70 + (seed % 15) as u32;
    for x in 16..160 {
        if x % (3 + seed % 4) == 0 {
            img.put_pixel(x, ay, accent);
            if ay + 1 < 112 {
                img.put_pixel(x, ay + 1, accent);
            }
        }
    }
    draw_text(&mut img, 8, 8, label);
    img.save(path).map_err(std::io::Error::other)
}

fn draw_text(img: &mut RgbImage, mut x: u32, y: u32, text: &str) {
    for ch in text.chars().take(28) {
        let bits = glyph(ch);
        for row in 0..7 {
            for col in 0..6 {
                let bit = 41 - (row * 6 + col);
                if ((bits >> bit) & 1) == 1 && x + col < img.width() && y + row < img.height() {
                    img.put_pixel(x + col, y + row, Rgb([240, 240, 245]));
                }
            }
        }
        x += 7;
        if x + 6 >= img.width() {
            break;
        }
    }
}
