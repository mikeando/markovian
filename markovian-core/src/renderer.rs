use crate::symbol::{SymbolTable, SymbolTableEntry, SymbolTableEntryId};

pub trait SymbolIdRenderer {
    fn render(&self, symbol_id: SymbolTableEntryId) -> String;
}

pub struct SymbolIdRendererU8<'a> {
    pub table: &'a SymbolTable<u8>,
    pub start: &'a str,
    pub end: &'a str,
}

impl<'a> SymbolIdRendererU8<'a> {
    pub fn new(table: &'a SymbolTable<u8>) -> SymbolIdRendererU8<'a> {
        SymbolIdRendererU8 {
            table,
            start: "^",
            end: "$",
        }
    }
}

pub fn utf8_or_escaped(v: &[u8]) -> String {
    match std::str::from_utf8(v) {
        Ok(s) => s.to_string(),
        // TODO: Can print as a mix of valid utf-8 and escapes using
        //       the information stored in the error - see documentation
        //       on from_utf8.
        Err(_) => {
            let part: Vec<u8> = v
                .iter()
                .flat_map(|b| std::ascii::escape_default(*b).collect::<Vec<_>>())
                .collect();
            String::from_utf8(part).unwrap()
            //Or we could use Strng::from_utf_8_lossy - but it wouldn't include as much info in the failure.
        }
    }
}

impl<'a> SymbolIdRenderer for SymbolIdRendererU8<'a> {
    fn render(&self, symbol_id: SymbolTableEntryId) -> String {
        let symbol = self.table.get_by_id(symbol_id).unwrap();
        match symbol {
            SymbolTableEntry::Start => self.start.to_string(),
            SymbolTableEntry::End => self.end.to_string(),
            SymbolTableEntry::Single(b) => {
                let part: Vec<u8> = std::ascii::escape_default(*b).collect();
                String::from_utf8(part).unwrap()
            }
            SymbolTableEntry::Compound(bs) => utf8_or_escaped(&bs),
            SymbolTableEntry::Dead(_) => "✞".to_string(),
        }
    }
}

pub struct SymbolIdRendererChar<'a> {
    pub table: &'a SymbolTable<char>,
    pub start: &'a str,
    pub end: &'a str,
}

impl<'a> SymbolIdRenderer for SymbolIdRendererChar<'a> {
    fn render(&self, symbol_id: SymbolTableEntryId) -> String {
        let symbol = self.table.get_by_id(symbol_id).unwrap();
        match symbol {
            SymbolTableEntry::Start => self.start.to_string(),
            SymbolTableEntry::End => self.end.to_string(),
            SymbolTableEntry::Single(c) => format!("{}", c),
            SymbolTableEntry::Compound(cs) => cs.iter().collect::<String>(),
            SymbolTableEntry::Dead(_) => "✞".to_string(),
        }
    }
}

pub trait Renderer {
    //TODO: Should this instead use a writer?
    //TODO: Should it report errors?
    fn render<'a>(&self, v: &'a [SymbolTableEntryId]) -> String;
}

pub struct RendererId {}

impl Renderer for RendererId {
    fn render<'a>(&self, v: &'a [SymbolTableEntryId]) -> String {
        format!("{:?}", v.iter().map(|id| id.0).collect::<Vec<_>>())
    }
}

pub struct RenderU8<'a> {
    pub table: &'a SymbolTable<u8>,
    pub start: &'a [u8],
    pub end: &'a [u8],
}

impl<'b> Renderer for RenderU8<'b> {
    fn render<'a>(&self, ids: &'a [SymbolTableEntryId]) -> String {
        //TODO: Get rid of SymbolRender and replace with symbol_id_render?
        let mut result: Vec<u8> = vec![];
        for id in ids {
            let s = self.table.get_by_id(*id).unwrap();
            match s {
                SymbolTableEntry::Start => {
                    result.extend_from_slice(self.start);
                }
                SymbolTableEntry::End => {
                    result.extend_from_slice(self.end);
                }
                SymbolTableEntry::Single(e) => {
                    result.push(*e);
                }
                SymbolTableEntry::Compound(e) => {
                    result.extend_from_slice(&e);
                }
                SymbolTableEntry::Dead(_) => panic!("DEAD"),
            }
        }
        utf8_or_escaped(&result)
    }
}

pub struct RenderChar<'a> {
    pub table: &'a SymbolTable<char>,
    pub start: &'a str,
    pub end: &'a str,
}

impl<'b> Renderer for RenderChar<'b> {
    fn render<'a>(&self, ids: &'a [SymbolTableEntryId]) -> String {
        //TODO: Get rid of SymbolRender and replace with symbol_id_render?
        let mut result: String = String::new();
        for id in ids {
            let s = self.table.get_by_id(*id).unwrap();
            match s {
                SymbolTableEntry::Start => {
                    result = format!("{}{}", result, self.start);
                }
                SymbolTableEntry::End => {
                    result = format!("{}{}", result, self.end);
                }
                SymbolTableEntry::Single(e) => {
                    result = format!("{}{}", result, e);
                }
                SymbolTableEntry::Compound(e) => {
                    let ss: String = e.iter().collect();
                    result = format!("{}{}", result, ss);
                }
                SymbolTableEntry::Dead(_) => panic!("DEAD"),
            }
        }
        result
    }
}

pub fn renderer_for_u8_with_separator<'a>(
    table: &'a SymbolTable<u8>,
    separator: &'a str,
) -> RendererWithSeparator<'a, SymbolIdRendererU8<'a>> {
    RendererWithSeparator {
        symbol_renderer: SymbolIdRendererU8 {
            table,
            start: "^",
            end: "$",
        },
        separator,
    }
}

pub fn renderer_for_char_with_separator<'a>(
    table: &'a SymbolTable<char>,
    separator: &'a str,
) -> RendererWithSeparator<'a, SymbolIdRendererChar<'a>> {
    RendererWithSeparator {
        symbol_renderer: SymbolIdRendererChar {
            table,
            start: "^",
            end: "$",
        },
        separator,
    }
}

pub struct RendererWithSeparator<'a, T: SymbolIdRenderer> {
    pub symbol_renderer: T,
    pub separator: &'a str,
}

impl<'b, T> Renderer for RendererWithSeparator<'b, T>
where
    T: SymbolIdRenderer,
{
    fn render<'a>(&self, v: &'a [SymbolTableEntryId]) -> String {
        use std::fmt::Write;
        let mut result: String = String::new();
        let mut is_start = true;
        for s in v {
            let symbol_str = self.symbol_renderer.render(*s);
            if is_start {
                is_start = false;
                write!(&mut result, "{}", symbol_str).unwrap();
            } else {
                write!(&mut result, "{}{}", self.separator, symbol_str).unwrap();
            }
        }
        result
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;

    struct ContextDefault {
        start: SymbolTableEntryId,
        end: SymbolTableEntryId,
        a: SymbolTableEntryId,
        xyz: SymbolTableEntryId,
    }

    fn default_symbol_table() -> (SymbolTable<u8>, ContextDefault) {
        let mut result = SymbolTable::new();

        let start = result.add(SymbolTableEntry::Start);
        let end = result.add(SymbolTableEntry::End);
        let a = result.add(SymbolTableEntry::Single(b'a'));
        let xyz = result.add(SymbolTableEntry::Compound(vec![b'x', b'y', b'z']));

        (result, ContextDefault { start, end, a, xyz })
    }

    #[test]
    pub fn check_symbol_table_render() {
        let (s, c) = default_symbol_table();

        let sr = RenderU8 {
            table: &s,
            start: b"^",
            end: b"$",
        };

        let u: String = sr.render(&vec![]);
        assert_eq!(u, "");

        //TODO: Should this be an error?
        // let u: String = sr.render(&vec![SymbolTableEntryId(123)]);
        // assert_eq!(u, "?");

        let u: String = sr.render(&vec![c.start]);
        assert_eq!(u, "^");

        let u: String = sr.render(&vec![c.end]);
        assert_eq!(u, "$");

        let u: String = sr.render(&vec![c.a]);
        assert_eq!(u, "a");

        let u: String = sr.render(&vec![c.xyz]);
        assert_eq!(u, "xyz");

        let u: String = sr.render(&vec![c.start, c.a, c.xyz, c.end]);
        assert_eq!(u, "^axyz$");
    }
}
