#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum Symbol {
    Start,
    End,
    Char(u8),
    Compound(Vec<u8>),
}

pub fn raw_symbolify_word(s: &str) -> Vec<Symbol> {
    s.as_bytes().iter().cloned().map(Symbol::Char).collect()
}
