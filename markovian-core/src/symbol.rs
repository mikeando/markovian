#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum Symbol {
    Start,
    End,
    Char(u8),
    Compound(Vec<u8>),
}
