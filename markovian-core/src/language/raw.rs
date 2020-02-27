
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new<T: Into<String>>(v: T) -> Self {
        Symbol(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal(pub String);

impl Literal {
    pub fn new<T: Into<String>>(v: T) -> Self {
        Literal(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolOrLiteral {
    Symbol(Symbol),
    Literal(Literal),
}

impl SymbolOrLiteral {
    pub fn literal<T: Into<String>>(v: T) -> Self {
        SymbolOrLiteral::Literal(Literal::new(v))
    }

    pub fn symbol<T: Into<String>>(v: T) -> Self {
        SymbolOrLiteral::Symbol(Symbol::new(v))
    }

    pub fn as_symbol(&self) -> Option<&Symbol> {
        match &self {
            SymbolOrLiteral::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_literal(&self) -> Option<&Literal> {
        match &self {
            SymbolOrLiteral::Literal(s) => Some(s),
            _ => None,
        }
    }

    pub fn map_literal<F, E>(self, f: F) -> Result<SymbolOrLiteral, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let r = match self {
            SymbolOrLiteral::Symbol(_) => self,
            SymbolOrLiteral::Literal(Literal(v)) => SymbolOrLiteral::literal(f(v)?),
        };
        Ok(r)
    }

    pub fn map_symbol<F, E>(self, f: F) -> Result<SymbolOrLiteral, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let r = match self {
            SymbolOrLiteral::Symbol(Symbol(v)) => SymbolOrLiteral::symbol(f(v)?),
            SymbolOrLiteral::Literal(_) => self,
        };
        Ok(r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production {
    pub from: Symbol,
    pub weight: u32,
    pub to: Vec<SymbolOrLiteral>,
}

impl Production {
    pub fn map_literals<F, E>(self, f: F) -> Result<Production, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let mut result = self;
        result.to = result
            .to
            .into_iter()
            .map(|s| s.map_literal(&f))
            .collect::<Result<_, _>>()?;
        Ok(result)
    }

    pub fn map_symbols<F, E>(self, f: F) -> Result<Production, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let mut result = self;
        result.from = Symbol::new(f(result.from.0)?);
        result.to = result
            .to
            .into_iter()
            .map(|s| s.map_symbol(&f))
            .collect::<Result<_, _>>()?;
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct Language {
    pub entries: Vec<Production>,
}

impl Language {
    pub fn new() -> Language {
        Language { entries: vec![] }
    }

    pub fn map_literals<F, E>(self, f: F) -> Result<Language, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let mut result = self;
        result.entries = result
            .entries
            .into_iter()
            .map(|p| p.map_literals(&f))
            .collect::<Result<_, _>>()?;
        Ok(result)
    }

    pub fn map_symbols<F, E>(self, f: F) -> Result<Language, E>
    where
        F: Fn(String) -> Result<String, E>,
    {
        let mut result = self;
        result.entries = result
            .entries
            .into_iter()
            .map(|p| p.map_symbols(&f))
            .collect::<Result<_, _>>()?;
        Ok(result)
    }
}