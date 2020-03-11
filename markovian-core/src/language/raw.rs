#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new<T: Into<String>>(v: T) -> Self {
        Symbol(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Literal<T>(pub T);

impl<T> Literal<T> {
    pub fn new<V: Into<T>>(v: V) -> Self {
        Literal(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SymbolOrLiteral<T> {
    Symbol(Symbol),
    Literal(Literal<T>),
}

impl<T> SymbolOrLiteral<T> {
    pub fn literal<V: Into<T>>(v: V) -> Self {
        SymbolOrLiteral::Literal(Literal::new(v))
    }

    pub fn symbol<V: Into<String>>(v: V) -> Self {
        SymbolOrLiteral::Symbol(Symbol::new(v))
    }

    pub fn as_symbol(&self) -> Option<&Symbol> {
        match &self {
            SymbolOrLiteral::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_literal(&self) -> Option<&Literal<T>> {
        match &self {
            SymbolOrLiteral::Literal(s) => Some(s),
            _ => None,
        }
    }

    pub fn map_literal<F, E, U>(self, f: F) -> Result<SymbolOrLiteral<U>, E>
    where
        F: Fn(T) -> Result<U, E>,
    {
        let r = match self {
            SymbolOrLiteral::Symbol(Symbol(v)) => SymbolOrLiteral::symbol(v),
            SymbolOrLiteral::Literal(Literal(v)) => SymbolOrLiteral::literal(f(v)?),
        };
        Ok(r)
    }

    pub fn map_symbol<F, E>(self, f: F) -> Result<SymbolOrLiteral<T>, E>
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
pub struct Production<T> {
    pub from: Symbol,
    pub weight: u32,
    pub to: Vec<SymbolOrLiteral<T>>,
}

impl<T> Production<T> {
    pub fn map_literals<F, E, U>(self, f: F) -> Result<Production<U>, E>
    where
        F: Fn(T) -> Result<U, E>,
    {
        let to: Vec<SymbolOrLiteral<U>> = self
            .to
            .into_iter()
            .map(|s| s.map_literal(&f))
            .collect::<Result<_, _>>()?;

        let result: Production<U> = Production {
            to,
            from: self.from,
            weight: self.weight,
        };
        Ok(result)
    }

    pub fn map_symbols<F, E>(self, f: F) -> Result<Production<T>, E>
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Language<T> {
    pub entries: Vec<Production<T>>,
}

impl<T> Language<T> {
    pub fn new() -> Language<T> {
        Language { entries: vec![] }
    }

    pub fn map_literals<F, E, U>(self, f: F) -> Result<Language<U>, E>
    where
        F: Fn(T) -> Result<U, E>,
    {
        let entries: Vec<Production<U>> = self
            .entries
            .into_iter()
            .map(|p| p.map_literals(&f))
            .collect::<Result<_, _>>()?;

        let result: Language<U> = Language { entries };
        Ok(result)
    }

    pub fn map_symbols<F, E>(self, f: F) -> Result<Language<T>, E>
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
