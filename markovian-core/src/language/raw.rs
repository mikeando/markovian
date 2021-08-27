//! The raw module contains a simple language implementation.
//! A Language contains a list of Productions.
//!
//! Each Production is a symbol (as a String) that the production acts upon,
//! the relative likelyhood of picking that production from all the ones
//! that operate on that symbols, and the sequence of symbols and literals
//! that the symbol becomes if that symbol is selected.
//!
//! This format for the language is very inefficient. It is both wasteful
//! of space and would be slow to use to perform operations with. However
//! it is easy to understand, edit and save or load.
//!
//! A much faster and more space efficient language is available in
//! `compiled`, and a `raw::Language` can easily be transformed
//! into a `compiled::Language`
//!
//! Example:
//!
//! ```
//! use markovian_core::language::raw::*;
//! use markovian_core::language::compiled;
//! use markovian_core::utils::nf32::nf32;
//!
//! let mut l: Language<String> = Language::new();
//! l.entries.push(
//!     Production{
//!        from: Symbol("A".into()),
//!        weight: nf32(1.0),
//!        to: vec![SymbolOrLiteral::literal("A"), SymbolOrLiteral::symbol("B")],
//!    }
//! );
//! l.entries.push(
//!     Production{
//!        from: Symbol("B".into()),
//!        weight: nf32(1.0),
//!        to: vec![SymbolOrLiteral::literal("B")],
//!    }
//! );
//! let compiled_language = compiled::Language::from_raw(&l);
//! ```

use std::hash::Hash;

use snafu::{ensure, OptionExt, ResultExt, Snafu};

use super::context::{ContextError, InvalidOperation};
use super::parse;
use crate::utils::nf32::nf32;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new<T: Into<String>>(v: T) -> Self {
        Symbol(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Literal<T>(pub T);

impl<T> Literal<T> {
    pub fn new<V: Into<T>>(v: V) -> Self {
        Literal(v.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash)]
pub struct Production<T> {
    pub from: Symbol,
    pub weight: nf32,
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

impl<T> Default for Language<T> {
    fn default() -> Self {
        Self { entries: vec![] }
    }
}

impl<T> Language<T> {
    pub fn new() -> Language<T> {
        Language::default()
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

pub trait Context {
    fn get_word_list(&mut self, name: &str) -> Result<Vec<String>, ContextError>;
    fn get_language(&mut self, name: &str) -> Result<Language<String>, ContextError>;
}

pub struct EmptyContext;
impl Context for EmptyContext {
    fn get_word_list(&mut self, name: &str) -> Result<Vec<String>, ContextError> {
        InvalidOperation {
            op_name: format!("{} does not exist in EmptyContext", name),
        }
        .fail()
    }
    fn get_language(&mut self, name: &str) -> Result<Language<String>, ContextError> {
        InvalidOperation {
            op_name: format!("{} does not exist in EmptyContext", name),
        }
        .fail()
    }
}

//TODO: Create more informative error types?
#[derive(Debug, Snafu)]
pub enum DirectiveError {
    #[snafu(display(
        "incorrect argument count for {}, expected {} but got {}",
        f,
        expected,
        actual
    ))]
    IncorrectArgumentCount {
        f: String,
        expected: usize,
        actual: usize,
    },

    #[snafu(display(
        "invalid argument {} for {}, expected {} but got {}",
        i,
        f,
        expected,
        actual
    ))]
    IncorrectArgumentType {
        i: usize,
        f: String,
        expected: String,
        actual: String,
    },

    #[snafu(display("context error"))]
    DirectiveContextError {
        #[snafu(source(from(ContextError, Box::new)))]
        source: Box<ContextError>,
    },

    #[snafu(display("import list error: {}", mesg))]
    ImportListError { mesg: String },

    #[snafu(display("unknown directive '{}'", directive))]
    UnknownDirective { directive: String },
}

// import_list( "Name.txt" Symbol )
pub fn apply_import_list_directive(
    language: &mut Language<String>,
    directive: &parse::Directive,
    ctx: &mut dyn Context,
) -> Result<(), DirectiveError> {
    ensure!(
        directive.arguments.len() == 2,
        IncorrectArgumentCount {
            f: "import_list",
            expected: 2usize,
            actual: directive.arguments.len()
        }
    );
    let name = directive.arguments[0]
        .as_literal()
        .context(IncorrectArgumentType {
            i: 0usize,
            f: "import_list",
            expected: "literal",
            actual: "symbol",
        })?;
    let from = Symbol(
        directive.arguments[1]
            .as_symbol()
            .context(IncorrectArgumentType {
                i: 1usize,
                f: "import_list",
                expected: "symbol",
                actual: "literal",
            })?
            .0
            .clone(),
    );
    for v in ctx.get_word_list(&name.0).map_err(|e| {
        ImportListError {
            mesg: format!("{:?}", e),
        }
        .build()
    })? {
        language.entries.push(Production {
            from: from.clone(),
            weight: nf32(1.0),
            to: vec![SymbolOrLiteral::literal(v)],
        });
    }
    Ok(())
}

pub fn apply_import_language_directive(
    language: &mut Language<String>,
    directive: &parse::Directive,
    ctx: &mut dyn Context,
) -> Result<(), DirectiveError> {
    // TODO we should support other modes rather than import everything into the
    //      root namespace.
    ensure!(
        directive.arguments.len() == 1,
        IncorrectArgumentCount {
            f: "import_language",
            expected: 1usize,
            actual: directive.arguments.len()
        }
    );

    let name = directive.arguments[0]
        .as_literal()
        .context(IncorrectArgumentType {
            i: 0usize,
            f: "import_language",
            expected: "literal",
            actual: "symbol",
        })?;
    let l: Language<String> = ctx.get_language(&name.0).context(DirectiveContextError)?;
    for e in l.entries {
        language.entries.push(e.clone());
    }
    Ok(())
}

pub fn apply_directive(
    language: &mut Language<String>,
    directive: &parse::Directive,
    ctx: &mut dyn Context,
) -> Result<(), DirectiveError> {
    match &directive.name[..] {
        // import_list( "Name.txt" Symbol )
        "import_list" => apply_import_list_directive(language, directive, ctx),
        // import_language( "Foo.lang" )
        "import_language" => apply_import_language_directive(language, directive, ctx),
        _ => Err(UnknownDirective {
            directive: directive.name.clone(),
        }
        .build()),
    }
}

#[derive(Debug, Snafu)]
pub enum LoadLanguageError {
    #[snafu(display("invalid line"))]
    InvalidLine,
    #[snafu(display("invalid directive"))]
    XDirectiveError { source: DirectiveError },
}

pub fn load_language(
    language_raw: &str,
    ctx: &mut dyn Context,
) -> Result<Language<String>, LoadLanguageError> {
    let mut language = Language::new();
    for line in language_raw.lines() {
        // Check if its a comment or empty line.
        let (_, rest) = parse::eat_spaces(line).map_err(|_e| LoadLanguageError::InvalidLine)?;
        if rest.is_empty() || rest.starts_with('#') {
            continue;
        }

        match parse::parse_language_line(line) {
            Err(e) => {
                println!("Unable to parse line '{:?} {:?}'", line, e);
                return Err(LoadLanguageError::InvalidLine);
            }
            Ok(parse::Line::MultiProduction(p)) => {
                for production in p {
                    language.entries.push(production);
                }
            }
            Ok(parse::Line::Directive(d)) => {
                apply_directive(&mut language, &d, ctx).context(XDirectiveError)?;
            }
        }
    }
    Ok(language)
}
