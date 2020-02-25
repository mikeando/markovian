use rand::Rng;
use std::collections::BTreeMap;

// TODO: Remove all unwraps and use proper error handling.

#[derive(Debug, Ord, Eq, PartialEq, PartialOrd, Copy, Clone, Default)]
pub struct SymbolId(u32);

mod raw {
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

        pub fn map_literal<F,E>(self, f: F) -> Result<SymbolOrLiteral,E>
        where
            F: Fn(String) -> Result<String,E>,
        {
            let r = match self {
                SymbolOrLiteral::Symbol(_) => self,
                SymbolOrLiteral::Literal(Literal(v)) => SymbolOrLiteral::literal(f(v)?),
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
        pub fn map_literals<F,E>(self, f: F) -> Result<Production,E>
        where
            F: Fn(String) -> Result<String,E>,
        {
            let mut result = self;
            result.to = result.to
                .into_iter()
                .map( |s| s.map_literal(&f) )
                .collect::<Result<_,_>>()?;
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

        pub fn map_literals<F,E>(self, f: F) -> Result<Language,E>
        where
            F: Fn(String) -> Result<String,E>,
        {
            let mut result = self;
            result.entries = result.entries
                .into_iter()
                .map(
                    |p| p.map_literals(&f)
                )
                .collect::<Result<_,_>>()?;
            Ok(result)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Production {
    weight: u32,
    keys: Vec<SymbolId>,
}

impl Production {
    pub fn new(weight: u32, keys: &[SymbolId]) -> Production {
        Production {
            weight,
            keys: keys.to_vec(),
        }
    }
}

pub trait SimpleRNG {
    fn gen_range(&mut self, low: u32, high: u32) -> u32;
}

impl<T> SimpleRNG for T
where
    T: Rng,
{
    fn gen_range(&mut self, low: u32, high: u32) -> u32 {
        Rng::gen_range(self, low, high)
    }
}

pub fn choose_by_weight<'a, R: SimpleRNG, T, F>(
    rng: &mut R,
    values: &'a [T],
    weightfn: &F,
) -> Option<&'a T>
where
    R: SimpleRNG,
    F: Fn(&T) -> u32,
{
    let sum_w: u32 = values.iter().map(weightfn).sum();
    if sum_w == 0 {
        return None;
    }
    let r = rng.gen_range(0, sum_w);
    let mut s: u32 = 0;
    for v in values {
        s += weightfn(v);
        if s > r {
            return Some(v);
        }
    }
    unreachable!();
}

#[derive(Debug, Default)]
pub struct ProductionGroup {
    productions: Vec<Production>,
}

impl ProductionGroup {
    pub fn new() -> ProductionGroup {
        ProductionGroup {
            productions: vec![],
        }
    }

    pub fn add(&mut self, p: Production) {
        self.productions.push(p);
    }
}

#[derive(Debug, Default)]
pub struct Language {
    terminals_by_value: BTreeMap<String, SymbolId>,
    terminals_by_id: BTreeMap<SymbolId, String>,
    symbols_by_name: BTreeMap<String, SymbolId>,
    productions_by_id: BTreeMap<SymbolId, ProductionGroup>,
    last_id: SymbolId,
}

impl Language {
    pub fn new() -> Language {
        Language {
            terminals_by_value: BTreeMap::new(),
            terminals_by_id: BTreeMap::new(),
            symbols_by_name: BTreeMap::new(),
            productions_by_id: BTreeMap::new(),
            last_id: SymbolId(0),
        }
    }

    pub fn new_symbol(&mut self) -> SymbolId {
        self.last_id.0 += 1;
        self.last_id
    }

    pub fn terminal<T: Into<String>>(&mut self, v: T) -> SymbolId {
        //TODO: Check if it already exists in the list.
        let symbol = self.new_symbol();
        let value: String = v.into();
        self.terminals_by_value.insert(value.clone(), symbol);
        self.terminals_by_id.insert(symbol, value);
        symbol
    }

    pub fn add_or_get_named_symbol<T: Into<String>>(&mut self, v: T) -> SymbolId {
        let s: String = v.into();

        if let Some(symbol_id) = self.symbols_by_name.get(&s) {
            return *symbol_id;
        }
        let symbol_id = self.new_symbol();
        self.symbols_by_name.insert(s, symbol_id);
        symbol_id
    }

    pub fn add_or_get_literal<T: Into<String>>(&mut self, v: T) -> SymbolId {
        let s: String = v.into();
        if let Some(symbol_id) = self.terminals_by_value.get(&s) {
            return *symbol_id;
        }
        let symbol_id = self.new_symbol();
        self.terminals_by_id.insert(symbol_id, s.clone());
        self.terminals_by_value.insert(s, symbol_id);
        symbol_id
    }

    pub fn token_by_name<T: Into<String>>(&self, v: T) -> Option<SymbolId> {
        self.symbols_by_name.get(&v.into()).cloned()
    }

    pub fn add_terminal_production<T: Into<String>>(
        &mut self,
        symbol_id: SymbolId,
        weight: u32,
        v: T,
    ) {
        let symbol = self.terminal(v);
        self.add_production(symbol_id, weight, &[symbol])
    }

    pub fn add_production(&mut self, symbol_id: SymbolId, weight: u32, keys: &[SymbolId]) {
        self.productions_by_id
            .entry(symbol_id)
            .or_insert_with(ProductionGroup::new)
            .add(Production::new(weight, keys))
    }

    pub fn expand<R: SimpleRNG>(&self, tokens: &[SymbolId], rng: &mut R) -> String {
        let mut expansion_stack: Vec<&[SymbolId]> = vec![tokens];
        let mut complete: String = "".to_string();

        while let Some(cur_tokens) = expansion_stack.pop() {
            if cur_tokens.is_empty() {
                continue;
            }
            let token = cur_tokens[0];
            if cur_tokens.len() > 1 {
                expansion_stack.push(&cur_tokens[1..]);
            }
            if let Some(s) = self.terminals_by_id.get(&token) {
                complete = format!("{}{}", complete, s);
            } else {
                //TODO: unwrap is bad.
                //TODO: Need to select by weight.
                //TODO: Need to handle productions being empty.
                let pg = self.productions_by_id.get(&token).unwrap();
                let p = choose_by_weight(rng, &pg.productions, &|x: &Production| x.weight).unwrap();
                expansion_stack.push(&p.keys);
            }
        }
        complete
    }



    //TODO: This really is only needed at the raw level. It's also simpler there.
    pub fn rename_symbols<F>(self, f: F) -> Language
    where
        F: Fn(String) -> String,
    {
        let mut result = self;
        result.symbols_by_name = result
            .symbols_by_name
            .into_iter()
            .map(|(k, v)| (f(k), v))
            .collect();
        result
    }

    pub fn format_symbol(&self, sid: SymbolId) -> String {
        if let Some(v) = self.terminals_by_id.get(&sid) {
            return format!("'{}'", v);
        }
        if let Some((v, _id)) = self.symbols_by_name.iter().find(|(_v, id)| **id == sid) {
            return v.clone();
        }
        format!("{:?}", sid)
    }

    pub fn from_raw(raw: &raw::Language) -> Self {
        let mut result = Language::new();
        for p in &raw.entries {
            let from = result.add_or_get_named_symbol(&p.from.0);
            let prod: Vec<SymbolId> =
                p.to.iter()
                    .map(|q| match q {
                        raw::SymbolOrLiteral::Symbol(v) => result.add_or_get_named_symbol(&v.0),
                        raw::SymbolOrLiteral::Literal(v) => result.add_or_get_literal(&v.0),
                    })
                    .collect();
            result.add_production(from, p.weight, &prod);
        }
        result
    }

    pub fn to_raw(&self) -> raw::Language {
        let x: Vec<(String, &ProductionGroup)> = self
            .productions_by_id
            .iter()
            .map(|(k, v)| (self.format_symbol(*k), v))
            .collect();
        let y: Vec<(String, Production)> = x
            .iter()
            .flat_map(|(k, g)| g.productions.iter().map(move |v| (k.clone(), v.clone())))
            .collect();
        let z: Vec<(String, u32, Vec<SymbolId>)> = y
            .iter()
            .map(|(k, v)| (k.clone(), v.weight, v.keys.clone()))
            .collect();
        let mut entries: Vec<raw::Production> = vec![];
        for e in z {
            let from = raw::Symbol(e.0);
            let weight = e.1;
            let to: Option<Vec<_>> =
                e.2.into_iter()
                    .map(|vid| {
                        let s = self
                            .symbols_by_name
                            .iter()
                            .find(|(_k, id)| **id == vid)
                            .map(|(k, _id)| k);
                        if let Some(s) = s {
                            return Some(raw::SymbolOrLiteral::symbol(s));
                        }
                        self.terminals_by_id
                            .get(&vid)
                            .cloned()
                            .map(raw::SymbolOrLiteral::literal)
                    })
                    .collect();
            let to = to.unwrap();
            entries.push(raw::Production { from, weight, to })
        }
        raw::Language { entries }
    }
}

pub mod parse {

    use super::raw;
    use raw::Production;
    use raw::Symbol;
    use raw::SymbolOrLiteral;

    #[derive(Debug, PartialEq, Eq)]
    pub enum ParseError<'a> {
        MissingSymbol,
        InvalidSymbol(&'a str),
        GeneralError,
    }

    impl<'a> ParseError<'a> {
        pub fn description(&self) -> String {
            match self {
                ParseError::MissingSymbol => format!("missing symbol"),
                ParseError::InvalidSymbol(s) => format!("'{}' is not a valid symbol", s),
                ParseError::GeneralError => format!("general error"),
            }
        }
    }

    pub fn eat_spaces(v: &str) -> Result<((), &str), ParseError> {
        let mut rest = v;
        //TODO: This should really work on chars..
        while !rest.is_empty() && (&rest[0..1] == " " || &rest[0..1] == "\t") {
            rest = &rest[1..];
        }
        Ok(((), rest))
    }

    pub fn eat_nonspaces(v: &str) -> Result<(&str, &str), ParseError> {
        let mut rest = v;
        //TODO: This should really work on chars..
        while !rest.is_empty() && (&rest[0..1] != " " && &rest[0..1] != "\t") {
            rest = &rest[1..];
        }
        Ok((&v[0..(v.len() - rest.len())], rest))
    }

    pub fn is_symbol_character(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    pub fn eat_symbol_chars(v: &str) -> Result<(&str, &str), ParseError> {
        for (idx, cc) in v.char_indices() {
            let mut b = [0; 4];
            if !is_symbol_character(cc) {
                return Ok((&v[..idx], &v[idx..]));
            }
        }
        Ok((v, ""))
    }

    pub fn take<'a>(p: &str, v: &'a str) -> Result<(&'a str, &'a str), ParseError<'a>> {
        let len = p.len();
        if v.len() < len {
            return Err(ParseError::GeneralError);
        }
        let pp = &v[0..len];
        if pp != p {
            return Err(ParseError::GeneralError);
        }
        Ok((pp, &v[len..]))
    }

    pub fn take_char(c: char, v: &str) -> Result<((), &str), ParseError> {
        let mut cit = v.chars();
        if cit.next() != Some(c) {
            Err(ParseError::GeneralError)
        } else {
            Ok(((), cit.as_str()))
        }
    }

    pub fn take_while_not_char(c: char, v: &str) -> Result<(&str, &str), ParseError> {
        for (idx, cc) in v.char_indices() {
            if cc == c {
                return Ok((&v[..idx], &v[idx..]));
            }
        }
        Err(ParseError::GeneralError)
    }

    pub fn parse_weight(v: &str) -> Result<(u32, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        let (x, rest): (&str, &str) = eat_nonspaces(rest)?;
        let (_, rest) = eat_spaces(rest)?;
        let w: u32 = x.parse::<u32>().map_err(|_| ParseError::GeneralError)?;
        Ok((w, rest))
    }

    pub fn parse_symbol(v: &str) -> Result<(Symbol, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        let (x, rest): (&str, &str) =
            eat_symbol_chars(rest).map_err(|_e| ParseError::MissingSymbol)?;
        let (_, rest) = eat_spaces(rest)?;
        if x.is_empty() {
            return Err(ParseError::MissingSymbol);
        }
        Ok((Symbol::new(x), rest))
    }

    pub fn parse_tag<'a>(tag: &str, v: &'a str) -> Result<(&'a str, &'a str), ParseError<'a>> {
        let (_, rest) = eat_spaces(v)?;
        let (x, rest): (&str, &str) = eat_nonspaces(rest)?;
        let (_, rest) = eat_spaces(rest)?;
        if x == tag {
            Ok((x, rest))
        } else {
            Err(ParseError::GeneralError)
        }
    }

    pub fn parse_string(v: &str) -> Result<(&str, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        let (_, rest) = take_char('"', rest)?;
        let (x, rest) = take_while_not_char('"', rest)?;
        let (_, rest) = take_char('"', rest)?;
        let (_, rest) = eat_spaces(rest)?;
        Ok((x, rest))
    }

    pub fn parse_alt<'a, F1, F2, I, R>(v: I, f1: F1, f2: F2) -> Result<R, ParseError<'a>>
    where
        F1: Fn(I) -> Result<R, ParseError<'a>>,
        F2: Fn(I) -> Result<R, ParseError<'a>>,
        R: 'a,
        I: 'a + Copy,
    {
        let r1 = f1(v);
        if r1.is_ok() {
            return r1;
        }
        let r2 = f2(v);
        if r2.is_ok() {
            return r2;
        }
        Err(ParseError::GeneralError)
    }

    //NOTE: I couldn't get this working cleanly with parse_alt
    //      so I've just inlined it
    pub fn parse_symbol_or_literal(v: &str) -> Result<(SymbolOrLiteral, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        match parse_string(rest) {
            Ok((w, rest)) => {
                return Ok((SymbolOrLiteral::literal(w), rest));
            }
            Err(_e) => {}
        }
        match parse_symbol(rest) {
            Ok((w, rest)) => {
                return Ok((SymbolOrLiteral::Symbol(w), rest));
            }
            Err(_e) => {}
        }
        Err(ParseError::GeneralError)
    }

    pub fn parse_symbol_or_literal_list(
        v: &str,
    ) -> Result<(Vec<SymbolOrLiteral>, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        let mut result = vec![];
        let mut rest = rest;
        loop {
            let r = parse_symbol_or_literal(rest);
            match r {
                Ok((s, rest_)) => {
                    result.push(s);
                    rest = rest_;
                }
                Err(e) => {
                    break;
                }
            }
        }
        Ok((result, rest))
    }

    pub fn parse_production(v: &str) -> Result<(Vec<Production>, &str), ParseError> {
        let (weight, rest): (u32, &str) = parse_weight(v)?;
        let (from, rest): (Symbol, &str) = parse_symbol(rest)?;
        let (_, rest) = parse_tag("=>", rest)?;

        let mut options: Vec<Vec<SymbolOrLiteral>> = vec![];
        let mut rest = rest;
        loop {
            let (symbols, r): (Vec<SymbolOrLiteral>, &str) = parse_symbol_or_literal_list(rest)?;
            rest = r;
            options.push(symbols);
            let (_, r) = eat_spaces(r)?;
            let e = take_char('|', r);
            match e {
                Ok((_, r)) => {
                    let (_, r) = eat_spaces(r)?;
                    rest = r;
                }
                Err(_) => break,
            }
        }
        if options.is_empty() {
            return Err(ParseError::GeneralError);
        }

        let (_, rest) = eat_spaces(rest)?;
        Ok((
            options
                .into_iter()
                .map(|v| Production {
                    weight,
                    from: from.clone(),
                    to: v,
                })
                .collect(),
            rest,
        ))
    }

    #[derive(Debug, PartialEq, Eq)]
    pub struct Directive {
        pub name: String,
        pub arguments: Vec<SymbolOrLiteral>,
    }

    pub fn parse_directive(v: &str) -> Result<(Directive, &str), ParseError> {
        let (_, rest) = eat_spaces(v)?;
        let (_, rest) = take_char('@', rest)?;
        let (name, rest) = parse_symbol(rest)?;
        let (_, rest) = eat_spaces(rest)?;
        let (_, rest) = take_char('(', rest)?;
        let (_, rest) = eat_spaces(rest)?;
        let (arguments, rest) = parse_symbol_or_literal_list(rest)?;
        let (_, rest) = take_char(')', rest)?;
        Ok((
            Directive {
                name: name.0,
                arguments,
            },
            rest,
        ))
    }

    pub enum Line {
        MultiProduction(Vec<Production>),
        Directive(Directive),
    }

    pub fn parse_language_line(line: &str) -> Result<Line, ParseError> {
        let r = parse_directive(line);
        if let Ok(v) = r {
            let (_, rest) = eat_spaces(v.1).unwrap();
            if rest.is_empty() {
                return Ok(Line::Directive(v.0));
            }
        }
        let r = parse_production(line);
        if let Ok(v) = r {
            let (_, rest) = eat_spaces(v.1).unwrap();
            if rest.is_empty() {
                return Ok(Line::MultiProduction(v.0));
            }
        }
        Err(ParseError::GeneralError)
    }
}

#[derive(Debug)]
pub enum ContextError {
    InvalidOperation,
    InvalidKey,
}

pub trait Context {
    fn get_word_list(&self, name: &str) -> Result<Vec<String>, ContextError>;
    fn get_language(&self, name: &str) -> Result<raw::Language, ContextError>;
}

pub struct EmptyContext;
impl Context for EmptyContext {
    fn get_word_list(&self, name: &str) -> Result<Vec<String>, ContextError> {
        Err(ContextError::InvalidOperation)
    }
    fn get_language(&self, name: &str) -> Result<raw::Language, ContextError> {
        Err(ContextError::InvalidOperation)
    }
}

pub fn apply_directive(
    language: &mut raw::Language,
    directive: &parse::Directive,
    ctx: &mut dyn Context,
) {
    println!("Applying directive : {:?}", directive);
    match &directive.name[..] {
        // import_list( "Name.txt" Symbol )
        "import_list" => {
            println!("args = {:?}", directive.arguments);
            assert_eq!(directive.arguments.len(), 2); // TODO: This should become an error
            let name = directive.arguments[0].as_literal().unwrap();
            let from = raw::Symbol(directive.arguments[1].as_symbol().unwrap().0.clone());
            for v in ctx.get_word_list(&name.0).unwrap() {
                language.entries.push(raw::Production {
                    from: from.clone(),
                    weight: 1,
                    to: vec![raw::SymbolOrLiteral::literal(v)],
                });
            }
        }
        // import_language( "Foo.lang" )
        "import_language" => {
            // TODO we should support other modes rather than import everything into the
            //      root namespace.
            assert_eq!(directive.arguments.len(), 1); //TODO: This should be an error
            let name = directive.arguments[0].as_literal().unwrap();
            let l: raw::Language = ctx.get_language(&name.0).unwrap();
            for e in l.entries {
                language.entries.push(e.clone());
            }
        }
        //TODO: Make this error.
        _ => {
            println!("Unknown directive: {:?}", directive.name);
        }
    }
}

pub fn load_language(language_raw: &str, ctx: &mut dyn Context) -> raw::Language {
    let mut language = raw::Language::new();
    for line in language_raw.lines() {
        match parse::parse_language_line(line) {
            Err(e) => {
                println!("Unable to parse line '{:?} {:?}'", line, e);
            }
            Ok(parse::Line::MultiProduction(p)) => {
                for production in p {
                    language.entries.push(production);
                }
            }
            Ok(parse::Line::Directive(d)) => {
                apply_directive(&mut language, &d, ctx);
            }
        }
    }
    language
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn dummy_language() -> raw::Language {
        let rules = r#"2 tofu => "tofu"
               1 tofu => tofu " " tofu
               3 tofu => "I like to eat " tofu"#;

        let mut ctx = EmptyContext;
        load_language(rules, &mut ctx)
    }

    fn towns_language_mod() -> raw::Language {
        let mut word_lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let ll = &[
            "borough", "city", "fort", "hamlet", "parish", "town", "township", "village",
        ];
        word_lists.insert(
            "city_types.txt".to_string(),
            ll.iter().map(|v| v.to_string()).collect(),
        );
        let mut ctx = MockContext {
            word_lists,
            languages: BTreeMap::new(),
        };

        let rules = r#"1 town => town_x
            1 town => preword " " town_x
            1 preword => "new" | "old" | "north" | "south" | "east" | "west" | "upper" | "lower"
            1 town_x => descriptive_word " " settlement_type
            1 town_x => descriptive_word " " place_word
            1 town_x => descriptive_word common_suffix
            1 town_x => person_name "s " settlement_type
            1 town_x => person_name "s " place_word
            1 town_x => person_name common_suffix
            @import_list("city_types.txt" settlement_type)
            1 place_word => "acres"
            1 place_word => "basin"
            1 place_word => "bottom"
            1 place_word => "bridge"
            1 place_word => "corner"
            1 place_word => "crossing"
            1 place_word => "drift"
            1 place_word => "fell"
            1 place_word => "ferry"
            1 place_word => "flats"
            1 place_word => "ford"
            1 place_word => "gap"
            1 place_word => "garden"
            1 place_word => "gate"
            1 place_word => "grove"
            1 place_word => "heath"
            1 place_word => "harbour"
            1 place_word => "heights"
            1 place_word => "hole"
            1 place_word => "jetty"
            1 place_word => "landing"
            1 place_word => "lane"
            1 place_word => "meadow"
            1 place_word => "mound"
            1 place_word => "moor"
            1 place_word => "mouth"
            1 place_word => "nook"
            1 place_word => "notch"
            1 place_word => "orchard"
            1 place_word => "point"
            1 place_word => "ranch"
            1 place_word => "rim"
            1 place_word => "rise"
            1 place_word => "run"
            1 place_word => "wharf"
            1 place_word => "view"
            1 place_word => "vineyard"
            1 place_word => "vista"
            1 common_suffix => "bar"
            1 common_suffix => "bee"
            1 common_suffix => "berg"
            1 common_suffix => "berry"
            1 common_suffix => "boro"
            1 common_suffix => "burg"
            1 common_suffix => "bugrh"
            1 common_suffix => "bury"
            1 common_suffix => "by"
            1 common_suffix => "cester"
            1 common_suffix => "chase"
            1 common_suffix => "chester"
            1 common_suffix => "cross"
            1 common_suffix => "don"
            1 common_suffix => "ham"
            1 common_suffix => "haven"
            1 common_suffix => "kirk"
            1 common_suffix => "lea"
            1 common_suffix => "ly"
            1 common_suffix => "mar"
            1 common_suffix => "mead"
            1 common_suffix => "meade"
            1 common_suffix => "mer"
            1 common_suffix => "mont"
            1 common_suffix => "more"
            1 common_suffix => "moore"
            1 common_suffix => "rose"
            1 common_suffix => "rise"
            1 common_suffix => "side"
            1 common_suffix => "ton"
            1 common_suffix => "ville"
            1 common_suffix => "wall"
            1 common_suffix => "way"
            1 common_suffix => "wick"
            1 common_suffix => "which"
            1 common_suffix => "worth"
            1 person_name => "jim"
            1 descriptive_word => "pink""#;

        //TODO: Add more names
        //TODO: Add more descriptive terms
        //TODO: Add geographical features
        //TODO: Add weather
        //TODO: Add colors
        //TODO: Add air_quality
        //TODO: Add royalty
        //TODO: Add religion
        //TODO: Add animals
        //TODO: Add quality (good grand messy)
        //TODO: Add jobs
        //TODO: Add seasons

        load_language(rules, &mut ctx)
    }

    fn towns_language() -> raw::Language {
        let rules = r#"1 town => town_x
               1 town => preword " " town_x
               1 preword => "new"
               1 preword => "old"
               1 preword => "north"
               1 preword => "south"
               1 preword => "east"
               1 preword => "west"
               1 preword => "upper"
               1 preword => "lower"
               1 town_x => descriptive_word " " settlement_type
               1 town_x => descriptive_word " " place_word
               1 town_x => descriptive_word common_suffix
               1 town_x => person_name "s " settlement_type
               1 town_x => person_name "s " place_word
               1 town_x => person_name common_suffix
               1 settlement_type => "borough"
               1 settlement_type => "city"
               1 settlement_type => "fort"
               1 settlement_type => "hamlet"
               1 settlement_type => "parish"
               1 settlement_type => "town"
               1 settlement_type => "township"
               1 settlement_type => "village"
               1 place_word => "acres"
               1 place_word => "basin"
               1 place_word => "bottom"
               1 place_word => "bridge"
               1 place_word => "corner"
               1 place_word => "crossing"
               1 place_word => "drift"
               1 place_word => "fell"
               1 place_word => "ferry"
               1 place_word => "flats"
               1 place_word => "ford"
               1 place_word => "gap"
               1 place_word => "garden"
               1 place_word => "gate"
               1 place_word => "grove"
               1 place_word => "heath"
               1 place_word => "harbour"
               1 place_word => "heights"
               1 place_word => "hole"
               1 place_word => "jetty"
               1 place_word => "landing"
               1 place_word => "lane"
               1 place_word => "meadow"
               1 place_word => "mound"
               1 place_word => "moor"
               1 place_word => "mouth"
               1 place_word => "nook"
               1 place_word => "notch"
               1 place_word => "orchard"
               1 place_word => "point"
               1 place_word => "ranch"
               1 place_word => "rim"
               1 place_word => "rise"
               1 place_word => "run"
               1 place_word => "wharf"
               1 place_word => "view"
               1 place_word => "vineyard"
               1 place_word => "vista"
               1 common_suffix => "bar"
               1 common_suffix => "bee"
               1 common_suffix => "berg"
               1 common_suffix => "berry"
               1 common_suffix => "boro"
               1 common_suffix => "burg"
               1 common_suffix => "bugrh"
               1 common_suffix => "bury"
               1 common_suffix => "by"
               1 common_suffix => "cester"
               1 common_suffix => "chase"
               1 common_suffix => "chester"
               1 common_suffix => "cross"
               1 common_suffix => "don"
               1 common_suffix => "ham"
               1 common_suffix => "haven"
               1 common_suffix => "kirk"
               1 common_suffix => "lea"
               1 common_suffix => "ly"
               1 common_suffix => "mar"
               1 common_suffix => "mead"
               1 common_suffix => "meade"
               1 common_suffix => "mer"
               1 common_suffix => "mont"
               1 common_suffix => "more"
               1 common_suffix => "moore"
               1 common_suffix => "rose"
               1 common_suffix => "rise"
               1 common_suffix => "side"
               1 common_suffix => "ton"
               1 common_suffix => "ville"
               1 common_suffix => "wall"
               1 common_suffix => "way"
               1 common_suffix => "wick"
               1 common_suffix => "which"
               1 common_suffix => "worth"
               1 person_name => "jim"
               1 descriptive_word => "pink""#;

        //TODO: Add more names
        //TODO: Add more descriptive terms
        //TODO: Add geographical features
        //TODO: Add weather
        //TODO: Add colors
        //TODO: Add air_quality
        //TODO: Add royalty
        //TODO: Add religion
        //TODO: Add animals
        //TODO: Add quality (good grand messy)
        //TODO: Add jobs
        //TODO: Add seasons

        let mut ctx = EmptyContext;
        load_language(rules, &mut ctx)
    }

    #[test]
    fn test_generate_towns() {
        let mut rng = thread_rng();

        let language = towns_language_mod();
        let language = language.map_literals(|v|->Result<String,()> {Ok(format!("{}|", v))}).unwrap();
        let language = Language::from_raw(&language);
        let s1 = language.token_by_name("town").unwrap();
        for _i in 0..10 {
            let v = language.expand(&[s1], &mut rng);
            println!("{:?}", v);
        }
        assert_eq!(false, true);
    }

    #[test]
    fn test_register_token() {
        let mut language = Language::new();
        let s1 = language.add_or_get_named_symbol("a_symbol");
        assert_eq!(Some(s1), language.token_by_name("a_symbol"));
        assert_eq!(None, language.token_by_name("no_such_symbol"));
    }

    #[test]
    fn test_simplest_language() {
        let mut rng = thread_rng();

        let mut language = Language::new();
        let s1 = language.terminal("hello");
        let r = language.expand(&[s1], &mut rng);
        assert_eq!("hello", r);
    }

    #[test]
    fn test_next_simplest_language() {
        let mut rng = thread_rng();

        let mut language = Language::new();
        let hello = language.terminal("hello");
        let space = language.terminal(" ");
        let world = language.terminal("world");
        let r = language.expand(&[hello, space, world], &mut rng);
        assert_eq!("hello world", r);
        let hello_world = language.new_symbol();
        language.add_production(hello_world, 1, &[hello, space, world]);
        let r = language.expand(&[hello_world], &mut rng);
        assert_eq!("hello world", r);
    }

    #[test]
    fn test_language_can_produce_single() {
        let mut rng = thread_rng();

        let language = dummy_language();
        let language = Language::from_raw(&language);
        let s1 = language.token_by_name("tofu").unwrap();
        for _i in 0..10 {
            let v = language.expand(&[s1], &mut rng);
            println!("{:?}", v);
        }
        assert_eq!(false, true);
    }

    #[test]
    fn test_choose_by_weight() {
        let vs: Vec<(u32, &str)> = vec![(1, "a"), (2, "b"), (3, "c")];
        let mut rng = thread_rng();
        let mut counts: BTreeMap<&str, u32> = vec![("a", 0u32), ("b", 0u32), ("c", 0u32)]
            .into_iter()
            .collect();
        for _i in 0..600 {
            let v = choose_by_weight(&mut rng, &vs, &|x: &(u32, &str)| x.0).unwrap();
            *counts.get_mut(v.1).unwrap() += 1;
        }
        assert!(counts["a"] < 150, "a = {} should be < 150", counts["a"]);
        assert!(counts["a"] > 50, "a = {} should be > 50", counts["a"]);
        assert!(counts["b"] < 250, "b = {} should be < 250", counts["b"]);
        assert!(counts["b"] > 150, "b = {} should be > 150", counts["b"]);
        assert!(counts["c"] < 350, "c = {} should be < 150", counts["c"]);
        assert!(counts["c"] > 250, "c = {} should be > 50", counts["c"]);
        assert!(
            counts["a"] + counts["b"] + counts["c"] == 600,
            "a+b+c = {}+{}+{} = {} should be 600",
            counts["a"],
            counts["b"],
            counts["c"],
            counts["a"] + counts["b"] + counts["c"]
        );
    }

    #[test]
    fn tokenize_single_rule() {
        use raw::Production;
        let rule = r#"3 T => T "bar" Q"#;
        let (productions, _) = parse::parse_production(rule).unwrap();
        assert_eq!(
            productions,
            vec![Production {
                from: raw::Symbol::new("T"),
                weight: 3,
                to: vec![
                    raw::SymbolOrLiteral::symbol("T"),
                    raw::SymbolOrLiteral::literal("bar"),
                    raw::SymbolOrLiteral::symbol("Q")
                ]
            },]
        );
    }

    #[test]
    fn tokenize_alt_rule() {
        use raw::Production;
        let rule = r#"3 T => T A | Q B"#;
        let (productions, _) = parse::parse_production(rule).unwrap();

        assert_eq!(
            productions,
            vec![
                Production {
                    from: raw::Symbol::new("T"),
                    weight: 3,
                    to: vec![
                        raw::SymbolOrLiteral::symbol("T"),
                        raw::SymbolOrLiteral::symbol("A")
                    ]
                },
                Production {
                    from: raw::Symbol::new("T"),
                    weight: 3,
                    to: vec![
                        raw::SymbolOrLiteral::symbol("Q"),
                        raw::SymbolOrLiteral::symbol("B")
                    ]
                },
            ]
        );
    }

    #[test]
    fn tokenize_weight() {
        let rule = "3 XXX";
        let (weight, rest) = parse::parse_weight(rule).unwrap();
        assert_eq!(weight, 3);
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn tokenize_symbol() {
        assert_eq!(
            parse::parse_symbol("AX XXX"),
            Ok((raw::Symbol::new("AX"), "XXX"))
        );
        assert_eq!(
            parse::parse_symbol(""),
            Err(parse::ParseError::MissingSymbol)
        );
    }

    #[test]
    fn test_parse_take() {
        let rule = "XYZW";
        let (taken, rest) = parse::take("XY", rule).unwrap();
        assert_eq!(taken, "XY");
        assert_eq!(rest, "ZW");
    }

    #[test]
    fn test_take_char() {
        let rule = "XYZW";
        let (_, rest) = parse::take_char('X', rule).unwrap();
        assert_eq!(rest, "YZW");

        let e = parse::take_char('Y', rule);
        assert_eq!(e, Err(parse::ParseError::GeneralError));

        let rule = "⇄es";
        let (_, rest) = parse::take_char('⇄', rule).unwrap();
        assert_eq!(rest, "es");
    }

    #[test]
    fn test_take_while_not_char() {
        let rule = "XYZW";
        let (pre, rest) = parse::take_while_not_char('Z', rule).unwrap();
        assert_eq!(pre, "XY");
        assert_eq!(rest, "ZW");

        let (pre, rest) = parse::take_while_not_char('X', rule).unwrap();
        assert_eq!(pre, "");
        assert_eq!(rest, "XYZW");

        let (pre, rest) = parse::take_while_not_char('W', rule).unwrap();
        assert_eq!(pre, "XYZ");
        assert_eq!(rest, "W");

        let e = parse::take_while_not_char('Q', rule);
        assert_eq!(e, Err(parse::ParseError::GeneralError));
    }

    #[test]
    fn tokenize_arrow() {
        let rule = " => XXX";
        let (tag, rest) = parse::parse_tag("=>", rule).unwrap();
        assert_eq!(tag, "=>");
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn tokenize_string() {
        let rule = r#" "a string" XXX"#;
        let (s, rest) = parse::parse_string(rule).unwrap();
        assert_eq!(s, "a string");
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn parse_alt() {
        let parser = |w| {
            parse::parse_alt(
                w,
                |v| parse::parse_tag("tagA", v),
                |v| parse::parse_tag("tagB", v),
            )
        };

        assert_eq!(parser("tagA XXX"), Ok(("tagA", "XXX")));
        assert_eq!(parser("tagB XXX"), Ok(("tagB", "XXX")));
        assert_eq!(parser("tagQ XXX"), Err(parse::ParseError::GeneralError));
    }

    #[test]
    fn parse_symbol_or_literal() {
        assert_eq!(
            parse::parse_symbol_or_literal("tagA XXX"),
            Ok((raw::SymbolOrLiteral::symbol("tagA"), "XXX"))
        );
        assert_eq!(
            parse::parse_symbol_or_literal(r#" "a string"  XXX"#),
            Ok((raw::SymbolOrLiteral::literal("a string"), "XXX"))
        );
    }

    #[test]
    fn parse_symbol_or_literal_list() {
        assert_eq!(
            parse::parse_symbol_or_literal_list("tagA XXX"),
            Ok((
                vec![
                    raw::SymbolOrLiteral::symbol("tagA"),
                    raw::SymbolOrLiteral::symbol("XXX"),
                ],
                ""
            ))
        );

        assert_eq!(
            parse::parse_symbol_or_literal_list(r#" "a string"  XXX "hello" "#),
            Ok((
                vec![
                    raw::SymbolOrLiteral::literal("a string"),
                    raw::SymbolOrLiteral::symbol("XXX"),
                    raw::SymbolOrLiteral::literal("hello"),
                ],
                ""
            ))
        );
    }

    #[test]
    fn load_language_e2e() {
        let language_raw = r#"1 hello => "hello"
           1 space => " "
           1 world => "world"
           1 hw => hello space world"#;

        let mut ctx = EmptyContext;
        let language = load_language(language_raw, &mut ctx);
        let language = Language::from_raw(&language);
        let mut rng = thread_rng();
        let s = language.expand(&[language.token_by_name("hw").unwrap()], &mut rng);
        assert_eq!("hello world", s);
    }

    #[test]
    fn load_language_alternation() {
        use raw::{Production, Symbol, SymbolOrLiteral};
        let language_raw = r#"1 foo => "bar" | "baz" | "zap""#;

        let mut ctx = EmptyContext;
        let language = load_language(language_raw, &mut ctx);

        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("foo"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("bar")]
                },
                Production {
                    from: Symbol::new("foo"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("baz")]
                },
                Production {
                    from: Symbol::new("foo"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("zap")]
                },
            ]
        );
    }

    #[test]
    fn parse_symbol_fails_on_alternation() {
        let e = parse::parse_symbol("|");
        assert_eq!(
            e.map_err(|ee| ee.description()),
            Err("missing symbol".to_string())
        );
    }

    #[test]
    fn parse_symbol_with_underscore() {
        let e = parse::parse_symbol("a_b def").unwrap();
        assert_eq!(e, (raw::Symbol("a_b".to_string()), "def"));
    }

    #[test]
    fn parse_symbol_simple() {
        assert_eq!(
            parse::parse_symbol("XXX"),
            Ok((raw::Symbol::new("XXX"), ""))
        );
    }

    #[test]
    fn parse_symbol_or_literal_simple() {
        assert_eq!(
            parse::parse_symbol_or_literal("XXX"),
            Ok((raw::SymbolOrLiteral::symbol("XXX"), ""))
        );
    }

    #[test]
    fn parse_symbol_ends_at_bracket() {
        let e = parse::parse_symbol("abc(def").unwrap();
        assert_eq!(e, (raw::Symbol("abc".to_string()), "(def"));
    }

    #[test]
    fn test_parse_directive() {
        let directive_raw = r#"@import_list("city_types.txt" settlement_type)"#;
        let (directive, rest) = parse::parse_directive(directive_raw).unwrap();
        assert_eq!(
            parse::Directive {
                name: "import_list".to_string(),
                arguments: vec![
                    raw::SymbolOrLiteral::literal("city_types.txt"),
                    raw::SymbolOrLiteral::symbol("settlement_type")
                ]
            },
            directive
        );
    }

    #[test]
    fn test_parse_empty_directive() {
        let directive_raw = r#"@import_list()"#;
        let (directive, rest) = parse::parse_directive(directive_raw).unwrap();
        assert_eq!("", rest);
        assert_eq!(
            parse::Directive {
                name: "import_list".to_string(),
                arguments: vec![]
            },
            directive
        );
    }

    struct MockContext {
        word_lists: BTreeMap<String, Vec<String>>,
        languages: BTreeMap<String, raw::Language>,
    }

    impl Context for MockContext {
        fn get_word_list(&self, name: &str) -> Result<Vec<String>, ContextError> {
            match self.word_lists.get(name) {
                Some(v) => Ok(v.clone()),
                None => Err(ContextError::InvalidKey),
            }
        }
        fn get_language(&self, name: &str) -> Result<raw::Language, ContextError> {
            match self.languages.get(name) {
                Some(v) => Ok(v.clone()),
                None => Err(ContextError::InvalidKey),
            }
        }
    }

    #[test]
    fn test_parse_language_with_import_list_directive() {
        use raw::{Production, Symbol, SymbolOrLiteral};

        let mut word_lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
        word_lists.insert("Q.txt".to_string(), vec!["Q".to_string(), "R".to_string()]);
        let mut ctx = MockContext {
            word_lists,
            languages: BTreeMap::new(),
        };

        let language_raw = r#"1 A => "A"
            @import_list("Q.txt" Q)"#;
        let language = load_language(language_raw, &mut ctx);
        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("A"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("R")]
                },
            ]
        );
    }

    #[test]
    fn test_parse_language_with_import_language_directive() {
        use raw::{Language, Production, Symbol, SymbolOrLiteral};

        let mut languages: BTreeMap<String, Language> = BTreeMap::new();
        let mut l = Language::new();
        l.entries.push(Production {
            from: Symbol::new("A"),
            weight: 2,
            to: vec![SymbolOrLiteral::literal("Q")],
        });
        l.entries.push(Production {
            from: Symbol::new("Q"),
            weight: 1,
            to: vec![SymbolOrLiteral::symbol("A")],
        });

        languages.insert("Q.lang".to_string(), l);
        let mut ctx = MockContext {
            word_lists: BTreeMap::new(),
            languages,
        };

        let language_raw = r#"1 A => "A"
            @import_language("Q.lang")"#;
        let language = load_language(language_raw, &mut ctx);
        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("A"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                Production {
                    from: Symbol::new("A"),
                    weight: 2,
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: 1,
                    to: vec![SymbolOrLiteral::symbol("A")]
                },
            ]
        );
    }

    /*
    #[test]
    fn test_expand_dummy() {
        let mut language = dummy_language();
        let mut result:[Symbol; 4] = [Symbol::Terminal(Terminal(1)), Symbol::Empty, Symbol::Empty, Symbol::Empty];
        language.expand_one(&result, 0, 1);
        let mut result:[Symbol ; 32] = [Symbol::Empty; 32];
        let r = language.produce(&mut result);
        assert_eq!(r, &[Symbol::Terminal(Terminal(1))]);

        language.t[0].results = [Symbol::Terminal(Terminal(2)), Symbol::Empty, Symbol::Empty, Symbol::Empty];
        let r = language.produce(&mut result);
        assert_eq!(r, &[Symbol::Terminal(Terminal(2))]);
    }
    */
}
