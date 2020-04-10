use rand::Rng;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use super::parse;
use super::raw;
use raw::nf32;

#[derive(Debug, Ord, Eq, PartialEq, PartialOrd, Copy, Clone, Default)]
pub struct SymbolId(u32);

#[derive(Debug, Clone)]
pub struct Production {
    weight: nf32,
    keys: Vec<SymbolId>,
}

impl Production {
    pub fn new(weight: nf32, keys: &[SymbolId]) -> Production {
        Production {
            weight,
            keys: keys.to_vec(),
        }
    }
}

pub trait SimpleRNG {
    fn gen_range(&mut self, low: f32, high: f32) -> f32;
}

impl<T> SimpleRNG for T
where
    T: Rng,
{
    fn gen_range(&mut self, low: f32, high: f32) -> f32 {
        Rng::gen_range(self, low, high)
    }
}

pub fn choose_by_weight<'a, R: SimpleRNG, T, F>(
    rng: &mut R,
    values: &'a [T],
    weight_fn: &F,
) -> Option<&'a T>
where
    R: SimpleRNG,
    F: Fn(&T) -> f32,
{
    let sum_w: f32 = values.iter().map(weight_fn).sum();
    if sum_w <= 0.0 {
        return None;
    }
    let r = rng.gen_range(0.0, sum_w);
    let mut s: f32 = 0.0;
    for v in values {
        s += weight_fn(v);
        if s >= r {
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

//TODO: These really do indicate behaviour that should be caught at the point at which a raw
// Language is converted into a compiled language. And so could become asserts.
// In the compile stage both really corrspond to the case where a Symbol appears in the from part
// of a production, but doesn't have any corresponding productions.SymbolId
#[derive(Debug, Eq, PartialEq)]
pub enum ExpansionError {
    InvalidSymbolId(SymbolId),
    MissingExpansion(SymbolId),
}

#[derive(Debug, Eq, PartialEq)]
pub enum ConversionError {
    GeneralError,
    MissingExpansions(BTreeSet<String>),
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
    fn new() -> Language {
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

    pub fn add_production(&mut self, symbol_id: SymbolId, weight: nf32, keys: &[SymbolId]) {
        self.productions_by_id
            .entry(symbol_id)
            .or_insert_with(ProductionGroup::new)
            .add(Production::new(weight, keys))
    }

    pub fn expand<R: SimpleRNG>(
        &self,
        tokens: &[SymbolId],
        rng: &mut R,
    ) -> Result<String, ExpansionError> {
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
                //TODO: Differentiate between these errors and give a better message
                let pg = self
                    .productions_by_id
                    .get(&token)
                    .ok_or_else(|| ExpansionError::InvalidSymbolId(token))?;
                let p = choose_by_weight(rng, &pg.productions, &|x: &Production| x.weight.0)
                    .ok_or_else(|| ExpansionError::MissingExpansion(token))?;
                expansion_stack.push(&p.keys);
            }
        }
        Ok(complete)
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

    pub fn from_raw(raw: &raw::Language<String>) -> Result<Self, ConversionError> {
        // Check that language is sane
        // At the moment that just means having all symbols
        // correspond to a production with non-zero weight
        // TODO: Probably should have raw::Language::from_symbols()
        //       and to_symbols() and derive what we need to actually
        //       use BTreeSet<raw::Symbol>.
        // TODO: Probably can hoist all of this check into raw::Language
        //       so that this becomes `raw.check_complete()?` or similar.
        {
            let mut from_symbols: BTreeSet<String> = BTreeSet::new();
            let mut to_symbols: BTreeSet<String> = BTreeSet::new();

            for p in &raw.entries {
                if p.weight.0 == 0.0 {
                    continue;
                }
                from_symbols.insert(p.from.0.clone());
                for s in p.to.iter().filter_map(|s| s.as_symbol()) {
                    to_symbols.insert(s.0.clone());
                }
            }
            if !to_symbols.is_subset(&from_symbols) {
                return Err(ConversionError::MissingExpansions(
                    &to_symbols - &from_symbols,
                ));
            }
        }

        let mut result = Language::new();
        for p in &raw.entries {
            if p.weight.0 == 0.0 {
                continue;
            }
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
        Ok(result)
    }

    pub fn to_raw(&self) -> Result<raw::Language<String>, ConversionError> {
        let x: Vec<(String, &ProductionGroup)> = self
            .productions_by_id
            .iter()
            .map(|(k, v)| (self.format_symbol(*k), v))
            .collect();
        let y: Vec<(String, Production)> = x
            .iter()
            .flat_map(|(k, g)| g.productions.iter().map(move |v| (k.clone(), v.clone())))
            .collect();
        let z: Vec<(String, nf32, Vec<SymbolId>)> = y
            .iter()
            .map(|(k, v)| (k.clone(), v.weight, v.keys.clone()))
            .collect();
        let mut entries: Vec<raw::Production<String>> = vec![];
        for e in z {
            let from = raw::Symbol(e.0);
            let weight = e.1;
            let to: Result<Vec<_>, ConversionError> =
                e.2.into_iter()
                    .map(|vid| {
                        let s = self
                            .symbols_by_name
                            .iter()
                            .find(|(_k, id)| **id == vid)
                            .map(|(k, _id)| k);
                        if let Some(s) = s {
                            return Ok(raw::SymbolOrLiteral::symbol(s));
                        }
                        let s = self
                            .terminals_by_id
                            .get(&vid)
                            .cloned()
                            .map(raw::SymbolOrLiteral::literal);
                        if let Some(s) = s {
                            return Ok(s);
                        }
                        Err(ConversionError::GeneralError)
                    })
                    .collect();
            let to = to?;
            entries.push(raw::Production { from, weight, to })
        }
        Ok(raw::Language { entries })
    }
}

#[derive(Debug)]
pub enum ContextError {
    InvalidOperation,
    InvalidKey,
}

pub trait Context {
    fn get_word_list(&self, name: &str) -> Result<Vec<String>, ContextError>;
    fn get_language(&self, name: &str) -> Result<raw::Language<String>, ContextError>;
}

pub struct EmptyContext;
impl Context for EmptyContext {
    fn get_word_list(&self, _name: &str) -> Result<Vec<String>, ContextError> {
        Err(ContextError::InvalidOperation)
    }
    fn get_language(&self, _name: &str) -> Result<raw::Language<String>, ContextError> {
        Err(ContextError::InvalidOperation)
    }
}

//TODO: Create more informative error types?
#[derive(Debug, PartialEq, Eq)]
pub enum DirectiveError {
    GeneralError,
}

pub fn apply_directive(
    language: &mut raw::Language<String>,
    directive: &parse::Directive,
    ctx: &mut dyn Context,
) -> Result<(), DirectiveError> {
    println!("Applying directive : {:?}", directive);
    match &directive.name[..] {
        // import_list( "Name.txt" Symbol )
        "import_list" => {
            println!("args = {:?}", directive.arguments);
            if directive.arguments.len() != 2 {
                return Err(DirectiveError::GeneralError);
            }
            let name = directive.arguments[0]
                .as_literal()
                .ok_or_else(|| DirectiveError::GeneralError)?;
            let from = raw::Symbol(
                directive.arguments[1]
                    .as_symbol()
                    .ok_or_else(|| DirectiveError::GeneralError)?
                    .0
                    .clone(),
            );
            for v in ctx
                .get_word_list(&name.0)
                .map_err(|_e| DirectiveError::GeneralError)?
            {
                language.entries.push(raw::Production {
                    from: from.clone(),
                    weight: nf32(1.0),
                    to: vec![raw::SymbolOrLiteral::literal(v)],
                });
            }
            Ok(())
        }
        // import_language( "Foo.lang" )
        "import_language" => {
            // TODO we should support other modes rather than import everything into the
            //      root namespace.
            if directive.arguments.len() != 1 {
                return Err(DirectiveError::GeneralError);
            }
            let name = directive.arguments[0]
                .as_literal()
                .ok_or_else(|| DirectiveError::GeneralError)?;
            let l: raw::Language<String> = ctx
                .get_language(&name.0)
                .map_err(|_e| DirectiveError::GeneralError)?;
            for e in l.entries {
                language.entries.push(e.clone());
            }
            Ok(())
        }
        _ => {
            println!("Unknown directive: {:?}", directive.name);
            Err(DirectiveError::GeneralError)
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoadLanguageError {
    GeneralError,
    //TODO: Add more information to this type
    InvalidLine,
    DirectiveError(DirectiveError),
}

pub fn load_language(
    language_raw: &str,
    ctx: &mut dyn Context,
) -> Result<raw::Language<String>, LoadLanguageError> {
    let mut language = raw::Language::new();
    for line in language_raw.lines() {
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
                apply_directive(&mut language, &d, ctx)
                    .map_err(LoadLanguageError::DirectiveError)?;
            }
        }
    }
    Ok(language)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn dummy_language() -> raw::Language<String> {
        let rules = r#"2 tofu => "tofu"
               1 tofu => tofu " " tofu
               3 tofu => "I like to eat " tofu"#;

        let mut ctx = EmptyContext;
        load_language(rules, &mut ctx).unwrap()
    }

    fn towns_language_mod() -> raw::Language<String> {
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

        load_language(rules, &mut ctx).unwrap()
    }

    fn towns_language() -> raw::Language<String> {
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
        load_language(rules, &mut ctx).unwrap()
    }

    #[test]
    fn test_generate_towns() {
        let mut rng = thread_rng();

        let language = towns_language_mod();
        let language = language
            .map_literals(|v| -> Result<String, ()> { Ok(format!("{}|", v)) })
            .unwrap();
        let language = Language::from_raw(&language).unwrap();
        let s1 = language.token_by_name("town").unwrap();
        for _i in 0..10 {
            let v = language.expand(&[s1], &mut rng);
            println!("{:?}", v);
        }
        assert_eq!(false, true);
    }

    fn prod_symbols(from: &str, to: &[&str]) -> raw::Production<String> {
        raw::Production {
            from: raw::Symbol::new(from),
            weight: nf32(1.0),
            to: to
                .iter()
                .map(|s| raw::SymbolOrLiteral::symbol(*s))
                .collect(),
        }
    }

    fn prod_literals(from: &str, to: &[&str]) -> raw::Production<String> {
        raw::Production {
            from: raw::Symbol::new(from),
            weight: nf32(1.0),
            to: to
                .iter()
                .map(|s| raw::SymbolOrLiteral::literal(*s))
                .collect(),
        }
    }

    #[test]
    fn test_register_token() {
        let raw = raw::Language {
            entries: vec![prod_symbols("a_symbol", &[])],
        };
        let language = Language::from_raw(&raw).unwrap();
        assert!(language.token_by_name("a_symbol").is_some());
        assert_eq!(None, language.token_by_name("no_such_symbol"));
    }

    #[test]
    fn test_simplest_language() {
        let mut rng = thread_rng();
        let raw = raw::Language {
            entries: vec![prod_literals("A", &["hello"])],
        };
        let language = Language::from_raw(&raw).unwrap();
        let s1 = language.token_by_name("A").unwrap();
        let r = language.expand(&[s1], &mut rng).unwrap();
        assert_eq!("hello", r);
    }

    #[test]
    fn test_next_simplest_language() {
        let mut rng = thread_rng();
        let raw = raw::Language {
            entries: vec![
                prod_symbols("A", &["hello", "space", "world"]),
                prod_literals("hello", &["hello"]),
                prod_literals("space", &[" "]),
                prod_literals("world", &["world"]),
            ],
        };
        let language = Language::from_raw(&raw).unwrap();
        let a = language.token_by_name("A").unwrap();
        let r = language.expand(&[a], &mut rng).unwrap();
        assert_eq!("hello world", r);
    }

    #[test]
    fn test_language_can_produce_single() {
        let mut rng = thread_rng();

        let language = dummy_language();
        let language = Language::from_raw(&language).unwrap();
        let s1 = language.token_by_name("tofu").unwrap();
        for _i in 0..10 {
            let v = language.expand(&[s1], &mut rng).unwrap();
            println!("{:?}", v);
        }
        assert_eq!(false, true);
    }

    #[test]
    fn test_choose_by_weight() {
        let vs: Vec<(f32, &str)> = vec![(1.0, "a"), (2.0, "b"), (3.0, "c")];
        let mut rng = thread_rng();
        let mut counts: BTreeMap<&str, u32> = vec![("a", 0u32), ("b", 0u32), ("c", 0u32)]
            .into_iter()
            .collect();
        for _i in 0..600 {
            let v = choose_by_weight(&mut rng, &vs, &|x: &(f32, &str)| x.0).unwrap();
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
    fn load_language_e2e() {
        let language_raw = r#"1 hello => "hello"
           1 space => " "
           1 world => "world"
           1 hw => hello space world"#;

        let mut ctx = EmptyContext;
        let language = load_language(language_raw, &mut ctx).unwrap();
        let language = Language::from_raw(&language).unwrap();
        let mut rng = thread_rng();
        let s = language
            .expand(&[language.token_by_name("hw").unwrap()], &mut rng)
            .unwrap();
        assert_eq!("hello world", s);
    }

    #[test]
    fn load_language_alternation() {
        use raw::{Production, Symbol, SymbolOrLiteral};
        let language_raw = r#"1 foo => "bar" | "baz" | "zap""#;

        let mut ctx = EmptyContext;
        let language = load_language(language_raw, &mut ctx).unwrap();

        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("foo"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("bar")]
                },
                Production {
                    from: Symbol::new("foo"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("baz")]
                },
                Production {
                    from: Symbol::new("foo"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("zap")]
                },
            ]
        );
    }

    struct MockContext {
        word_lists: BTreeMap<String, Vec<String>>,
        languages: BTreeMap<String, raw::Language<String>>,
    }

    impl Context for MockContext {
        fn get_word_list(&self, name: &str) -> Result<Vec<String>, ContextError> {
            match self.word_lists.get(name) {
                Some(v) => Ok(v.clone()),
                None => Err(ContextError::InvalidKey),
            }
        }
        fn get_language(&self, name: &str) -> Result<raw::Language<String>, ContextError> {
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
        let language = load_language(language_raw, &mut ctx).unwrap();
        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("A"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("R")]
                },
            ]
        );
    }

    #[test]
    fn test_parse_language_with_import_language_directive() {
        use raw::{Language, Production, Symbol, SymbolOrLiteral};

        let mut languages: BTreeMap<String, Language<String>> = BTreeMap::new();
        let mut l = Language::new();
        l.entries.push(Production {
            from: Symbol::new("A"),
            weight: nf32(2.0),
            to: vec![SymbolOrLiteral::literal("Q")],
        });
        l.entries.push(Production {
            from: Symbol::new("Q"),
            weight: nf32(1.0),
            to: vec![SymbolOrLiteral::symbol("A")],
        });

        languages.insert("Q.lang".to_string(), l);
        let mut ctx = MockContext {
            word_lists: BTreeMap::new(),
            languages,
        };

        let language_raw = r#"1 A => "A"
            @import_language("Q.lang")"#;
        let language = load_language(language_raw, &mut ctx).unwrap();
        assert_eq!(
            language.entries,
            vec![
                Production {
                    from: Symbol::new("A"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                Production {
                    from: Symbol::new("A"),
                    weight: nf32(2.0),
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                Production {
                    from: Symbol::new("Q"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::symbol("A")]
                },
            ]
        );
    }

    #[test]
    pub fn test_language_from_to_raw_ping_pong() {
        let initial = raw::Language {
            entries: vec![
                raw::Production {
                    from: raw::Symbol::new("A"),
                    weight: nf32(1.0),
                    to: vec![
                        raw::SymbolOrLiteral::symbol("B"),
                        raw::SymbolOrLiteral::literal("X"),
                    ],
                },
                raw::Production {
                    from: raw::Symbol::new("B"),
                    weight: nf32(1.0),
                    to: vec![raw::SymbolOrLiteral::literal("Y")],
                },
            ],
        };
        let compiled = Language::from_raw(&initial).unwrap();
        let reverted = compiled.to_raw().unwrap();
        assert_eq!(initial.entries, reverted.entries);
    }
}
