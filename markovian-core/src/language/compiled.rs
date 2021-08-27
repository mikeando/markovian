use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;
use snafu::{OptionExt, Snafu};

use super::raw;
use crate::utils::nf32::nf32;

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
        Rng::gen_range(self, low..high)
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
#[derive(Debug, Snafu, Eq, PartialEq)]
pub enum ExpansionError {
    #[snafu(display("invalid symbol id={}", id.0))]
    InvalidSymbolId { id: SymbolId },

    #[snafu(display("missing expansion for symbol id={}", id.0))]
    MissingExpansion { id: SymbolId },
}

#[derive(Debug, Snafu, Eq, PartialEq)]
pub enum ConversionError {
    #[snafu(display("general conversion error"))]
    GeneralError,
    #[snafu(display("missing expansions {:?}", missing))]
    MissingExpansions { missing: BTreeSet<String> },
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
                    .with_context(|| InvalidSymbolId { id: token })?;
                let p = choose_by_weight(rng, &pg.productions, &|x: &Production| x.weight.0)
                    .with_context(|| MissingExpansion { id: token })?;
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
                return Err(MissingExpansions {
                    missing: &to_symbols - &from_symbols,
                }
                .build());
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

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;
    use crate::language::context::{ContextError, InvalidKey};
    use crate::language::raw::{Context, EmptyContext};

    fn dummy_language() -> raw::Language<String> {
        let rules = r#"2 tofu => "tofu"
               1 tofu => tofu " " tofu
               3 tofu => "I like to eat " tofu"#;

        let mut ctx = EmptyContext;
        raw::load_language(rules, &mut ctx).unwrap()
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

        let rules = include_str!("../../test_resources/town_01.lang");
        raw::load_language(rules, &mut ctx).unwrap()
    }

    #[test]
    fn test_generate_towns_doesnt_explode() {
        let mut rng = thread_rng();

        let language = towns_language_mod();
        let language = language
            .map_literals(|v| -> Result<String, ()> { Ok(format!("{}|", v)) })
            .unwrap();
        let language = Language::from_raw(&language).unwrap();
        let s1 = language.token_by_name("town").unwrap();
        for _i in 0..10 {
            let _v = language.expand(&[s1], &mut rng).unwrap();
        }
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
            assert!(!v.is_empty());
        }
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
        let language = raw::load_language(language_raw, &mut ctx).unwrap();
        let language = Language::from_raw(&language).unwrap();
        let mut rng = thread_rng();
        let s = language
            .expand(&[language.token_by_name("hw").unwrap()], &mut rng)
            .unwrap();
        assert_eq!("hello world", s);
    }

    #[test]
    fn load_language_alternation() {
        use raw::{Symbol, SymbolOrLiteral};
        let language_raw = r#"1 foo => "bar" | "baz" | "zap""#;

        let mut ctx = EmptyContext;
        let language = raw::load_language(language_raw, &mut ctx).unwrap();

        assert_eq!(
            language.entries,
            vec![
                raw::Production {
                    from: Symbol::new("foo"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("bar")]
                },
                raw::Production {
                    from: Symbol::new("foo"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("baz")]
                },
                raw::Production {
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
        fn get_word_list(&mut self, name: &str) -> Result<Vec<String>, ContextError> {
            match self.word_lists.get(name) {
                Some(v) => Ok(v.clone()),
                None => Err(InvalidKey {
                    mesg: format!("word list '{}' not found in MockContext", name),
                }
                .build()),
            }
        }
        fn get_language(&mut self, name: &str) -> Result<raw::Language<String>, ContextError> {
            match self.languages.get(name) {
                Some(v) => Ok(v.clone()),
                None => Err(InvalidKey {
                    mesg: format!("language '{}' not found in MockContext", name),
                }
                .build()),
            }
        }
    }

    #[test]
    fn test_parse_language_with_import_list_directive() {
        use raw::{Symbol, SymbolOrLiteral};

        let mut word_lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
        word_lists.insert("Q.txt".to_string(), vec!["Q".to_string(), "R".to_string()]);
        let mut ctx = MockContext {
            word_lists,
            languages: BTreeMap::new(),
        };

        let language_raw = r#"1 A => "A"
            @import_list("Q.txt" Q)"#;
        let language = raw::load_language(language_raw, &mut ctx).unwrap();
        assert_eq!(
            language.entries,
            vec![
                raw::Production {
                    from: Symbol::new("A"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                raw::Production {
                    from: Symbol::new("Q"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                raw::Production {
                    from: Symbol::new("Q"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("R")]
                },
            ]
        );
    }

    #[test]
    fn test_parse_language_with_import_language_directive() {
        use raw::{Language, Symbol, SymbolOrLiteral};

        let mut languages: BTreeMap<String, Language<String>> = BTreeMap::new();
        let mut l = Language::new();
        l.entries.push(raw::Production {
            from: Symbol::new("A"),
            weight: nf32(2.0),
            to: vec![SymbolOrLiteral::literal("Q")],
        });
        l.entries.push(raw::Production {
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
        let language = raw::load_language(language_raw, &mut ctx).unwrap();
        assert_eq!(
            language.entries,
            vec![
                raw::Production {
                    from: Symbol::new("A"),
                    weight: nf32(1.0),
                    to: vec![SymbolOrLiteral::literal("A")]
                },
                raw::Production {
                    from: Symbol::new("A"),
                    weight: nf32(2.0),
                    to: vec![SymbolOrLiteral::literal("Q")]
                },
                raw::Production {
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
