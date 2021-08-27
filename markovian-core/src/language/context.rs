use std::collections::HashMap;
use std::path::PathBuf;

use snafu::{Backtrace, ResultExt, Snafu};

use super::raw::{self, Context, LoadLanguageError};
use crate::language::raw::Language;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum ContextError {
    #[snafu(display("Invalid operation '{}'", op_name))]
    InvalidOperation {
        op_name: String,
        backtrace: Backtrace,
    },

    #[snafu(display("Invalid key: {}", mesg))]
    InvalidKey { mesg: String, backtrace: Backtrace },

    #[snafu(display("The language '{}' is invalid", lang_name))]
    InvalidLanguage {
        lang_name: String,
        source: LoadLanguageError,
    },

    #[snafu(display("recursion detected when loading '{}'", lang_name))]
    LanguageRecursionDetected { lang_name: String },
}

pub struct GrammarLoaderContext {
    path: PathBuf,
    loaded_word_lists: HashMap<String, Vec<String>>,

    /// Entry is set to None while the language is loading,
    /// then set to the right value once loaded, to avoid recursive calls
    loaded_languages: HashMap<String, Option<Language<String>>>,
}

impl GrammarLoaderContext {
    pub fn new(path: PathBuf) -> Self {
        GrammarLoaderContext {
            path,
            loaded_word_lists: HashMap::new(),
            loaded_languages: HashMap::new(),
        }
    }
}

impl Context for GrammarLoaderContext {
    fn get_word_list(&mut self, name: &str) -> Result<Vec<String>, ContextError> {
        let v = self.loaded_word_lists.get(name);
        if let Some(v) = v {
            return Ok(v.clone());
        }

        let list_path = self.path.join(name);
        let content = std::fs::read_to_string(&list_path).map_err(|e| {
            InvalidKey {
                mesg: format!(
                    "Unable to load word list for {}: {}",
                    list_path.to_string_lossy(),
                    e
                ),
            }
            .build()
        })?;
        let v: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        self.loaded_word_lists.insert(String::from(name), v.clone());
        Ok(v)
    }

    fn get_language(&mut self, name: &str) -> Result<Language<String>, ContextError> {
        let v = self.loaded_languages.get(name);
        if let Some(v) = v {
            return match v {
                Some(v) => Ok(v.clone()),
                None => Err(LanguageRecursionDetected { lang_name: name }.build()),
            };
        }

        self.loaded_languages.insert(String::from(name), None);

        let file_path = self.path.join(&name);
        let language_raw = std::fs::read_to_string(&file_path).map_err(|e| {
            InvalidKey {
                mesg: format!(
                    "Unable to load language file for for {}: {}",
                    file_path.to_string_lossy(),
                    e
                ),
            }
            .build()
        })?;
        let language =
            raw::load_language(&language_raw, self).context(InvalidLanguage { lang_name: name })?;
        self.loaded_languages
            .insert(String::from(name), Some(language.clone()));
        Ok(language)
    }
}

#[cfg(test)]
pub mod test {
    use std::path::PathBuf;

    use super::*;
    use crate::language::raw::{Context, Production, Symbol, SymbolOrLiteral};
    use crate::utils::nf32::nf32;

    fn s(s: &str) -> SymbolOrLiteral<String> {
        SymbolOrLiteral::symbol(String::from(s))
    }

    fn l(s: &str) -> SymbolOrLiteral<String> {
        SymbolOrLiteral::literal(String::from(s))
    }

    fn p(n: &str, v: SymbolOrLiteral<String>) -> Production<String> {
        Production {
            from: Symbol::new(String::from(n)),
            weight: nf32(1.0),
            to: vec![v],
        }
    }

    #[test]
    pub fn get_language_works() {
        let mut ctx =
            GrammarLoaderContext::new(PathBuf::from("test_resources/language/sub_language_01"));
        let lang = ctx.get_language("main.lang").unwrap();

        assert_eq!(
            lang.entries,
            vec![p("A", l("A")), p("A", s("B")), p("B", l("B")),]
        );
    }

    pub fn get_err_recursive(v: &dyn std::error::Error) -> Vec<String> {
        let mut e = Some(v);
        let mut result = vec![];
        while let Some(ee) = e {
            let r = ee.to_string();
            // Some errors have more detail nested inside them,
            // we may as well just skip those
            if r != "invalid directive" && r != "context error" {
                result.push(ee.to_string());
            }
            e = ee.source();
        }
        result
    }

    #[test]
    pub fn errors_on_self_recursive() {
        let mut ctx = GrammarLoaderContext::new(PathBuf::from(
            "test_resources/language/invalid_self_recursive",
        ));
        let e = ctx.get_language("main.lang");
        let e = if let Err(e) = e { e } else { unreachable!() };
        assert_eq!(
            vec![
                "The language 'main.lang' is invalid".to_string(),
                "recursion detected when loading 'main.lang'".to_string(),
            ],
            get_err_recursive(&e),
        );
    }

    #[test]
    pub fn errors_on_other_recursive() {
        let mut ctx = GrammarLoaderContext::new(PathBuf::from(
            "test_resources/language/invalid_other_recursive",
        ));
        let e = ctx.get_language("main.lang");
        let e = if let Err(e) = e { e } else { unreachable!() };
        assert_eq!(
            vec![
                "The language 'main.lang' is invalid".to_string(),
                "The language 'child.lang' is invalid".to_string(),
                "recursion detected when loading 'main.lang'".to_string(),
            ],
            get_err_recursive(&e)
        );
    }
}
