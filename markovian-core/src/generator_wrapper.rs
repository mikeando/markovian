use std::borrow::Borrow;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::generator::{GenerationError, Generator};
use crate::renderer::{RenderChar, RenderU8};
use crate::symbol::SymbolTableEntryId;
use crate::symboltable_wrapper::SymbolTableWrapper;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorWrapper {
    Bytes(Generator<u8, f32>),
    String(Generator<char, f32>),
}

impl GeneratorWrapper {
    pub fn log_prob(&self, word: &str, katz_coefficient: Option<f32>) -> f32 {
        match &self {
            GeneratorWrapper::Bytes(gen) => gen.log_prob(word.as_bytes(), katz_coefficient),
            GeneratorWrapper::String(gen) => {
                gen.log_prob(&word.chars().collect::<Vec<_>>(), katz_coefficient)
            }
        }
    }
}

impl GeneratorWrapper {
    pub fn into_symbol_table_and_ngrams(
        self,
    ) -> (SymbolTableWrapper, BTreeMap<Vec<SymbolTableEntryId>, f32>) {
        match self {
            GeneratorWrapper::Bytes(gen) => {
                let (st, ngrams) = gen.into_symbol_table_and_ngrams();
                (SymbolTableWrapper::Bytes(st), ngrams)
            }
            GeneratorWrapper::String(gen) => {
                let (st, ngrams) = gen.into_symbol_table_and_ngrams();
                (SymbolTableWrapper::String(st), ngrams)
            }
        }
    }

    pub fn from_ngrams(
        symbol_table: SymbolTableWrapper,
        ngrams: BTreeMap<Vec<SymbolTableEntryId>, f32>,
    ) -> GeneratorWrapper {
        match symbol_table {
            SymbolTableWrapper::Bytes(st) => {
                GeneratorWrapper::Bytes(Generator::from_ngrams(st, ngrams))
            }
            SymbolTableWrapper::String(st) => {
                GeneratorWrapper::String(Generator::from_ngrams(st, ngrams))
            }
        }
    }
}

//TODO: Move this into GeneratorWrapper?
pub fn generate_words(
    generator: &GeneratorWrapper,
    count: usize,
    prefix: &Option<String>,
    suffix: &Option<String>,
    katz_coefficient: Option<f32>,
) -> Result<Vec<String>, GenerationError> {
    let mut rng = rand::thread_rng();

    match generator {
        GeneratorWrapper::Bytes(gen) => {
            let renderer = RenderU8 {
                table: &gen.symbol_table,
                start: b"^",
                end: b"$",
            };
            let prefix = prefix.as_ref().map(|s| s.as_bytes());
            let suffix = suffix.as_ref().map(|s| s.as_bytes());
            gen.generate_multi(prefix, suffix, count, katz_coefficient, &mut rng, &renderer)
        }
        GeneratorWrapper::String(gen) => {
            let renderer = RenderChar {
                table: &gen.symbol_table,
                start: "^",
                end: "$",
            };
            let prefix = prefix.as_ref().map(|s| s.chars().collect::<Vec<_>>());
            let suffix = suffix.as_ref().map(|s| s.chars().collect::<Vec<_>>());
            let p_temp = prefix.as_ref().map(|s| s.borrow());
            let s_temp = suffix.as_ref().map(|s| s.borrow());

            gen.generate_multi(p_temp, s_temp, count, katz_coefficient, &mut rng, &renderer)
        }
    }
}
