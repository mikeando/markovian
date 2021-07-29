use serde::{Deserialize, Serialize};

use crate::generator::{InverseSquareOfLengthWeighter, ShortestOnlyWeighter, ToSymbolsAndWeights};
use crate::symbol::{SymbolRemapper, SymbolTable, SymbolTableEntryId, TableEncoding};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolTableWrapper {
    Bytes(SymbolTable<u8>),
    String(SymbolTable<char>),
}

impl SymbolTableWrapper {
    pub fn encoding(&self) -> TableEncoding {
        match self {
            SymbolTableWrapper::Bytes(_) => TableEncoding::Bytes,
            SymbolTableWrapper::String(_) => TableEncoding::String,
        }
    }

    pub fn max_symbol_id(&self) -> usize {
        match self {
            SymbolTableWrapper::Bytes(table) => table.max_symbol_id(),
            SymbolTableWrapper::String(table) => table.max_symbol_id(),
        }
    }

    pub fn symbolifications(&self, s: &str) -> Vec<Vec<SymbolTableEntryId>> {
        match self {
            SymbolTableWrapper::Bytes(table) => table.symbolifications_str(s),
            SymbolTableWrapper::String(table) => {
                table.symbolifications(&s.chars().collect::<Vec<_>>())
            }
        }
    }

    pub fn get_word_breaker<'a>(
        &'a self,
        mode: &SymbolificationWeightingMode,
    ) -> Box<dyn XToSymbolsAndWeights + 'a> {
        match (self, mode) {
            (SymbolTableWrapper::Bytes(st), SymbolificationWeightingMode::ShortestOnly) => {
                Box::new(ViaBytes {
                    w: ShortestOnlyWeighter::new(st),
                })
            }
            (
                SymbolTableWrapper::Bytes(st),
                SymbolificationWeightingMode::InverseSquareOfLength,
            ) => Box::new(ViaBytes {
                w: InverseSquareOfLengthWeighter::new(st),
            }),
            (SymbolTableWrapper::String(st), SymbolificationWeightingMode::ShortestOnly) => {
                Box::new(ViaChars {
                    w: ShortestOnlyWeighter::new(st),
                })
            }
            (
                SymbolTableWrapper::String(st),
                SymbolificationWeightingMode::InverseSquareOfLength,
            ) => Box::new(ViaChars {
                w: InverseSquareOfLengthWeighter::new(st),
            }),
        }
    }

    pub fn add_symbols(&mut self, symbols: &[String]) {
        match self {
            SymbolTableWrapper::Bytes(st) => {
                let ss: Vec<Vec<u8>> = symbols
                    .iter()
                    .map(|s| s.as_bytes().iter().copied().collect())
                    .collect();
                st.add_symbols(ss).unwrap();
            }
            SymbolTableWrapper::String(st) => {
                let ss: Vec<Vec<char>> = symbols.iter().map(|s| s.chars().collect()).collect();
                st.add_symbols(ss).unwrap();
            }
        }
    }

    pub fn remove_symbols_and_compress(&mut self, symbols: &[String]) -> SymbolRemapper {
        match self {
            SymbolTableWrapper::Bytes(st) => {
                let ss: Vec<Vec<u8>> = symbols
                    .iter()
                    .map(|s| s.as_bytes().iter().copied().collect())
                    .collect();
                st.remove_symbols_and_compress(ss).unwrap()
            }
            SymbolTableWrapper::String(st) => {
                let ss: Vec<Vec<char>> = symbols.iter().map(|s| s.chars().collect()).collect();
                st.remove_symbols_and_compress(ss).unwrap()
            }
        }
    }
}

pub fn to_symbolifications_and_weights(
    symboltable: &SymbolTableWrapper,
    breaker_mode: &SymbolificationWeightingMode,
    input_tokens: &[String],
) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
    let breaker = symboltable.get_word_breaker(breaker_mode);

    //Symbolify them
    input_tokens
        .iter()
        .flat_map(|s| breaker.to_symbols_and_weights(s))
        .collect()
}

pub trait XToSymbolsAndWeights {
    fn to_symbols_and_weights(&self, v: &str) -> Vec<(Vec<SymbolTableEntryId>, f32)>;
}

struct ViaBytes<W>
where
    W: ToSymbolsAndWeights<u8>,
{
    w: W,
}

struct ViaChars<W>
where
    W: ToSymbolsAndWeights<char>,
{
    w: W,
}

impl<W> XToSymbolsAndWeights for ViaBytes<W>
where
    W: ToSymbolsAndWeights<u8>,
{
    fn to_symbols_and_weights(&self, v: &str) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
        self.w.to_symbols_and_weights(v.as_bytes())
    }
}

impl<W> XToSymbolsAndWeights for ViaChars<W>
where
    W: ToSymbolsAndWeights<char>,
{
    fn to_symbols_and_weights(&self, v: &str) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
        let ss_copy = v.chars().collect::<Vec<_>>();
        let ss: &[char] = &ss_copy;
        self.w.to_symbols_and_weights(ss)
    }
}

pub enum SymbolificationWeightingMode {
    ShortestOnly,
    InverseSquareOfLength,
}
