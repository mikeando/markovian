use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use markovian_core::generator::create_ngrams;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::SymbolTableEntryId;
use markovian_core::symboltable_wrapper::{self, SymbolTableWrapper};
use markovian_core::vecutils::pad;
use structopt::clap::arg_enum;

pub fn read_input_lines<F>(input_files: &[PathBuf], f: F) -> Vec<String>
where
    F: Fn(String) -> String,
{
    let stdin_flag = PathBuf::from("-");
    let mut extra_lines: Vec<String> = if input_files.contains(&stdin_flag) {
        let stdin = std::io::stdin();
        let handle = stdin.lock();
        let lines: Vec<String> = handle
            .lines()
            .map(|n| n.unwrap().trim().to_string())
            .map(|s| f(s)) // TODO: Extract and make configurable
            .filter(|s| s.len() >= 3)
            .collect();
        lines
    } else {
        vec![]
    };

    // Load the text
    let mut input_tokens: Vec<String> = input_files
        .iter()
        .filter(|&p| *p != stdin_flag)
        .map(|n| {
            let v: Vec<_> = std::fs::read_to_string(n)
                .unwrap()
                .lines()
                .map(|n| n.trim().to_string())
                .map(|s| f(s)) // TODO: Extract and make configurable
                .filter(|s| s.len() >= 3)
                .collect();
            v
        })
        .flatten()
        .collect();
    input_tokens.append(&mut extra_lines);
    drop(extra_lines);
    input_tokens
}

arg_enum! {
    #[derive(Debug, Clone, Copy)]
    pub enum SymbolificationWeightingMode {
        ShortestOnly,
        InverseSquareOfLength,
    }
}

impl From<SymbolificationWeightingMode> for symboltable_wrapper::SymbolificationWeightingMode {
    fn from(v: SymbolificationWeightingMode) -> Self {
        match v {
            SymbolificationWeightingMode::ShortestOnly => {
                symboltable_wrapper::SymbolificationWeightingMode::ShortestOnly
            }
            SymbolificationWeightingMode::InverseSquareOfLength => {
                symboltable_wrapper::SymbolificationWeightingMode::InverseSquareOfLength
            }
        }
    }
}

pub fn load_generator(generator_file: &Path) -> GeneratorWrapper {
    // Load the generator
    let data = std::fs::read(&generator_file).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();
    generator
}

pub fn save_generator(generator: &GeneratorWrapper, generator_file: &Path) {
    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&generator_file, &encoded).unwrap();
    println!("wrote {} ", generator_file.display());
}

//TODO: In an ideal world this would not allocate and clone, but instead return an iterator of slices
fn ngramify<T>(v: &[T], n_max: usize) -> Vec<Vec<T>>
where
    T: Clone,
{
    let mut result = vec![];
    for n in 1..=n_max {
        for w in v.windows(n) {
            result.push(w.to_vec())
        }
    }
    result
}

// TODO: Use this in other places?
pub fn add_ngrams(
    symbol_table: &SymbolTableWrapper,
    breaker_mode: &symboltable_wrapper::SymbolificationWeightingMode,
    input_tokens: &[String],
    nmax: usize,
    ngrams: &mut BTreeMap<Vec<SymbolTableEntryId>, f32>,
) {
    let breaker = symbol_table.get_word_breaker(breaker_mode);

    let start_id = SymbolTableEntryId(0);
    let end_id = SymbolTableEntryId(1);

    //Symbolify them
    for (ngram, w) in input_tokens
        .iter()
        .flat_map(|s| breaker.to_symbols_and_weights(s))
        .map(|(x, w)| (pad(nmax - 1, start_id, end_id, x), w))
        .flat_map(|(ss, w)| {
            ngramify(&ss, nmax)
                .into_iter()
                .map(move |ngram: Vec<SymbolTableEntryId>| (ngram, w))
        })
    {
        // Only add entries that are not all ^ or $
        if ngram.last().copied() == Some(start_id) || ngram[0] == end_id {
            continue;
        }
        *ngrams.entry(ngram).or_default() += w;
    }
}

pub fn to_padded_symbolifications_and_weights(
    symboltable: &SymbolTableWrapper,
    breaker_mode: &symboltable_wrapper::SymbolificationWeightingMode,
    input_tokens: &[String],
    n: usize,
) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
    let start_id = SymbolTableEntryId(0);
    let end_id = SymbolTableEntryId(1);

    let breaker = symboltable.get_word_breaker(breaker_mode);

    //Symbolify them (and add prefix-suffix)
    let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = input_tokens
        .iter()
        .flat_map(|s| breaker.to_symbols_and_weights(s))
        .map(|(x, w)| (pad(n - 1, start_id, end_id, x), w))
        .collect();

    drop(breaker);

    symbolified_values
}

pub fn build_generator(
    symboltable: SymbolTableWrapper,
    breaker_mode: &SymbolificationWeightingMode,
    input_tokens: &[String],
    n: usize,
) -> GeneratorWrapper {
    assert!(n > 1);
    let symbolified_values = to_padded_symbolifications_and_weights(
        &symboltable,
        &(*breaker_mode).into(),
        input_tokens,
        n,
    );
    let ngrams = create_ngrams(&symbolified_values, n);
    GeneratorWrapper::from_ngrams(symboltable, ngrams)
}
