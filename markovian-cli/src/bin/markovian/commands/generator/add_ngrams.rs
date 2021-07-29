use std::collections::BTreeMap;
use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::SymbolTableEntryId;
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

use crate::utils::{add_ngrams, SymbolificationWeightingMode};

#[derive(Debug, StructOpt)]
pub struct AddNGramsCommand {
    /// input generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// output generator file to use
    #[structopt(parse(from_os_str))]
    output: PathBuf,

    /// Files for input words
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,

    /// maximum length of ngrams to add
    #[structopt(long, short)]
    nmax: usize,

    /// how to weight the symbolifications
    #[structopt(long, possible_values = &SymbolificationWeightingMode::variants(), case_insensitive = true, default_value="ShortestOnly")]
    breaker_mode: SymbolificationWeightingMode,
}

pub fn command_add_ngrams(cmd: &AddNGramsCommand) {
    // TODO: Maybe get this from the generator if its not specified
    let nmax: usize = cmd.nmax;

    // Load the generator
    info!("reading {} ", cmd.generator.display());
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    // Load all the words to work on

    // Get the symbol table
    let (symbol_table, mut ngrams): (SymbolTableWrapper, BTreeMap<Vec<SymbolTableEntryId>, f32>) =
        generator.into_symbol_table_and_ngrams();

    // Tokenize the input
    // ngram-ify the input

    let input_tokens: Vec<String> = crate::utils::read_input_lines(&cmd.input_file, |s| s);

    add_ngrams(
        &symbol_table,
        &cmd.breaker_mode.into(),
        &input_tokens,
        nmax,
        &mut ngrams,
    );

    // add these ngrams to the generator
    let generator = GeneratorWrapper::from_ngrams(symbol_table, ngrams);

    // save the generator
    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}
