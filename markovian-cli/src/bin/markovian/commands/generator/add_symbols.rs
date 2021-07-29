use std::collections::BTreeMap;
use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::SymbolTableEntryId;
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct AddSymbolsCommand {
    /// input generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// symbols to add
    #[structopt(short)]
    symbols: Vec<String>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,
}

pub fn command_add_symbols(cmd: &AddSymbolsCommand) {
    // Load the generator
    info!("reading {} ", cmd.generator.display());
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    // Load all the words to work on

    // Get the symbol table
    let (mut symbol_table, ngrams): (SymbolTableWrapper, BTreeMap<Vec<SymbolTableEntryId>, f32>) =
        generator.into_symbol_table_and_ngrams();

    symbol_table.add_symbols(&cmd.symbols);

    let generator = GeneratorWrapper::from_ngrams(symbol_table, ngrams);

    // save the generator
    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}
