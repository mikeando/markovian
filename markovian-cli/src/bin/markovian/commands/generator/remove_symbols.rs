use std::collections::BTreeMap;
use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::{SymbolRemapper, SymbolTableEntryId};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct RemoveSymbolsCommand {
    /// input generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// symbols to remove
    #[structopt(short)]
    symbols: Vec<String>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,
}

pub fn command_remove_symbols(cmd: &RemoveSymbolsCommand) {
    // Load the generator
    info!("reading {} ", cmd.generator.display());
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    // Get the symbol table
    let (mut symbol_table, ngrams) = generator.into_symbol_table_and_ngrams();

    let remapper: SymbolRemapper = symbol_table.remove_symbols_and_compress(&cmd.symbols);

    let ngrams: BTreeMap<Vec<SymbolTableEntryId>, f32> = ngrams
        .into_iter()
        .filter_map(|(k, w)| remapper.map(k).map(|k| (k, w)))
        .collect();

    let generator = GeneratorWrapper::from_ngrams(symbol_table, ngrams);

    // save the generator
    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}
