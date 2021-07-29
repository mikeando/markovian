use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::renderer::{SymbolIdRenderer, SymbolIdRendererChar, SymbolIdRendererU8};
use markovian_core::symbol::{SymbolTableEntry, SymbolTableEntryId};
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct PrintSymbolTableCommand {
    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,
}

pub fn command_print_symbol_table(cmd: &PrintSymbolTableCommand) {
    // Load the generator
    info!("reading {} ", cmd.input.display());
    let data = std::fs::read(&cmd.input).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    let (symbol_table, _ngrams) = generator.into_symbol_table_and_ngrams();

    println!("encoding: {}", symbol_table.encoding().encoding_name());
    println!("max symbol id: {}", symbol_table.max_symbol_id());

    match symbol_table {
        SymbolTableWrapper::Bytes(table) => {
            let symbol_renderer = SymbolIdRendererU8 {
                table: &table,
                start: "START",
                end: "END",
            };
            for e in table.iter() {
                let (k, _v): (SymbolTableEntryId, &SymbolTableEntry<u8>) = e;
                println!("{} => {}", k.0, symbol_renderer.render(k).unwrap());
            }
        }
        SymbolTableWrapper::String(table) => {
            let symbol_renderer = SymbolIdRendererChar {
                table: &table,
                start: "START",
                end: "END",
            };

            for e in table.iter() {
                let (k, _v): (SymbolTableEntryId, &SymbolTableEntry<char>) = e;
                println!("{} => {}", k.0, symbol_renderer.render(k).unwrap());
            }
        }
    }
}
