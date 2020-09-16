use std::path::PathBuf;

use structopt::StructOpt;

use log::info;

use crate::{
    generator::build_generator,
    generator::GeneratorWrapper,
    symboltable::{
        build_symbol_table, improve_symbol_table, table_encoding_from_string,
        ImproveSymbolTableCallbacks,
    },
};
use markovian_core::{
    renderer::RenderChar,
    renderer::RenderU8,
    symbol::{SymbolTableWrapper, TableEncoding},
};

#[derive(Debug, StructOpt)]
pub enum Command {
    /// simple word generation
    Generate(GenerateCommand),
}

#[derive(Debug, StructOpt)]
pub struct GenerateCommand {
    /// Files for input words
    #[structopt(parse(from_os_str))]
    input_files: Vec<PathBuf>,

    /// Encoding for table
    #[structopt(short, long, parse(try_from_str = table_encoding_from_string))]
    encoding: TableEncoding,
}

pub fn run(cmd: &Command) {
    match cmd {
        Command::Generate(cmd) => command_generate(cmd),
    }
}

struct GenerateImproveSymbolTableCallbacks {}

impl ImproveSymbolTableCallbacks for GenerateImproveSymbolTableCallbacks {
    fn on_init(&self, _analyser: &markovian_core::analyser::AnalyserWrapper) {
        info!("Beginning improve symbol table");
    }

    fn on_iteration_after_merge(
        &self,
        analyser: &markovian_core::analyser::AnalyserWrapper,
        bigram: (
            markovian_core::symbol::SymbolTableEntryId,
            markovian_core::symbol::SymbolTableEntryId,
        ),
        count: f64,
        _new_symbol: markovian_core::symbol::SymbolTableEntryId,
    ) {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        info!(
            "Merged bigram {}|{} (count={})",
            symbol_renderer.render(bigram.0).unwrap(),
            symbol_renderer.render(bigram.1).unwrap(),
            count,
        );
    }

    fn on_end(&self, _analyser: &markovian_core::analyser::AnalyserWrapper) {
        info!("End of improve symbol table");
    }
}

//TODO: This needs better error handling
fn command_generate(cmd: &GenerateCommand) {
    if cmd.input_files.is_empty() {
        panic!("Need to specify at least one input file");
    }

    //TODO: Extract this as a function as it is reused a lot...
    // Load the text
    let input_tokens: Vec<String> = cmd
        .input_files
        .iter()
        .map(|n| {
            let v: Vec<_> = std::fs::read_to_string(n)
                .unwrap()
                .lines()
                .map(|n| n.trim().to_string())
                .filter(|s| s.len() >= 3)
                .collect();
            v
        })
        .flatten()
        .collect();

    info!("Loaded {} words", input_tokens.len());

    // Next we build a symbol table from these words
    let symbol_table = build_symbol_table(&cmd.encoding, &input_tokens);
    info!("found {} symbols", symbol_table.max_symbol_id());

    // Now we improve the symbol table
    let callbacks = GenerateImproveSymbolTableCallbacks {};
    //TODO: This clone is a bit sad.
    let symbol_table: SymbolTableWrapper =
        improve_symbol_table(symbol_table, input_tokens.clone(), callbacks);
    info!("now have {} symbols", symbol_table.max_symbol_id());

    // Now we build the transition tables
    let generator = build_generator(symbol_table, &input_tokens);

    // Finally we generate some words
    // TODO: This is a duplicate of what is in generator.rs - extract it?
    let mut rng = rand::thread_rng();

    match generator {
        // TODO: implement this
        GeneratorWrapper::Bytes(gen) => {
            let renderer = RenderU8 {
                table: &gen.symbol_table,
                start: b"^",
                end: b"$",
            };
            // TODO: Make this configurable
            for x in gen.generate(20, &mut rng, &renderer).unwrap() {
                println!("{}", x);
            }
        }
        GeneratorWrapper::String(gen) => {
            let renderer = RenderChar {
                table: &gen.symbol_table,
                start: "^",
                end: "$",
            };
            // TODO: Make this configurable
            for x in gen.generate(20, &mut rng, &renderer).unwrap() {
                println!("{}", x);
            }
        }
    }
}
