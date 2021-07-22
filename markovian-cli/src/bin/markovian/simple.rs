use std::path::PathBuf;

use log::info;
use markovian_core::symbol::{SymbolTableWrapper, TableEncoding};
use structopt::StructOpt;

use crate::generator::{build_generator, generate_words, BreakerMode, GeneratorWrapper};
use crate::symboltable::{
    build_symbol_table,
    improve_symbol_table,
    table_encoding_from_string,
    ImproveSymbolTableCallbacks,
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
    #[structopt(short, long, parse(try_from_str = table_encoding_from_string), default_value="string")]
    encoding: TableEncoding,

    /// prefix for generated words
    #[structopt(long)]
    prefix: Option<String>,

    /// suffix for generated words
    #[structopt(long)]
    suffix: Option<String>,

    /// number of strings to generate
    #[structopt(long, default_value = "20")]
    count: usize,

    /// Order of generator to build
    #[structopt(short, default_value = "3")]
    n: usize,

    /// Number of symbol combine steps to perform
    #[structopt(long, default_value = "50")]
    combine_steps: usize,

    /// Where to save the generator
    #[structopt(long, parse(from_os_str))]
    save_generator: Option<PathBuf>,

    /// Bias to apply to calculated probabilities
    #[structopt(long)]
    bias: Option<f32>,

    /// Katz back-off coefficient - minimum weight
    /// to use before falling back to a shorter context
    #[structopt(long)]
    katz_coefficient: Option<f32>,

    /// Mode for breaking words into symbols
    #[structopt(long, possible_values = &BreakerMode::variants(), case_insensitive = true, default_value="ShortestOnly")]
    breaker_mode: BreakerMode,
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
    let symbol_table: SymbolTableWrapper = improve_symbol_table(
        symbol_table,
        input_tokens.clone(),
        callbacks,
        cmd.combine_steps,
    );
    info!("now have {} symbols", symbol_table.max_symbol_id());

    // Now we build the transition tables
    let generator = build_generator(symbol_table, &cmd.breaker_mode, &input_tokens, cmd.n);

    // Apply bias if requested
    let generator = if let Some(bias) = cmd.bias {
        match generator {
            GeneratorWrapper::Bytes(gen) => {
                GeneratorWrapper::Bytes(gen.map_probabilities(|p| p.powf(bias)))
            }
            GeneratorWrapper::String(gen) => {
                GeneratorWrapper::String(gen.map_probabilities(|p| p.powf(bias)))
            }
        }
    } else {
        generator
    };

    if let Some(save_path) = &cmd.save_generator {
        let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
        std::fs::write(save_path, &encoded).unwrap();
        info!("wrote {} ", save_path.display());
    }

    // Finally we generate some words
    let words = generate_words(
        &generator,
        cmd.count,
        &cmd.prefix,
        &cmd.suffix,
        cmd.katz_coefficient,
    )
    .unwrap();

    for x in words {
        println!("{}", x);
    }
}
