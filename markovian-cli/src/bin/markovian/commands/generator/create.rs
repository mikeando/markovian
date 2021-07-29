use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::generate_words;
use markovian_core::symbol::{SymbolTable, SymbolTableEntry, TableEncoding};
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

use super::improve_symbols::{improve_symbol_table, ImproveSymbolTableCallbacks};
use super::SymbolTrimmingMode;
use crate::utils::{build_generator, read_input_lines, SymbolificationWeightingMode};

#[derive(Debug, StructOpt)]
pub struct CreateCommand {
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
    #[structopt(long, possible_values = &SymbolificationWeightingMode::variants(), case_insensitive = true, default_value="ShortestOnly")]
    breaker_mode: SymbolificationWeightingMode,

    /// Weight percent for symbol trimming
    #[structopt(long)]
    symbol_trim_percent_weight: Option<f64>,

    /// Min weight for symbol trimming
    #[structopt(long)]
    symbol_trim_min_weight: Option<f64>,
}

impl CreateCommand {
    pub fn get_symbol_trim_mode(&self) -> Result<SymbolTrimmingMode, String> {
        match (self.symbol_trim_min_weight, self.symbol_trim_percent_weight) {
            (None, None) => Ok(SymbolTrimmingMode::None),
            (None, Some(w)) => Ok(SymbolTrimmingMode::MaxSumPercent(w)),
            (Some(w), None) => Ok(SymbolTrimmingMode::MaxWeight(w)),
            (Some(_), Some(_)) => Err(
                "Can only specify one of symbol_trim_percent_weight and symbol_trim_min_weight"
                    .to_string(),
            ),
        }
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

    fn on_iteration_merge_deterministic_pair(
        &self,
        analyser: &markovian_core::analyser::AnalyserWrapper,
        bigram: (
            markovian_core::symbol::SymbolTableEntryId,
            markovian_core::symbol::SymbolTableEntryId,
        ),
        count: (f64, f64),
        _new_symbol: markovian_core::symbol::SymbolTableEntryId,
    ) {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        info!(
            "Deterministic bigram {}|{} (count={}/{})",
            symbol_renderer.render(bigram.0).unwrap(),
            symbol_renderer.render(bigram.1).unwrap(),
            count.0,
            count.1,
        );
    }

    fn on_end(&self, _analyser: &markovian_core::analyser::AnalyserWrapper) {
        info!("End of improve symbol table");
    }
}

pub fn table_encoding_from_string(v: &str) -> Result<TableEncoding, String> {
    if v.to_lowercase() == "bytes" {
        return Ok(TableEncoding::Bytes);
    }
    if v.to_lowercase() == "string" {
        return Ok(TableEncoding::String);
    }
    Err(format!(
        "Unknown table encoding '{}' valid values are 'bytes', 'string'",
        v
    ))
}

pub fn build_symbol_table(encoding: &TableEncoding, input_tokens: &[String]) -> SymbolTableWrapper {
    match encoding {
        TableEncoding::Bytes => {
            let mut symbol_table = SymbolTable::<u8>::new();
            for k in input_tokens {
                for b in k.as_bytes() {
                    symbol_table.add(SymbolTableEntry::Single(*b)).unwrap();
                }
            }
            SymbolTableWrapper::Bytes(symbol_table)
        }
        TableEncoding::String => {
            let mut symbol_table = SymbolTable::<char>::new();
            for k in input_tokens {
                for c in k.chars() {
                    symbol_table.add(SymbolTableEntry::Single(c)).unwrap();
                }
            }
            SymbolTableWrapper::String(symbol_table)
        }
    }
}

impl CreateCommand {
    //TODO: This needs better error handling
    pub fn run(&self) {
        if self.input_files.is_empty() {
            panic!("Need to specify at least one input file");
        }

        // TODO: Add options to this to allow lowercasing etc.
        // let input_tokens = read_input_lines(&cmd.input_files, |s| s.to_lowercase());
        let input_tokens = read_input_lines(&self.input_files, |s| s);

        info!("Loaded {} words", input_tokens.len());

        // Next we build a symbol table from these words
        let symbol_table = build_symbol_table(&self.encoding, &input_tokens);
        info!("found {} symbols", symbol_table.max_symbol_id());

        // Now we improve the symbol table
        let callbacks = GenerateImproveSymbolTableCallbacks {};
        let symbol_trim_mode = self.get_symbol_trim_mode().unwrap();
        //TODO: This clone is a bit sad.
        let symbol_table: SymbolTableWrapper = improve_symbol_table(
            symbol_table,
            input_tokens.clone(),
            callbacks,
            self.combine_steps,
            symbol_trim_mode,
        );
        info!("now have {} symbols", symbol_table.max_symbol_id());

        // Now we build the transition tables
        let generator = build_generator(symbol_table, &self.breaker_mode, &input_tokens, self.n);

        // Apply bias if requested
        let generator = if let Some(bias_power) = self.bias {
            crate::modify::bias::bias(generator, bias_power)
        } else {
            generator
        };

        if let Some(save_path) = &self.save_generator {
            let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
            std::fs::write(save_path, &encoded).unwrap();
            info!("wrote {} ", save_path.display());
        }

        // Finally we generate some words
        let words = generate_words(
            &generator,
            self.count,
            &self.prefix,
            &self.suffix,
            self.katz_coefficient,
        )
        .unwrap();

        for x in words {
            println!("{}", x);
        }
    }
}
