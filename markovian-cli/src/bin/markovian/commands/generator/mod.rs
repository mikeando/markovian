use std::collections::BTreeMap;
use std::path::Path;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::SymbolTableEntryId;
use structopt::StructOpt;

pub mod add_ngrams;
pub mod add_symbols;
pub mod analyse_symbols;
pub mod bias;
pub mod create;
pub mod generate;
pub mod improve_symbols;
pub mod info;
pub mod print_probabilities;
pub mod print_symboltable;
pub mod remove_symbols;
pub mod strip_ngrams;
pub mod symbolify;
pub mod trim_ngrams;

pub fn wrap_load_transform_ngrams_save<F>(from: &Path, to: &Path, f: F)
where
    F: FnOnce(BTreeMap<Vec<SymbolTableEntryId>, f32>) -> BTreeMap<Vec<SymbolTableEntryId>, f32>,
{
    // Load the generator
    info!("reading {} ", from.display());
    let data = std::fs::read(from).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    let (symbol_table, ngrams) = generator.into_symbol_table_and_ngrams();
    let ngrams = f(ngrams);

    let generator: GeneratorWrapper = GeneratorWrapper::from_ngrams(symbol_table, ngrams);

    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(to, &encoded).unwrap();
    println!("wrote {} ", to.display());
}

use self::add_ngrams::AddNGramsCommand;
use self::add_symbols::AddSymbolsCommand;
use self::analyse_symbols::AnalyseSymbolTableCommand;
use self::bias::BiasCommand;
use self::create::CreateCommand;
use self::generate::GenerateCommand;
use self::improve_symbols::ImproveSymbolTableCommand;
use self::info::InfoCommand;
use self::print_probabilities::ProbabilityCommand;
use self::print_symboltable::PrintSymbolTableCommand;
use self::remove_symbols::RemoveSymbolsCommand;
use self::strip_ngrams::StripNGramsCommand;
use self::symbolify::SymbolifyCommand;
use self::trim_ngrams::TrimNGramsCommand;

#[derive(Debug, StructOpt)]
pub enum Command {
    /// Create a new generation table
    Create(CreateCommand),

    /// Create some new words
    Generate(GenerateCommand),

    /// Print word probabilities
    Probability(ProbabilityCommand),

    /// Apply bias to probabilities
    Bias(BiasCommand),

    /// Print information about the generator
    Info(InfoCommand),

    /// Trim out low weight n-grams
    TrimNGrams(TrimNGramsCommand),

    /// Remove all ngrams in this generator. This leaves it unusable
    /// until ngrams are added.
    StripNgrams(StripNGramsCommand),

    /// Process an input file and add all ngrams to a generator
    AddNgrams(AddNGramsCommand),

    /// Add specific symbols to a generators symbol table
    AddSymbols(AddSymbolsCommand),

    /// Remove specific symbols (and all ngrams referencing them)
    /// from a generator.
    RemoveSymbols(RemoveSymbolsCommand),

    // Print the symbol table information in a generator
    PrintSymbolTable(PrintSymbolTableCommand),

    AnalyseSymbolTable(AnalyseSymbolTableCommand),

    ImproveSymbolTable(ImproveSymbolTableCommand),

    Symbolify(SymbolifyCommand),
}

#[derive(PartialEq, Debug)]
pub enum SymbolTrimmingMode {
    None,
    MaxSumPercent(f64),
    MaxWeight(f64),
}

pub fn run(cmd: &Command) {
    match cmd {
        Command::Create(cmd) => cmd.run(),
        Command::Generate(cmd) => cmd.run(),
        Command::Probability(cmd) => cmd.run(),
        Command::Bias(cmd) => cmd.run(),
        Command::Info(cmd) => info::command_info(cmd),
        Command::TrimNGrams(cmd) => trim_ngrams::command_trim_ngrams(cmd),
        Command::StripNgrams(cmd) => strip_ngrams::command_strip_ngrams(cmd),
        Command::AddNgrams(cmd) => add_ngrams::command_add_ngrams(cmd),
        Command::AddSymbols(cmd) => add_symbols::command_add_symbols(cmd),
        Command::RemoveSymbols(cmd) => remove_symbols::command_remove_symbols(cmd),
        Command::PrintSymbolTable(cmd) => print_symboltable::command_print_symbol_table(cmd),
        Command::AnalyseSymbolTable(cmd) => analyse_symbols::command_analyse_symbol_table(cmd),
        Command::ImproveSymbolTable(cmd) => improve_symbols::command_improve_symbol_table(cmd),
        Command::Symbolify(s) => symbolify::command_symbolify(s),
    }
}
