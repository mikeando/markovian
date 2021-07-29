use std::collections::BTreeMap;
use std::path::PathBuf;

use log::info;
use markovian_core::analyser::AnalyserWrapper;
use markovian_core::generator_wrapper::GeneratorWrapper;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct AnalyseSymbolTableCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Files to analyse
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,
}

pub fn command_analyse_symbol_table(cmd: &AnalyseSymbolTableCommand) {
    // Load the generator
    info!("reading {} ", cmd.input.display());
    let data = std::fs::read(&cmd.input).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    let (symbol_table, _ngrams) = generator.into_symbol_table_and_ngrams();

    // Load the text
    // TODO: options to allow to_lowercase in the lambda
    let input_tokens: Vec<String> = crate::utils::read_input_lines(&cmd.input_file, |s| s);

    let analyser = AnalyserWrapper::new(symbol_table, input_tokens);
    let symbol_renderer = analyser.get_symbol_renderer("^", "$");

    //TODO: This counts only shortest tokenizations (which is what we want in the minimize case, but maybe not in the
    // analyse case)
    let symbolization_counts: BTreeMap<usize, usize> = analyser.get_symbolization_ways_counts();
    println!();
    for (len, count) in symbolization_counts {
        println!("{} entries each symbolize exactly {} ways", count, len)
    }
    println!();

    let symbol_counts = analyser.get_ordered_symbol_counts();

    println!("Individual symbol counts");
    for (k, v) in symbol_counts {
        println!("{} {:?}", symbol_renderer.render(k).unwrap(), v);
    }

    println!("--- bigrams ---");

    let bigram_counts = analyser.get_bigram_counts();

    let mut ordered_bigram_counts = bigram_counts.iter().collect::<Vec<_>>();
    ordered_bigram_counts.sort_by(|a, b| (a.1).1.total_cmp(&(b.1).1));

    for (k, c) in ordered_bigram_counts.iter() {
        println!(
            "{}|{} {}",
            symbol_renderer.render(k.0).unwrap(),
            symbol_renderer.render(k.1).unwrap(),
            c.1
        )
    }
}
