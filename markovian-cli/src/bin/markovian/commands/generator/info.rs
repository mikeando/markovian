use std::path::PathBuf;

use markovian_core::generator::{Generator, WeightRange, WeightSummary};
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::renderer::{
    renderer_for_char_with_separator,
    renderer_for_u8_with_separator,
    Renderer,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct InfoCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,
}

pub fn command_info(cmd: &InfoCommand) {
    // Load the generator
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    match &generator {
        GeneratorWrapper::Bytes(gen) => {
            let renderer = renderer_for_u8_with_separator(&gen.symbol_table, ".");
            print_info(gen, &renderer);
        }
        GeneratorWrapper::String(gen) => {
            let renderer = renderer_for_char_with_separator(&gen.symbol_table, ".");
            print_info(gen, &renderer);
        }
    };
}

fn format_weight_range(w: &WeightRange) -> String {
    format!(
        "range: {} -- {}  mean: {}  count: {}",
        w.min_weight, w.max_weight, w.mean_weight, w.count
    )
}

fn format_weight_summary(w: &WeightSummary, renderer: &dyn Renderer) -> String {
    let mut result = String::new();
    for e in &w.quantiles {
        result = format!(
            "{}\n   {}:{:.2}  [{}]",
            result,
            100.0 * e.q,
            e.w,
            renderer.render(&e.sym).unwrap()
        );
    }
    result
}

fn print_info<T>(gen: &Generator<T, f32>, renderer: &dyn Renderer)
where
    T: Ord + Clone,
{
    let info = gen.get_info();
    println!("Prefix totals (Katz Coefficients)");
    for (k, v) in &info.prefix_weights_by_length {
        println!("  {} : {}", k, format_weight_range(v));
    }
    println!("N-Gram weights");
    for (k, v) in &info.ngram_weights_by_length {
        println!("  {} : {:?}", k, format_weight_range(v));
    }

    println!("N-gram weight summaries");
    for (k, v) in &info.ngram_weight_summaries_by_length {
        println!("  {} : {}", k, format_weight_summary(v, renderer));
    }
}
