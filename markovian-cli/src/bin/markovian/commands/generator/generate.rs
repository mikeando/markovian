use std::path::PathBuf;

use markovian_core::generator_wrapper::generate_words;
use structopt::StructOpt;

use crate::utils::load_generator;

#[derive(Debug, StructOpt)]
pub struct GenerateCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// prefix for generated words
    #[structopt(long)]
    prefix: Option<String>,

    /// suffix for generated words
    #[structopt(long)]
    suffix: Option<String>,

    /// number of strings to generate
    #[structopt(long, default_value = "20")]
    count: usize,

    /// Bias to apply to calculated probabilities
    #[structopt(long)]
    bias: Option<f32>,

    /// Katz back-off coefficient - minimum weight
    /// to use before falling back to a shorter context
    #[structopt(long)]
    katz_coefficient: Option<f32>,
}

impl GenerateCommand {
    pub fn run(&self) {
        let generator = load_generator(&self.generator);

        // Apply bias if requested
        let generator = if let Some(bias_power) = self.bias {
            crate::modify::bias::bias(generator, bias_power)
        } else {
            generator
        };

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
