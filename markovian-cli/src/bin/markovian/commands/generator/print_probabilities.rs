use std::io::BufRead;
use std::path::PathBuf;

use structopt::StructOpt;

use crate::utils::load_generator;

#[derive(Debug, StructOpt)]
pub struct ProbabilityCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// words to print probabilities of
    words: Vec<String>,

    /// Katz back-off coefficient - minimum weight
    /// to use before falling back to a shorter context
    #[structopt(long)]
    katz_coefficient: Option<f32>,
}

impl ProbabilityCommand {
    pub fn run(&self) {
        let generator = load_generator(&self.generator);

        let extra_words: Vec<String> = if self.words.contains(&"-".to_string()) {
            let stdin = std::io::stdin();
            let handle = stdin.lock();
            let lines: Vec<String> = handle
                .lines()
                .map(|n| n.unwrap().trim().to_string())
                .collect();
            lines
        } else {
            vec![]
        };

        for w in &extra_words {
            let lp: f64 = generator.log_prob(w, self.katz_coefficient) as f64;
            println!("{} log10(p)={} p={}", w, lp / (10.0f64).ln(), lp.exp());
        }

        for w in &self.words {
            if w == "-" {
                continue;
            }
            let lp: f64 = generator.log_prob(w, self.katz_coefficient) as f64;
            println!("{} log10(p)={} p={}", w, lp / (10.0f64).ln(), lp.exp());
        }
    }
}
