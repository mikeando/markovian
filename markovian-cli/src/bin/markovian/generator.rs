use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, path::PathBuf};

use markovian_core::{
    generator::augment_and_symbolify,
    generator::create_ngrams,
    generator::GenerationError,
    generator::Generator,
    renderer::RenderChar,
    renderer::RenderU8,
    symbol::{SymbolTableEntryId, SymbolTableWrapper},
};
use structopt::StructOpt;

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
}

#[derive(Debug, StructOpt)]
pub struct CreateCommand {
    /// Symbol table to use to symbolify the input words
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Files for input words
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Order of generator to build
    #[structopt(short, default_value = "3")]
    n: usize,
}

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
}

#[derive(Debug, StructOpt)]
pub struct ProbabilityCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// words to pring probabilities of
    words: Vec<String>,
}

#[derive(Debug, StructOpt)]
pub struct BiasCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    // Power to use
    #[structopt(long)]
    power: f32,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorWrapper {
    Bytes(Generator<u8, f32>),
    String(Generator<char, f32>),
}

impl GeneratorWrapper {
    pub fn log_prob(&self, word: &str) -> f32 {
        match &self {
            GeneratorWrapper::Bytes(gen) => gen.log_prob(word.as_bytes()),
            GeneratorWrapper::String(gen) => gen.log_prob(&word.chars().collect::<Vec<_>>()),
        }
    }
}

pub fn build_generator(
    symboltable: SymbolTableWrapper,
    input_tokens: &[String],
    n: usize,
) -> GeneratorWrapper {
    //Symbolify them (and add prefix-suffix)
    let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = input_tokens
        .iter()
        .flat_map(|s| match &symboltable {
            SymbolTableWrapper::Bytes(st) => augment_and_symbolify(&st, s.as_bytes(), n),
            SymbolTableWrapper::String(st) => {
                augment_and_symbolify(&st, &s.chars().collect::<Vec<_>>(), n)
            }
        })
        .collect();

    let trigrams = create_ngrams(&symbolified_values, n);

    match symboltable {
        SymbolTableWrapper::Bytes(st) => {
            GeneratorWrapper::Bytes(Generator::from_ngrams(st, trigrams, n))
        }
        SymbolTableWrapper::String(st) => {
            GeneratorWrapper::String(Generator::from_ngrams(st, trigrams, n))
        }
    }
}

pub fn command_create(cmd: &CreateCommand) {
    println!("Create");

    // Load the symboltable
    let data = std::fs::read(&cmd.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();

    // Load the text
    let input_tokens: Vec<String> = cmd
        .input_file
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

    let generator = build_generator(symboltable, &input_tokens, cmd.n);

    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}

pub fn generate_words(
    generator: &GeneratorWrapper,
    count: usize,
    prefix: &Option<String>,
    suffix: &Option<String>,
) -> Result<Vec<String>, GenerationError> {
    let mut rng = rand::thread_rng();

    match generator {
        GeneratorWrapper::Bytes(gen) => {
            let renderer = RenderU8 {
                table: &gen.symbol_table,
                start: b"^",
                end: b"$",
            };
            let prefix = prefix.as_ref().map(|s| s.as_bytes());
            let suffix = suffix.as_ref().map(|s| s.as_bytes());
            gen.generate_multi(prefix, suffix, count, &mut rng, &renderer)
        }
        GeneratorWrapper::String(gen) => {
            let renderer = RenderChar {
                table: &gen.symbol_table,
                start: "^",
                end: "$",
            };
            let prefix = prefix.as_ref().map(|s| s.chars().collect::<Vec<_>>());
            let suffix = suffix.as_ref().map(|s| s.chars().collect::<Vec<_>>());
            let p_temp = prefix.as_ref().map(|s| s.borrow());
            let s_temp = suffix.as_ref().map(|s| s.borrow());

            gen.generate_multi(p_temp, s_temp, count, &mut rng, &renderer)
        }
    }
}

fn command_generate(cmd: &GenerateCommand) {
    // Load the generator
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

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

    let words = generate_words(&generator, cmd.count, &cmd.prefix, &cmd.suffix).unwrap();

    for x in words {
        println!("{}", x);
    }
}

fn command_print_probabilities(cmd: &ProbabilityCommand) {
    // Load the generator
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    for w in &cmd.words {
        let lp: f64 = generator.log_prob(w) as f64;
        println!("{} ln(p)={} p={}", w, lp, lp.exp());
    }
}

fn command_bias(cmd: &BiasCommand) {
    // Load the generator
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    let gen = match generator {
        GeneratorWrapper::Bytes(gen) => {
            GeneratorWrapper::Bytes(gen.map_probabilities(|p| p.powf(cmd.power)))
        }
        GeneratorWrapper::String(gen) => {
            GeneratorWrapper::String(gen.map_probabilities(|p| p.powf(cmd.power)))
        }
    };

    let encoded: Vec<u8> = bincode::serialize(&gen).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}

pub fn run(cmd: &Command) {
    match cmd {
        Command::Create(cmd) => command_create(cmd),
        Command::Generate(cmd) => command_generate(cmd),
        Command::Probability(cmd) => command_print_probabilities(cmd),
        Command::Bias(cmd) => command_bias(cmd),
    }
}
