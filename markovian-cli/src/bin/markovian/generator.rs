use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use markovian_core::{
    generator::augment_and_symbolify,
    generator::create_trigrams,
    generator::Generator,
    renderer::RenderChar,
    symbol::{SymbolTableEntryId, SymbolTableWrapper},
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub enum Command {
    /// Create a new generation table
    Create(CreateCommand),

    /// Create some new words
    Generate(GenerateCommand),
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
}

#[derive(Debug, StructOpt)]
pub struct GenerateCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorWrapper {
    Bytes(Generator<u8, f32>),
    String(Generator<char, f32>),
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

    //Symbolify them (and add prefix-suffix)
    let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = input_tokens
        .iter()
        .flat_map(|s| match &symboltable {
            SymbolTableWrapper::Bytes(st) => augment_and_symbolify(&st, s.as_bytes()),
            SymbolTableWrapper::String(st) => {
                augment_and_symbolify(&st, &s.chars().collect::<Vec<_>>())
            }
        })
        .collect();

    let trigrams = create_trigrams(&symbolified_values);

    let generator = match symboltable {
        SymbolTableWrapper::Bytes(st) => {
            GeneratorWrapper::Bytes(Generator::from_trigrams(st, trigrams))
        }
        SymbolTableWrapper::String(st) => {
            GeneratorWrapper::String(Generator::from_trigrams(st, trigrams))
        }
    };

    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}

fn command_generate(cmd: &GenerateCommand) {
    let mut rng = rand::thread_rng();

    // Load the generator
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    match generator {
        GeneratorWrapper::Bytes(_) => println!("BYTES..."),
        GeneratorWrapper::String(gen) => {
            let renderer = RenderChar {
                table: &gen.symbol_table,
                start: "^",
                end: "$",
            };
            for x in gen.generate(20, &mut rng, &renderer).unwrap() {
                println!("{}", x);
            }
        }
    }
}

pub fn run(cmd: &Command) {
    match cmd {
        Command::Create(cmd) => command_create(cmd),
        Command::Generate(cmd) => command_generate(cmd),
    }
}