use std::io::BufRead;
use std::path::PathBuf;

use log::info;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::renderer::{
    renderer_for_char_with_separator,
    renderer_for_u8_with_separator,
    RenderChar,
    RenderU8,
    Renderer,
    RendererId,
};
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub struct SymbolifyCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// Strings to symbolify
    input: Vec<String>,

    /// Print separator between adjacent symbols
    /// Note that for u8 data, printing using "symbol_separator=''"
    /// will give a different result to not specifying anything,
    /// due to unicode handling
    #[structopt(long)]
    symbol_separator: Option<String>,

    /// Print symbol-ids rather than the text form oe each symbol
    #[structopt(long)]
    use_symbol_ids: bool,

    /// Only print the shortest symbolifications
    #[structopt(long)]
    all: bool,
}

pub fn command_symbolify(cmd: &SymbolifyCommand) {
    info!("SYMBOLIFY: {:?}", cmd);
    let data = std::fs::read(&cmd.generator).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();
    let (symboltable, _ngrams) = generator.into_symbol_table_and_ngrams();

    info!("encoding: {}", symboltable.encoding().encoding_name());
    info!("max symbol id: {}", symboltable.max_symbol_id());

    let renderer: Box<dyn Renderer> = if cmd.use_symbol_ids {
        Box::new(RendererId {})
    } else {
        match (&symboltable, &cmd.symbol_separator) {
            (SymbolTableWrapper::Bytes(table), None) => Box::new(RenderU8 {
                table,
                start: b"^",
                end: b"$",
            }),
            (SymbolTableWrapper::Bytes(table), Some(sep)) => {
                Box::new(renderer_for_u8_with_separator(table, sep))
            }
            (SymbolTableWrapper::String(table), None) => Box::new(RenderChar {
                table,
                start: "^",
                end: "$",
            }),
            (SymbolTableWrapper::String(table), Some(sep)) => {
                Box::new(renderer_for_char_with_separator(table, sep))
            }
        }
    };

    let mut extra_words: Vec<String> = if cmd.input.contains(&"-".to_string()) {
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

    for s in &cmd.input {
        if s == "-" {
            continue;
        }
        extra_words.push(s.clone())
    }

    for s in &extra_words {
        let reprs = match &symboltable {
            SymbolTableWrapper::Bytes(table) => table.symbolifications(s.as_bytes()),
            SymbolTableWrapper::String(table) => {
                table.symbolifications(&s.chars().collect::<Vec<_>>())
            }
        };
        let reprs = if !cmd.all {
            if let Some(min_length) = reprs.iter().map(|s| s.len()).min() {
                reprs
                    .into_iter()
                    .filter(|s| s.len() == min_length)
                    .collect()
            } else {
                vec![]
            }
        } else {
            reprs
        };
        let result = reprs
            .iter()
            .map(|ids| renderer.render(ids))
            .collect::<Result<Vec<_>, _>>();
        match result {
            Ok(v) => {
                println!("{} => {:?}", s, v);
            }
            Err(e) => println!("error: {:?} caused {:?}", s, e),
        }
    }
}
