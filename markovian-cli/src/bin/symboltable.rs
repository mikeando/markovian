use log::{debug, info};
use std::{collections::BTreeMap, path::PathBuf};
use structopt::StructOpt;

use markovian_core::{
    analyser::AnalyserWrapper,
    renderer::{
        renderer_for_char_with_separator, renderer_for_u8_with_separator, RenderChar, RenderU8,
        Renderer, RendererId, SymbolIdRenderer, SymbolIdRendererChar, SymbolIdRendererU8,
    },
    symbol::{
        SymbolTable, SymbolTableEntry, SymbolTableEntryId, SymbolTableWrapper, TableEncoding,
    },
};

#[derive(Debug, StructOpt)]
#[structopt(name = "markovian", about = "Markov based name generator.")]
struct Opt {
    /// Verbosity level
    #[structopt(short, long, default_value = "0")]
    verbose: i32,

    #[structopt(subcommand)] // Note that we mark a field as a subcommand
    cmd: Command,
}

#[derive(Debug, StructOpt)]
enum Command {
    Print(PrintCommand),
    Generate(GenerateCommand),
    Symbolify(SymbolifyCommand),
    Analyse(AnalyseCommand),
    Improve(ImproveSymbolTableCommand),
}

#[derive(Debug, StructOpt)]
struct PrintCommand {
    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,
}

#[derive(Debug, StructOpt)]
struct GenerateCommand {
    /// Input files
    #[structopt(short, long, parse(from_os_str))]
    input: Vec<PathBuf>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Encoding for table
    #[structopt(short, long, parse(try_from_str = table_encoding_from_string))]
    encoding: TableEncoding,
}

#[derive(Debug, StructOpt)]
struct SymbolifyCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Strings to symbolify
    input: Vec<String>,

    /// Print separator between adjacent symbols
    /// Note that for u8 data, printing using "symbol_seperator=''"
    /// will give a different result to not specifying anything,
    /// due to unicode handling
    #[structopt(long)]
    symbol_separator: Option<String>,

    /// Print symbol-ids rather than the text form oe each symbol
    #[structopt(long)]
    use_symbol_ids: bool,
}

#[derive(Debug, StructOpt)]
struct AnalyseCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Files to analyse
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,
}

#[derive(Debug, StructOpt)]
struct ImproveSymbolTableCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Files to analyse
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,
}

pub fn table_encoding_from_string(v: &str) -> Result<TableEncoding, String> {
    if v.to_lowercase() == "bytes" {
        return Ok(TableEncoding::Bytes);
    }
    if v.to_lowercase() == "string" {
        return Ok(TableEncoding::String);
    }
    Err(format!("Unkown table encoding '{}'", v))
}

fn setup_logging(verbose: i32) {
    let level = match verbose {
        v if v <= 0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        _ => log::LevelFilter::Trace,
    };
    // With fern, we can:

    // Configure logger at runtime
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                //"{}[{}][{}] {}",
                "[{}][{}] {}",
                //chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        // Add blanket level filter -
        .level(level)
        // Output to stdout, files, and other Dispatch configurations
        .chain(std::io::stdout())
        //.chain(fern::log_file("output.log").unwrap())
        // Apply globally
        .apply()
        .unwrap();
}

fn main() {
    let opt = Opt::from_args();
    debug!("{:?}", opt);

    let verbose: i32 = opt.verbose;
    setup_logging(verbose);

    info!("SYMBOLTABLE! {:?}", opt);

    match &opt.cmd {
        Command::Print(p) => command_print(p),
        Command::Generate(g) => command_generate(g),
        Command::Symbolify(s) => command_symbolify(s),
        Command::Analyse(x) => command_analyse(x),
        Command::Improve(x) => command_improve_symbol_table(x),
    }
}

fn command_symbolify(cmd: &SymbolifyCommand) {
    info!("SYMBOLIFY: {:?}", cmd);
    let data = std::fs::read(&cmd.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();
    info!("encoding: {}", symboltable.encoding().encoding_name());
    info!("n_symbols: {}", symboltable.len());

    let renderer: Box<dyn Renderer> = if cmd.use_symbol_ids {
        Box::new(RendererId {})
    } else {
        match (&symboltable, &cmd.symbol_separator) {
            (SymbolTableWrapper::Bytes(table), None) => Box::new(RenderU8 {
                table: &table,
                start: b"^",
                end: b"$",
            }),
            (SymbolTableWrapper::Bytes(table), Some(sep)) => {
                Box::new(renderer_for_u8_with_separator(&table, &sep))
            }
            (SymbolTableWrapper::String(table), None) => Box::new(RenderChar {
                table: &table,
                start: "^",
                end: "$",
            }),
            (SymbolTableWrapper::String(table), Some(sep)) => {
                Box::new(renderer_for_char_with_separator(&table, &sep))
            }
        }
    };

    for s in &cmd.input {
        let reprs = match &symboltable {
            SymbolTableWrapper::Bytes(table) => table.symbolifications(s.as_bytes()),
            SymbolTableWrapper::String(table) => {
                table.symbolifications(&s.chars().collect::<Vec<_>>())
            }
        };
        let result = reprs
            .iter()
            .map(|ids| renderer.render(&ids))
            .collect::<Vec<_>>();
        println!("{} => {:?}", s, result);
    }
}

fn command_print(p: &PrintCommand) {
    info!("PRINT: {:?}", p);
    let data = std::fs::read(&p.input).unwrap();
    let decoded: SymbolTableWrapper = bincode::deserialize(&data).unwrap();
    println!("encoding: {}", decoded.encoding().encoding_name());
    println!("n_symbols: {}", decoded.len());

    match decoded {
        SymbolTableWrapper::Bytes(table) => {
            let symbol_renderer = SymbolIdRendererU8 {
                table: &table,
                start: "START",
                end: "END",
            };
            for e in table.iter() {
                let (k, _v): (SymbolTableEntryId, &SymbolTableEntry<u8>) = e;
                println!("{} => {}", k.0, symbol_renderer.render(k).unwrap());
            }
        }
        SymbolTableWrapper::String(table) => {
            let symbol_renderer = SymbolIdRendererChar {
                table: &table,
                start: "START",
                end: "END",
            };

            for e in table.iter() {
                let (k, _v): (SymbolTableEntryId, &SymbolTableEntry<char>) = e;
                println!("{} => {}", k.0, symbol_renderer.render(k).unwrap());
            }
        }
    }
}

fn command_generate(g: &GenerateCommand) {
    // TODO: Add potential for transformations and customizable min length, and trimming,
    let input_tokens: Vec<String> = g
        .input
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

    println!("using {} input strings", input_tokens.len());

    let file_data = match g.encoding {
        TableEncoding::Bytes => {
            let mut symbol_table = SymbolTable::<u8>::new();
            for k in &input_tokens {
                for b in k.as_bytes() {
                    symbol_table.add(SymbolTableEntry::Single(*b));
                }
            }
            SymbolTableWrapper::Bytes(symbol_table)
        }
        TableEncoding::String => {
            let mut symbol_table = SymbolTable::<char>::new();
            for k in &input_tokens {
                for c in k.chars() {
                    symbol_table.add(SymbolTableEntry::Single(c));
                }
            }
            SymbolTableWrapper::String(symbol_table)
        }
    };
    println!("found {} symbols", file_data.len());

    let encoded: Vec<u8> = bincode::serialize(&file_data).unwrap();
    std::fs::write(&g.output, &encoded).unwrap();
    println!("wrote {} ", g.output.display());
}

fn command_analyse(x: &AnalyseCommand) {
    // Load the symboltable
    let data = std::fs::read(&x.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();

    // Load the text
    let input_tokens: Vec<String> = x
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

    let analyser = AnalyserWrapper::new(symboltable, input_tokens);
    let symbol_renderer = analyser.get_symbol_renderer("^", "$");

    //TODO: This counts only shortest tokenizations (which is what we want in the minimize case, but maybe not in the
    // analyse case)
    let symbolization_counts: BTreeMap<usize, usize> = analyser.get_symbolization_ways_counts();
    println!("");
    for (len, count) in symbolization_counts {
        println!("{} entries each symbolize exactly {} ways", count, len)
    }
    println!("");

    let symbol_counts = analyser.get_ordered_symbol_counts();

    println!("Individual symbol counts");
    for (k, v) in symbol_counts {
        println!("{} {:?}", symbol_renderer.render(k).unwrap(), v);
    }

    println!("--- bigrams ---");

    let bigram_counts = analyser.get_ordered_bigram_counts();

    for (k, c) in bigram_counts.iter() {
        println!(
            "{}|{} {}",
            symbol_renderer.render(k.0).unwrap(),
            symbol_renderer.render(k.1).unwrap(),
            c
        )
    }
}

fn command_improve_symbol_table(x: &ImproveSymbolTableCommand) {
    // Load the symboltable
    let data = std::fs::read(&x.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();

    // Load the text
    let input_tokens: Vec<String> = x
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

    let analyser = AnalyserWrapper::new(symboltable, input_tokens);
    let symbol_renderer = analyser.get_symbol_renderer("^", "$");

    let symbolization_counts: BTreeMap<usize, usize> = analyser.get_symbolization_ways_counts();
    println!("");
    for (len, count) in symbolization_counts {
        println!("{} entries each symbolize exactly {} ways", count, len)
    }
    println!("");

    let symbol_counts = analyser.get_ordered_symbol_counts();

    println!("Individual symbol counts");
    for (k, v) in symbol_counts {
        println!("{} {:?}", symbol_renderer.render(k).unwrap(), v);
    }

    println!("--- bigrams ---");

    let bigram_counts = analyser.get_ordered_bigram_counts();

    for (k, c) in bigram_counts.iter() {
        println!(
            "{}|{} {}",
            symbol_renderer.render(k.0).unwrap(),
            symbol_renderer.render(k.1).unwrap(),
            c
        )
    }

    // Take the commonest bigram and remove it

    // Remove any symbols that don't occur in the input.
}
