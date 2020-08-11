use ::serde::{Deserialize, Serialize};
use log::{debug, info};
use std::{collections::BTreeMap, path::PathBuf};
use structopt::StructOpt;

use markovian_core::{
    ngram::BigramCount,
    symbol::{SymbolTable, SymbolTableEntry, SymbolTableEntryId},
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

#[derive(Debug, Clone, Serialize, Deserialize)]
enum SymbolTableFile {
    Bytes(SymbolTable<u8>),
    String(SymbolTable<char>),
}

impl SymbolTableFile {
    pub fn encoding(&self) -> TableEncoding {
        match self {
            SymbolTableFile::Bytes(_) => TableEncoding::Bytes,
            SymbolTableFile::String(_) => TableEncoding::String,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SymbolTableFile::Bytes(table) => table.len(),
            SymbolTableFile::String(table) => table.len(),
        }
    }

    pub fn symbolifications(&self, s: &str) -> Vec<Vec<SymbolTableEntryId>> {
        match self {
            SymbolTableFile::Bytes(table) => table.symbolifications_str(s),
            SymbolTableFile::String(table) => {
                table.symbolifications(&s.chars().collect::<Vec<_>>())
            }
        }
    }
}

#[derive(Debug)]
pub enum TableEncoding {
    Bytes,
    String,
}

impl TableEncoding {
    pub fn encoding_name(&self) -> &str {
        match self {
            TableEncoding::Bytes => "u8",
            TableEncoding::String => "char",
        }
    }
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

fn symbols_to_string_u8(
    symbols: &[SymbolTableEntryId],
    table: &SymbolTable<u8>,
    separator: &str,
) -> String {
    use std::fmt::Write;

    let mut result: String = String::new();
    let mut is_start = true;
    for s in symbols {
        let symbol = table.get_by_id(*s).unwrap();
        let symbol_str = symbol_table_entry_to_string_u8(symbol, "^", "$");
        if is_start {
            is_start = false;
            write!(&mut result, "{}", symbol_str).unwrap();
        } else {
            write!(&mut result, "{}{}", separator, symbol_str).unwrap();
        }
    }
    result
}

fn symbols_to_string_char(
    symbols: &[SymbolTableEntryId],
    table: &SymbolTable<char>,
    separator: &str,
) -> String {
    use std::fmt::Write;

    let mut result: String = String::new();
    let mut is_start = true;
    for s in symbols {
        let symbol = table.get_by_id(*s).unwrap();
        let symbol_str = symbol_table_entry_to_string_char(symbol, "^", "$");
        if is_start {
            is_start = false;
            write!(&mut result, "{}", symbol_str).unwrap();
        } else {
            write!(&mut result, "{}{}", separator, symbol_str).unwrap();
        }
    }
    result
}

fn symbol_table_entry_to_string_u8(s: &SymbolTableEntry<u8>, start: &str, end: &str) -> String {
    match s {
        SymbolTableEntry::Start => start.to_string(),
        SymbolTableEntry::End => end.to_string(),
        SymbolTableEntry::Single(b) => {
            let part: Vec<u8> = std::ascii::escape_default(*b).collect();
            String::from_utf8(part).unwrap()
        }
        SymbolTableEntry::Compound(bs) => {
            let bs2 = bs.clone();
            match String::from_utf8(bs2) {
                Ok(s) => s,
                Err(_) => {
                    let part: Vec<u8> = bs
                        .iter()
                        .flat_map(|b| std::ascii::escape_default(*b).collect::<Vec<_>>())
                        .collect();
                    String::from_utf8(part).unwrap()
                }
            }
        }
        SymbolTableEntry::Dead(_) => "✞".to_string(),
    }
}

fn symbol_table_entry_to_string_char(s: &SymbolTableEntry<char>, start: &str, end: &str) -> String {
    match s {
        SymbolTableEntry::Start => start.to_string(),
        SymbolTableEntry::End => end.to_string(),
        SymbolTableEntry::Single(c) => format!("{}", c),
        SymbolTableEntry::Compound(cs) => cs.iter().collect::<String>(),
        SymbolTableEntry::Dead(_) => "✞".to_string(),
    }
}

fn symbol_table_id_to_string(
    symbol_table: &SymbolTableFile,
    id: SymbolTableEntryId,
    start: &str,
    end: &str,
) -> String {
    match &symbol_table {
        SymbolTableFile::Bytes(table) => {
            symbol_table_entry_to_string_u8(table.get_by_id(id).unwrap(), start, end)
        }
        SymbolTableFile::String(table) => {
            symbol_table_entry_to_string_char(table.get_by_id(id).unwrap(), start, end)
        }
    }
}

fn command_symbolify(cmd: &SymbolifyCommand) {
    info!("SYMBOLIFY: {:?}", cmd);
    let data = std::fs::read(&cmd.symboltable).unwrap();
    let symboltable: SymbolTableFile = bincode::deserialize(&data).unwrap();
    info!("encoding: {}", symboltable.encoding().encoding_name());
    info!("n_symbols: {}", symboltable.len());

    match symboltable {
        SymbolTableFile::Bytes(table) => {
            for s in &cmd.input {
                let reprs = table.symbolifications(s.as_bytes());
                if cmd.use_symbol_ids {
                    let result = reprs
                        .iter()
                        .map(|ids| ids.iter().map(|id| id.0).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    println!("{} => {:?}", s, result);
                } else {
                    match &cmd.symbol_separator {
                        None => {
                            let result = reprs.iter().map(|v| table.render(&v)).collect::<Vec<_>>();
                            println!("{} => {:?}", s, result);
                        }
                        Some(sep) => {
                            let result = reprs
                                .iter()
                                .map(|ids| symbols_to_string_u8(ids, &table, &sep))
                                .collect::<Vec<_>>();
                            println!("{} => {:?}", s, result);
                        }
                    }
                }
            }
        }
        SymbolTableFile::String(table) => {
            for s in &cmd.input {
                let reprs = table.symbolifications(&s.chars().collect::<Vec<_>>());
                if cmd.use_symbol_ids {
                    let result = reprs
                        .iter()
                        .map(|ids| ids.iter().map(|id| id.0).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    println!("{} => {:?}", s, result);
                } else {
                    match &cmd.symbol_separator {
                        None => {
                            let result = reprs.iter().map(|v| table.render(&v)).collect::<Vec<_>>();
                            println!("{} => {:?}", s, result);
                        }
                        Some(sep) => {
                            let result = reprs
                                .iter()
                                .map(|ids| symbols_to_string_char(ids, &table, &sep))
                                .collect::<Vec<_>>();
                            println!("{} => {:?}", s, result);
                        }
                    }
                }
            }
        }
    }
}

fn command_print(p: &PrintCommand) {
    info!("PRINT: {:?}", p);
    let data = std::fs::read(&p.input).unwrap();
    let decoded: SymbolTableFile = bincode::deserialize(&data).unwrap();
    println!("encoding: {}", decoded.encoding().encoding_name());
    println!("n_symbols: {}", decoded.len());

    match decoded {
        SymbolTableFile::Bytes(table) => {
            for e in table.iter() {
                let (k, v): (SymbolTableEntryId, &SymbolTableEntry<u8>) = e;
                println!(
                    "{} => {}",
                    k.0,
                    symbol_table_entry_to_string_u8(v, "START", "END")
                );
            }
        }
        SymbolTableFile::String(table) => {
            for e in table.iter() {
                let (k, v): (SymbolTableEntryId, &SymbolTableEntry<char>) = e;
                println!(
                    "{} => {}",
                    k.0,
                    symbol_table_entry_to_string_char(v, "START", "END")
                );
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
            SymbolTableFile::Bytes(symbol_table)
        }
        TableEncoding::String => {
            let mut symbol_table = SymbolTable::<char>::new();
            for k in &input_tokens {
                for c in k.chars() {
                    symbol_table.add(SymbolTableEntry::Single(c));
                }
            }
            SymbolTableFile::String(symbol_table)
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
    let symboltable: SymbolTableFile = bincode::deserialize(&data).unwrap();

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

    let input_tokens = input_tokens
        .iter()
        .map(|s| symboltable.symbolifications(s))
        .collect::<Vec<_>>();

    let mut symbolization_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for k in &input_tokens {
        *symbolization_counts.entry(k.len()).or_insert(0) += 1;
    }

    println!("");
    for (len, count) in symbolization_counts {
        println!("{} entries each symbolize exactly {} ways", count, len)
    }
    println!("");

    let mut symbol_counts: BTreeMap<SymbolTableEntryId, usize> = BTreeMap::new();
    for x in input_tokens.iter().flatten().flatten() {
        *symbol_counts.entry(*x).or_insert(0) += 1
    }

    let mut symbol_counts: Vec<_> = symbol_counts.into_iter().collect();
    symbol_counts.sort_by_key(|e| e.1);
    symbol_counts.reverse();

    println!("Individual symbol counts");
    for (k, v) in symbol_counts {
        println!(
            "{} {:?}",
            symbol_table_id_to_string(&symboltable, k, "^", "$"),
            v
        );
    }

    println!("--- bigrams ---");
    let bigram_counts: BigramCount<SymbolTableEntryId, usize> = input_tokens
        .iter()
        .flatten()
        .map(|v| -> &[SymbolTableEntryId] { &v })
        .collect();

    let mut bigram_counts: Vec<_> = bigram_counts
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    bigram_counts.sort_by_key(|e| e.1);
    bigram_counts.reverse();

    for (k, c) in bigram_counts.iter() {
        println!(
            "{}|{} {}",
            symbol_table_id_to_string(&symboltable, k.0, "^", "$"),
            symbol_table_id_to_string(&symboltable, k.1, "^", "$"),
            c
        )
    }
}

fn command_improve_symbol_table(x: &ImproveSymbolTableCommand) {
    // Load the symboltable
    let data = std::fs::read(&x.symboltable).unwrap();
    let symboltable: SymbolTableFile = bincode::deserialize(&data).unwrap();

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

    let input_tokens = input_tokens
        .iter()
        .map(|s| {
            let ss = symboltable.symbolifications(&s);
            (s, ss)
        })
        .collect::<Vec<_>>();

    // In this case we
    //   1. take the shortest form
    //   2. error if there is no valid form
    //   3. If more than one form has the same length, we add all with that length.
    let input_tokens = input_tokens
        .iter()
        .map(|(k, v)| {
            //TODO: Remove the unwrap!
            let min_len = v.iter().map(|ss| ss.len()).min().unwrap();
            let short: Vec<_> = v.iter().filter(|ss| ss.len() == min_len).collect();
            (k, short)
        })
        .collect::<Vec<_>>();

    let mut symbol_counts: BTreeMap<SymbolTableEntryId, usize> = BTreeMap::new();

    for x in input_tokens.iter().flat_map(|(_k, v)| v).cloned().flatten() {
        *symbol_counts.entry(*x).or_insert(0) += 1
    }

    let mut symbol_counts: Vec<_> = symbol_counts.into_iter().collect();
    symbol_counts.sort_by_key(|e| e.1);
    symbol_counts.reverse();

    println!("Individual symbol counts");
    for (k, v) in symbol_counts {
        println!(
            "{} {:?}",
            symbol_table_id_to_string(&symboltable, k, "^", "$"),
            v
        );
    }

    println!("--- bigrams ---");
    let bigram_counts: BigramCount<SymbolTableEntryId, usize> = input_tokens
        .iter()
        .flat_map(|(_k, v)| v)
        .map(|v| -> &[SymbolTableEntryId] { &v })
        .collect();

    let mut bigram_counts: Vec<_> = bigram_counts
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    bigram_counts.sort_by_key(|e| e.1);
    bigram_counts.reverse();

    for (k, c) in bigram_counts.iter() {
        println!(
            "{}|{} {}",
            symbol_table_id_to_string(&symboltable, k.0, "^", "$"),
            symbol_table_id_to_string(&symboltable, k.1, "^", "$"),
            c
        )
    }

    // Take the commonest bigram and remove it

    // Remove any symbols that don't occur in the input.
}
