use std::collections::BTreeMap;
use std::path::PathBuf;

use log::info;
use markovian_core::analyser::AnalyserWrapper;
use markovian_core::renderer::{
    renderer_for_char_with_separator,
    renderer_for_u8_with_separator,
    RenderChar,
    RenderU8,
    Renderer,
    RendererId,
    SymbolIdRenderer,
    SymbolIdRendererChar,
    SymbolIdRendererU8,
};
use markovian_core::symbol::{
    SymbolTable,
    SymbolTableEntry,
    SymbolTableEntryId,
    SymbolTableWrapper,
    TableEncoding,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub enum Command {
    Print(PrintCommand),
    Generate(GenerateCommand),
    Symbolify(SymbolifyCommand),
    Analyse(AnalyseCommand),
    Improve(ImproveSymbolTableCommand),
}

#[derive(Debug, StructOpt)]
pub struct PrintCommand {
    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,
}

#[derive(Debug, StructOpt)]
pub struct GenerateCommand {
    /// Input files
    #[structopt(short, long, parse(from_os_str))]
    input: Vec<PathBuf>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Encoding for table
    #[structopt(short, long, parse(try_from_str = table_encoding_from_string), default_value="string")]
    encoding: TableEncoding,
}

#[derive(Debug, StructOpt)]
pub struct SymbolifyCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

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

#[derive(Debug, StructOpt)]
pub struct AnalyseCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Files to analyse
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,
}

#[derive(Debug, StructOpt)]
pub struct ImproveSymbolTableCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    symboltable: PathBuf,

    /// Files to analyse
    #[structopt(parse(from_os_str))]
    input_file: Vec<PathBuf>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Number of symbol combine steps to perform
    #[structopt(long, default_value = "50")]
    combine_steps: usize,

    /// Weight percent for symbol trimming
    #[structopt(long)]
    symbol_trim_percent_weight: Option<f64>,

    /// Min weight for symbol trimming
    #[structopt(long)]
    symbol_trim_min_weight: Option<f64>,
}

impl ImproveSymbolTableCommand {
    pub fn get_symbol_trim_mode(&self) -> Result<SymbolTrimmingMode, String> {
        match (self.symbol_trim_min_weight, self.symbol_trim_percent_weight) {
            (None, None) => Ok(SymbolTrimmingMode::None),
            (None, Some(w)) => Ok(SymbolTrimmingMode::MaxSumPercent(w)),
            (Some(w), None) => Ok(SymbolTrimmingMode::MaxWeight(w)),
            (Some(_), Some(_)) => Err(
                "Can only specify one of symbol_trim_percent_weight and symbol_trim_min_weight"
                    .to_string(),
            ),
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
    Err(format!(
        "Unknown table encoding '{}' valid values are 'bytes', 'string'",
        v
    ))
}

fn command_symbolify(cmd: &SymbolifyCommand) {
    info!("SYMBOLIFY: {:?}", cmd);
    let data = std::fs::read(&cmd.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();
    info!("encoding: {}", symboltable.encoding().encoding_name());
    info!("max symbol id: {}", symboltable.max_symbol_id());

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
            .map(|ids| renderer.render(&ids))
            .collect::<Result<Vec<_>, _>>();
        match result {
            Ok(v) => {
                println!("{} => {:?}", s, v);
            }
            Err(e) => println!("error: {:?} caused {:?}", s, e),
        }
    }
}

fn command_print(p: &PrintCommand) {
    info!("PRINT: {:?}", p);
    let data = std::fs::read(&p.input).unwrap();
    let decoded: SymbolTableWrapper = bincode::deserialize(&data).unwrap();
    println!("encoding: {}", decoded.encoding().encoding_name());
    println!("max symbol id: {}", decoded.max_symbol_id());

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

pub fn build_symbol_table(encoding: &TableEncoding, input_tokens: &[String]) -> SymbolTableWrapper {
    match encoding {
        TableEncoding::Bytes => {
            let mut symbol_table = SymbolTable::<u8>::new();
            for k in input_tokens {
                for b in k.as_bytes() {
                    symbol_table.add(SymbolTableEntry::Single(*b)).unwrap();
                }
            }
            SymbolTableWrapper::Bytes(symbol_table)
        }
        TableEncoding::String => {
            let mut symbol_table = SymbolTable::<char>::new();
            for k in input_tokens {
                for c in k.chars() {
                    symbol_table.add(SymbolTableEntry::Single(c)).unwrap();
                }
            }
            SymbolTableWrapper::String(symbol_table)
        }
    }
}

fn command_generate(g: &GenerateCommand) {
    // TODO: Add potential for transformations and customizable min length, and trimming,
    // Load the text
    // TODO: options to allow to_lowercase in the lambda
    let input_tokens: Vec<String> = crate::utils::read_input_lines(&g.input, |s| s);

    println!("using {} input strings", input_tokens.len());

    let symbol_table = build_symbol_table(&g.encoding, &input_tokens);
    println!("found {} symbols", symbol_table.max_symbol_id());

    let encoded: Vec<u8> = bincode::serialize(&symbol_table).unwrap();
    std::fs::write(&g.output, &encoded).unwrap();
    println!("wrote {} ", g.output.display());
}

fn command_analyse(x: &AnalyseCommand) {
    // Load the symboltable
    let data = std::fs::read(&x.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();

    // Load the text
    // TODO: options to allow to_lowercase in the lambda
    let input_tokens: Vec<String> = crate::utils::read_input_lines(&x.input_file, |s| s);

    let analyser = AnalyserWrapper::new(symboltable, input_tokens);
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

fn print_analyser_summary(analyser: &AnalyserWrapper) {
    let symbolization_counts: BTreeMap<usize, usize> = analyser.get_symbolization_ways_counts();
    println!();
    for (len, count) in symbolization_counts {
        println!("{} entries each symbolize exactly {} ways", count, len)
    }
    println!();

    let symbol_counts = analyser.get_ordered_symbol_counts();

    println!("Top 10 Individual symbol counts");
    {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        for (k, v) in symbol_counts.iter().take(10) {
            println!("{} {:?}", symbol_renderer.render(*k).unwrap(), v);
        }
    }

    println!("--- Top 10 bigrams ---");

    let bigram_counts = analyser.get_bigram_counts();

    let mut ordered_bigram_counts = bigram_counts
        .iter()
        .filter(|(bigram, _weight)| {
            bigram.0 != SymbolTableEntryId(0) && bigram.1 != SymbolTableEntryId(1)
        })
        .collect::<Vec<_>>();
    ordered_bigram_counts.sort_by(|a, b| (a.1).1.total_cmp(&(b.1).1));
    {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        for (k, c) in ordered_bigram_counts.iter().take(10) {
            println!(
                "{}|{} {}",
                symbol_renderer.render(k.0).unwrap(),
                symbol_renderer.render(k.1).unwrap(),
                c.1
            )
        }
    }

    println!("--- Last 10 bigrams ---");
    ordered_bigram_counts.reverse();
    {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        for (k, c) in ordered_bigram_counts.iter().take(10) {
            println!(
                "{}|{} {}",
                symbol_renderer.render(k.0).unwrap(),
                symbol_renderer.render(k.1).unwrap(),
                c.1
            )
        }
    }
    //TODO: Sometimes we get bigrams with a negative value on them .. why? (Cancellation errors?)
}

pub trait ImproveSymbolTableCallbacks {
    fn on_init(&self, analyser: &AnalyserWrapper);
    fn on_iteration_after_merge(
        &self,
        analyser: &AnalyserWrapper,
        bigram: (SymbolTableEntryId, SymbolTableEntryId),
        count: f64,
        new_symbol: SymbolTableEntryId,
    );
    fn on_end(&self, analyser: &AnalyserWrapper);
}

#[derive(PartialEq, Debug)]
pub enum SymbolTrimmingMode {
    None,
    MaxSumPercent(f64),
    MaxWeight(f64),
}

pub fn improve_symbol_table<CallBack: ImproveSymbolTableCallbacks>(
    symboltable: SymbolTableWrapper,
    input_tokens: Vec<String>,
    callback: CallBack,
    n_combine_steps: usize,
    symbol_trimming_mode: SymbolTrimmingMode,
) -> SymbolTableWrapper {
    let mut analyser = AnalyserWrapper::new(symboltable, input_tokens);

    callback.on_init(&analyser);

    if symbol_trimming_mode != SymbolTrimmingMode::None {
        let renderer = analyser.get_symbol_renderer("^", "$");

        let mut symbol_counts = analyser.get_symbol_counts();
        let symbol_weight_sum: f64 = symbol_counts.iter().map(|(_s, c)| c.1).sum();

        let cut_weight = match symbol_trimming_mode {
            SymbolTrimmingMode::None => unreachable!(),
            SymbolTrimmingMode::MaxSumPercent(percent_trim) => {
                symbol_counts.sort_by(|a, b| (a.1).1.partial_cmp(&(b.1).1).unwrap());
                let cum_sum_weights: Vec<f64> = symbol_counts
                    .iter()
                    .scan(0.0, |acc, &x| {
                        *acc += (x.1).1;
                        Some(*acc)
                    })
                    .collect();

                let symbol_cum_weight_max = (percent_trim / 100.0) * symbol_weight_sum;
                let idx = cum_sum_weights
                    .binary_search_by(|a| a.partial_cmp(&symbol_cum_weight_max).unwrap());
                let idx = match idx {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };
                if idx > 0 {
                    (symbol_counts[idx - 1].1).1
                } else {
                    0.0
                }
            }
            SymbolTrimmingMode::MaxWeight(w) => w,
        };

        info!(
            "Purging symbols with weights less than {} of total weight = {}",
            cut_weight, symbol_weight_sum
        );
        let mut symbols_and_weights_to_purge: Vec<(SymbolTableEntryId, String, f64)> =
            symbol_counts
                .into_iter()
                .filter_map(|(s, (_c, w))| {
                    if w <= cut_weight {
                        Some((s, renderer.render(s).unwrap(), w))
                    } else {
                        None
                    }
                })
                .collect();
        symbols_and_weights_to_purge.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        for s in &symbols_and_weights_to_purge {
            info!(
                "Purging {:?} with weight {:?} = {}%",
                s.1,
                s.2,
                100.0 * s.2 / symbol_weight_sum
            );
        }
        let symbols_to_purge: Vec<SymbolTableEntryId> = symbols_and_weights_to_purge
            .into_iter()
            .map(|(sid, _, _)| sid)
            .collect();
        drop(renderer);
        analyser.purge_symbols(&symbols_to_purge);
    }

    for _i in 0..n_combine_steps {
        // Take the commonest bigram and remove it
        let bigram_counts = analyser.get_bigram_counts();

        let q = bigram_counts
            .iter()
            .filter(|(bigram, _weight)| {
                bigram.0 != SymbolTableEntryId(0) && bigram.1 != SymbolTableEntryId(1)
            })
            .max_by(|a, b| (a.1).1.total_cmp(&(b.1).1));
        let bigram = *q.unwrap().0;
        let count = (q.unwrap().1).1;
        let new_symbol = analyser.concatenate_symbols(bigram.0, bigram.1);
        callback.on_iteration_after_merge(&analyser, bigram, count, new_symbol);

        // TODO: Remove unused symbols
        // TODO: Scan for "unique" prefix / suffix pairs in the bigrams e.g. we can often replace q|u with qu without
        // cost. This is especially important when working with UTF-8 as bytes.
    }

    callback.on_end(&analyser);
    analyser.get_symbol_table()
}

struct CommandImproveSymbolTableCallbacks {}

impl ImproveSymbolTableCallbacks for CommandImproveSymbolTableCallbacks {
    fn on_init(&self, analyser: &AnalyserWrapper) {
        print_analyser_summary(&analyser);
    }

    fn on_iteration_after_merge(
        &self,
        analyser: &AnalyserWrapper,
        bigram: (SymbolTableEntryId, SymbolTableEntryId),
        count: f64,
        _new_symbol: SymbolTableEntryId,
    ) {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        info!(
            "Commonest bigram is {}|{} (count={})",
            symbol_renderer.render(bigram.0).unwrap(),
            symbol_renderer.render(bigram.1).unwrap(),
            count,
        );
    }

    fn on_end(&self, analyser: &AnalyserWrapper) {
        print_analyser_summary(&analyser);
    }
}

fn command_improve_symbol_table(cmd: &ImproveSymbolTableCommand) {
    // Load the symboltable
    let data = std::fs::read(&cmd.symboltable).unwrap();
    let symboltable: SymbolTableWrapper = bincode::deserialize(&data).unwrap();

    // Load the text
    // TODO: options to allow to_lowercase in the lambda
    let input_tokens: Vec<String> = crate::utils::read_input_lines(&cmd.input_file, |s| s);

    let callbacks = CommandImproveSymbolTableCallbacks {};
    let symbol_table: SymbolTableWrapper = improve_symbol_table(
        symboltable,
        input_tokens,
        callbacks,
        cmd.combine_steps,
        cmd.get_symbol_trim_mode().unwrap(),
    );
    let encoded: Vec<u8> = bincode::serialize(&symbol_table).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}

pub fn run(cmd: &Command) {
    match cmd {
        Command::Print(p) => command_print(p),
        Command::Generate(g) => command_generate(g),
        Command::Symbolify(s) => command_symbolify(s),
        Command::Analyse(x) => command_analyse(x),
        Command::Improve(x) => command_improve_symbol_table(x),
    }
}
