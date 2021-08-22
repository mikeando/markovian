use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use log::info;
use markovian_core::analyser::AnalyserWrapper;
use markovian_core::generator_wrapper::GeneratorWrapper;
use markovian_core::symbol::SymbolTableEntryId;
use markovian_core::symboltable_wrapper::SymbolTableWrapper;
use structopt::StructOpt;

use super::SymbolTrimmingMode;

#[derive(Debug, StructOpt)]
pub struct ImproveSymbolTableCommand {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

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
    fn on_iteration_merge_deterministic_pair(
        &self,
        analyser: &AnalyserWrapper,
        bigram: (SymbolTableEntryId, SymbolTableEntryId),
        count: (f64, f64),
        new_symbol: SymbolTableEntryId,
    );
    fn on_iteration_after_merge(
        &self,
        analyser: &AnalyserWrapper,
        bigram: (SymbolTableEntryId, SymbolTableEntryId),
        count: f64,
        new_symbol: SymbolTableEntryId,
    );
    fn on_end(&self, analyser: &AnalyserWrapper);
}

struct CommandImproveSymbolTableCallbacks {}

impl ImproveSymbolTableCallbacks for CommandImproveSymbolTableCallbacks {
    fn on_init(&self, analyser: &AnalyserWrapper) {
        print_analyser_summary(analyser);
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

    fn on_iteration_merge_deterministic_pair(
        &self,
        analyser: &AnalyserWrapper,
        bigram: (SymbolTableEntryId, SymbolTableEntryId),
        count: (f64, f64),
        _new_symbol: SymbolTableEntryId,
    ) {
        let symbol_renderer = analyser.get_symbol_renderer("^", "$");
        info!(
            "Deterministic bigram {}|{} (count={}/{})",
            symbol_renderer.render(bigram.0).unwrap(),
            symbol_renderer.render(bigram.1).unwrap(),
            count.0,
            count.1,
        );
    }

    fn on_end(&self, analyser: &AnalyserWrapper) {
        print_analyser_summary(analyser);
    }
}

pub fn command_improve_symbol_table(cmd: &ImproveSymbolTableCommand) {
    let data = std::fs::read(&cmd.input).unwrap();
    let generator: GeneratorWrapper = bincode::deserialize(&data).unwrap();

    let (symbol_table, _ngrams) = generator.into_symbol_table_and_ngrams();

    // Load the text
    // TODO: options to allow to_lowercase in the lambda
    let input_tokens: Vec<String> = crate::utils::read_input_lines(&cmd.input_file, |s| s);

    let callbacks = CommandImproveSymbolTableCallbacks {};
    let symbol_table: SymbolTableWrapper = improve_symbol_table(
        symbol_table,
        input_tokens,
        callbacks,
        cmd.combine_steps,
        cmd.get_symbol_trim_mode().unwrap(),
    );

    // Any existing ngrams are invalid - so discard them
    let generator = GeneratorWrapper::from_ngrams(symbol_table, BTreeMap::new());

    let encoded: Vec<u8> = bincode::serialize(&generator).unwrap();
    std::fs::write(&cmd.output, &encoded).unwrap();
    println!("wrote {} ", cmd.output.display());
}

pub fn remove_low_weight_symbols(
    analyser: &mut AnalyserWrapper,
    cut_weight: f64,
    symbol_weight_sum: f64,
) {
    let symbol_counts = analyser.get_symbol_counts();
    let renderer = analyser.get_symbol_renderer("^", "$");

    let mut symbols_and_weights_to_purge: Vec<(SymbolTableEntryId, String, f64)> = symbol_counts
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

pub fn improve_symbol_table<CallBack: ImproveSymbolTableCallbacks>(
    symboltable: SymbolTableWrapper,
    input_tokens: Vec<String>,
    callback: CallBack,
    n_combine_steps: usize,
    symbol_trimming_mode: SymbolTrimmingMode,
) -> SymbolTableWrapper {
    let mut analyser = AnalyserWrapper::new(symboltable, input_tokens);

    callback.on_init(&analyser);

    let symbol_weight_sum: f64 = analyser.get_symbol_counts().iter().map(|(_s, c)| c.1).sum();

    let cut_weight = match symbol_trimming_mode {
        SymbolTrimmingMode::None => 0.0,
        SymbolTrimmingMode::MaxSumPercent(percent_trim) => {
            let mut symbol_counts = analyser.get_symbol_counts();
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

    if cut_weight > 0.0 {
        info!(
            "Purging symbols with weights less than {} of total weight = {}",
            cut_weight, symbol_weight_sum
        );
        remove_low_weight_symbols(&mut analyser, cut_weight, symbol_weight_sum);
    }

    for _i in 0..n_combine_steps {
        let renderer = analyser.get_symbol_renderer("^", "$");

        // Take the commonest bigram and remove it
        let bigram_counts = analyser.get_bigram_counts();

        // Does the analyser contain this information already?
        let mut prefix_counts: BTreeMap<SymbolTableEntryId, BTreeMap<SymbolTableEntryId, f64>> =
            BTreeMap::new();
        let mut suffix_counts: BTreeMap<SymbolTableEntryId, BTreeMap<SymbolTableEntryId, f64>> =
            BTreeMap::new();
        for ((a, b), w) in bigram_counts {
            if w.1 > 0.0 {
                *prefix_counts
                    .entry(*a)
                    .or_insert_with(BTreeMap::new)
                    .entry(*b)
                    .or_insert(0.0) += w.1;
                *suffix_counts
                    .entry(*b)
                    .or_insert_with(BTreeMap::new)
                    .entry(*a)
                    .or_insert(0.0) += w.1;
            }
        }

        let mut bigrams_to_merge = HashMap::new();

        let det_thresh = 0.95;

        for (p, next) in prefix_counts {
            if p.0 == 0 || p.0 == 1 {
                continue;
            }
            let sum: f64 = next.iter().map(|(_k, v)| *v).sum();
            let largest = next
                .iter()
                .max_by(|(_k1, v1), (_k2, v2)| v1.partial_cmp(v2).unwrap())
                .unwrap();
            if *largest.0 == SymbolTableEntryId(0) || *largest.0 == SymbolTableEntryId(1) {
                continue;
            }
            if *largest.1 > det_thresh * sum {
                bigrams_to_merge.insert((p, *largest.0), (*largest.1, sum));
            }
        }
        for (s, next) in suffix_counts {
            if s.0 == 0 || s.0 == 1 {
                continue;
            }
            let sum: f64 = next.iter().map(|(_k, v)| *v).sum();
            let largest = next
                .iter()
                .max_by(|(_k1, v1), (_k2, v2)| v1.partial_cmp(v2).unwrap())
                .unwrap();
            if *largest.0 == SymbolTableEntryId(0) || *largest.0 == SymbolTableEntryId(1) {
                continue;
            }
            if *largest.1 > det_thresh * sum {
                bigrams_to_merge.insert((*largest.0, s), (*largest.1, sum));
            }
        }

        drop(renderer);

        // Combine the most common bigram
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

        for ((a, b), count) in bigrams_to_merge {
            let new_symbol = analyser.concatenate_symbols(a, b);
            callback.on_iteration_merge_deterministic_pair(&analyser, (a, b), count, new_symbol)
        }

        if cut_weight > 0.0 {
            remove_low_weight_symbols(&mut analyser, cut_weight, symbol_weight_sum);
        }
    }

    callback.on_end(&analyser);
    analyser.get_symbol_table()
}
