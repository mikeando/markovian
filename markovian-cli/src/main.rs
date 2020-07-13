use log::{debug, info, log_enabled};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use structopt::StructOpt;

use markovian_core::markov_model::MarkovModel;
use markovian_core::symbol::Symbol;

fn raw_symbolify_word(s: &str) -> Vec<Symbol> {
    s.as_bytes().iter().cloned().map(Symbol::Char).collect()
}

fn reduce_symbols<T>(v: Vec<T>, key: (&T, &T), value: &T) -> Vec<T> 
where T: Eq + Clone
{
    let mut result: Vec<T> = vec![];
    let mut skip = false;
    for i in 0..v.len() - 1 {
        if skip {
            skip = false;
            continue;
        }
        if (&v[i] == key.0) && (&v[i + 1] == key.1) {
            result.push(value.clone());
            skip = true;
        } else {
            result.push(v[i].clone());
        }
    }
    if !skip {
        v.last()
            .iter()
            .cloned()
            .for_each(|s: &T| result.push(s.clone()));
    }
    result
}

trait AppendToVec {
    fn append_to_vec(&self, v:&mut Vec<u8>);
}

impl AppendToVec for Symbol {
    fn append_to_vec(&self, v:&mut Vec<u8>) {
        match self {
            Symbol::Start => {
                v.push(b'^');
            }
            Symbol::End => {
                v.push(b'$');
            }
            Symbol::Char(c) => {
                v.push(*c);
            }
            Symbol::Compound(cs) => {
                v.append(&mut cs.clone());
            }
        }
    }

}

fn symbols_to_vec<T>(v: &[T], insert_sep: bool) -> Vec<u8>
where T: AppendToVec
{
    let mut result: Vec<u8> = vec![];
    let mut is_start = true;
    for s in v {
        if insert_sep {
            if is_start {
                is_start = false
            } else {
                result.push(b'|')
            }
        }
        s.append_to_vec(&mut result);
    }
    result
}

fn symbolrefs_to_vec<T>(v: &[&T], insert_sep: bool) -> Vec<u8>
where T: AppendToVec
{
    let mut result: Vec<u8> = vec![];
    let mut is_start = true;
    for s in v {
        if insert_sep {
            if is_start {
                is_start = false
            } else {
                result.push(b'|')
            }
        }
        s.append_to_vec(&mut result);
    }
    result
}

fn symbols_to_word<T:AppendToVec>(v: &[T], insert_sep: bool) -> String {
    String::from_utf8_lossy(&symbols_to_vec(v, insert_sep)).to_string()
}

fn symbolrefs_to_word<T:AppendToVec>(v: &[&T], insert_sep: bool) -> String {
    String::from_utf8_lossy(&symbolrefs_to_vec(v, insert_sep)).to_string()
}

fn get_symbol_counts<T>(input_names: &[Vec<T>]) -> BTreeMap<T, usize>
where T: Clone + Ord
{
    let mut symbol_counts: BTreeMap<T, usize> = BTreeMap::new();
    for name in input_names {
        for w in name {
            *symbol_counts.entry(w.clone()).or_insert(0) += 1;
        }
    }
    symbol_counts
}

fn get_bigram_counts<T>(input_names: &[Vec<T>]) -> BTreeMap<(T, T), usize>
where T: Clone + Ord
{
    let mut bigram_counts: BTreeMap<(T, T), usize> = BTreeMap::new();
    for name in input_names {
        for w in name.windows(2) {
            *bigram_counts
                .entry((w[0].clone(), w[1].clone()))
                .or_insert(0) += 1;
        }
    }
    bigram_counts
}

fn get_trigram_counts<T>(input_names: &[Vec<T>]) -> BTreeMap<(T, T, T), usize>
where T: Clone + Ord
{
    let mut trigram_counts: BTreeMap<(T, T, T), usize> = BTreeMap::new();
    for name in input_names {
        for w in name.windows(3) {
            *trigram_counts
                .entry((w[0].clone(), w[1].clone(), w[2].clone()))
                .or_insert(0) += 1;
        }
    }
    trigram_counts
}

#[derive(Debug, StructOpt)]
#[structopt(name = "markovian", about = "Markov based name generator.")]
struct Opt {
    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    name_file: Vec<PathBuf>,

    /// Number of bigrams to reduce to their own symbols
    #[structopt(short, long, default_value = "-1")]
    bigram_reduce_count: i32,

    /// Verbosity level
    #[structopt(short, long, default_value = "0")]
    verbose: i32,

    /// prefix
    #[structopt(short, long)]
    prefix: Option<String>,

    /// print symbol separators
    #[structopt(long)]
    print_separators: bool,

    /// number of names to print
    #[structopt(short, long, default_value = "20")]
    count: u32,

    /// generate in reverse
    #[structopt(long)]
    reverse: bool,

    /// print symbol similarity information
    #[structopt(long)]
    print_symbol_similarity: bool,
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

fn print_model_summary(model: &MarkovModel) {
    use log::Level::Info;
    info!("total contexts = {:?}", model.contexts.len());
    let mut h: BTreeSet<Symbol> = BTreeSet::new();
    for k in model.contexts.keys() {
        for s in k {
            h.insert(s.clone());
        }
    }
    if log_enabled!(Info) {
        info!("total unique symbols = {:?}", h.len());
        for s in &h {
            info!("{:?}|", symbols_to_word(&[s.clone()], false));
        }
        info!("");
    }
}

fn replace<T>(haystack: &[T], needle: &[T], replacement: &[T]) -> Vec<T>
where
    T: Clone + PartialEq,
{
    let mut result: Vec<T> = vec![];
    let nh = haystack.len();
    let nn = needle.len();
    let mut i = 0;
    loop {
        if i >= nh {
            break;
        }
        if i + nn <= nh && haystack[i..i + nn] == needle[..] {
            result.extend_from_slice(&replacement[..]);
            i += nn;
        } else {
            result.push(haystack[i].clone());
            i += 1;
        }
    }
    result
}

fn combine_rare_symbols(input_names: Vec<Vec<Symbol>>) -> Vec<Vec<Symbol>> {
    let mut result = input_names;
    use log::Level::Info;
    // Hunt for symbols that are only ever used before another
    // These can be reduced to a combined symbol cheaply

    debug!("Looking for rare symbols....");

    loop {
        let bigram_counts = get_bigram_counts(&result);
        let mut symbol_to_followers: BTreeMap<Symbol, BTreeSet<Symbol>> = BTreeMap::new();
        for s in bigram_counts.keys() {
            symbol_to_followers
                .entry(s.0.clone())
                .or_insert_with(BTreeSet::new)
                .insert(s.1.clone());
        }
        let mut reduce_count: i32 = 0;
        for (k, ss) in &symbol_to_followers {
            if ss.len() == 1 {
                reduce_count += 1;
                let s = ss.iter().next().unwrap();
                let replacement = Symbol::Compound(symbolrefs_to_vec(&[k, s], false));
                debug!(
                    "Replacing {:?} with {}",
                    symbolrefs_to_word(&[k, s], true),
                    symbolrefs_to_word(&[&replacement], false),
                );
                result = result
                    .iter()
                    .map(|n| replace(&n, &[k.clone(), s.clone()], &[replacement.clone()]))
                    .collect();
            } else if ss.len() > 3 {
                debug!("{} => {}", symbolrefs_to_word(&[k], false), ss.len());
            } else if log_enabled!(Info) {
                let sss: Vec<String> = ss.iter().map(|k| symbolrefs_to_word(&[k], false)).collect();
                debug!("{} => {:?}", symbolrefs_to_word(&[k], false), sss);
            }
        }
        debug!("XA Pass reduced {} bigrams", reduce_count);
        if reduce_count == 0 {
            break;
        }
    }

    // Hunt for symbols that are only ever used after another
    // These can be reduced to a combined symbol cheaply
    loop {
        let bigram_counts = get_bigram_counts(&result);
        let mut symbol_to_prefix: BTreeMap<Symbol, BTreeSet<Symbol>> = BTreeMap::new();
        for s in bigram_counts.keys() {
            symbol_to_prefix
                .entry(s.1.clone())
                .or_insert_with(BTreeSet::new)
                .insert(s.0.clone());
        }
        let mut reduce_count: i32 = 0;
        for (k, ss) in &symbol_to_prefix {
            if ss.len() == 1 {
                reduce_count += 1;
                let s = ss.iter().next().unwrap();
                let replacement = Symbol::Compound(symbolrefs_to_vec(&[s, k], false));
                debug!(
                    "Replacing {:?} with {}",
                    symbolrefs_to_word(&[s, k], true),
                    symbolrefs_to_word(&[&replacement], false),
                );
                //TODO: These clones happen every iteration!
                result = result
                    .iter()
                    .map(|n| replace(&n, &[s.clone(), k.clone()], &[replacement.clone()]))
                    .collect();
            } else if ss.len() > 3 {
                debug!("{} => {}", ss.len(), symbolrefs_to_word(&[k], false));
            } else if log_enabled!(Info) {
                let sss: Vec<String> = ss.iter().map(|k| symbolrefs_to_word(&[k], false)).collect();
                debug!("{:?} => {}", sss, symbolrefs_to_word(&[k], false));
            }
        }
        debug!("AX Pass reduced {} bigrams", reduce_count);
        if reduce_count == 0 {
            break;
        }
    }

    debug!("Done looking for rare symbols....");
    result
}

fn get_sorted_bigram_counts<T>(input_names: &[Vec<T>]) -> Vec<((T, T), usize)> 
where T: Ord + Clone
{
    let bigram_counts_map = get_bigram_counts(&input_names);
    let mut bigram_counts: Vec<_> = bigram_counts_map.into_iter().collect();
    bigram_counts.sort_by_key(|e| e.1);
    bigram_counts.reverse();
    bigram_counts
}

//TODO: This really should take a "model" and convert it to a new "model"
//      rather than working on raw data.
fn convert_common_bigrams_to_symbols(
    input_names: Vec<Vec<Symbol>>,
    bigram_reduce_count: i32,
) -> Vec<Vec<Symbol>> {
    use log::Level::Debug;
    if bigram_reduce_count <= 0 {
        return input_names;
    }

    let mut input_names = input_names;
    for _i in 0..bigram_reduce_count {
        let bigram_counts = get_sorted_bigram_counts(&input_names);
        let most_common_bigram = bigram_counts[0].0.clone();
        info!("Removing bigram {:?}", most_common_bigram);
        let s = Symbol::Compound(symbolrefs_to_vec(
            &[&most_common_bigram.0, &most_common_bigram.1],
            false,
        ));
        input_names = input_names
            .into_iter()
            .map(|v| reduce_symbols(v, (&most_common_bigram.0, &most_common_bigram.1), &s))
            .collect();
        if log_enabled!(Debug) {
            let bigram_counts = get_sorted_bigram_counts(&input_names);
            debug!("--------");
            for x in &bigram_counts[0..10] {
                debug!(
                    "{:?} {:?}",
                    symbolrefs_to_word(&[&(x.0).0, &(x.0).1], false),
                    x.1
                );
            }
        }
    }
    input_names
}

fn log_symbol_counts<T>(input_names: &[Vec<T>])
where T: Clone + Ord + AppendToVec
{
    let symbol_counts = get_symbol_counts(&input_names);
    let mut symbol_counts: Vec<_> = symbol_counts.into_iter().collect();
    symbol_counts.sort_by_key(|e| e.1);
    symbol_counts.reverse();
    for x in &symbol_counts {
        info!("{:?} {:?}", symbolrefs_to_word(&[&x.0], false), x.1);
    }
}

fn log_bigram_counts<T>(input_names: &[Vec<T>])
where T: Clone + Ord + AppendToVec
{
    use std::cmp::min;

    let bigram_counts = get_bigram_counts(&input_names);
    let mut bigram_counts: Vec<_> = bigram_counts.into_iter().collect();
    bigram_counts.sort_by_key(|e| e.1);
    bigram_counts.reverse();
    for x in &bigram_counts[0..min(10, bigram_counts.len())] {
        //let v = [(x.0).0, (x.0).1];
        info!(
            "{:?} {:?}",
            symbolrefs_to_word(&[&(x.0).0, &(x.0).1], false),
            x.1
        );
    }
}

fn log_trigram_counts<T>(input_names: &[Vec<T>])
where T: Clone + Ord + AppendToVec
{
    use std::cmp::min;

    let trigram_counts = get_trigram_counts(&input_names);
    let mut trigram_counts: Vec<_> = trigram_counts.into_iter().collect();
    trigram_counts.sort_by_key(|e| e.1);
    trigram_counts.reverse();
    for x in &trigram_counts[0..min(10, trigram_counts.len())] {
        info!(
            "{:?} {:?}",
            symbolrefs_to_word(&[&(x.0).0, &(x.0).1, &(x.0).2], false),
            x.1
        );
    }
}

fn print_similar_symbols(input_names: &Vec<Vec<Symbol>>) {
    let symbols = get_symbol_counts(input_names);
    let symbols: Vec<Symbol> = symbols.keys().cloned().collect();
    let bigrams = get_bigram_counts(input_names);
    let reprs: Vec<_> = symbols
        .iter()
        .map(|s| repr_for_symbol(s, &symbols, &bigrams))
        .collect();
    println!("{} symbols", symbols.len());
    for (s, rr) in symbols.iter().zip(reprs.iter()) {
        println!("{:?} => {:?}", s, rr);
    }

    let mut compared: Vec<(f32, Symbol, Symbol)> = vec![];
    for (ra, sa) in reprs.iter().zip(symbols.iter()) {
        print!("{} ", symbols_to_word(&[sa.clone()], false));
        for (rb, sb) in reprs.iter().zip(symbols.iter()) {
            let c = dot(ra, rb);
            print!("{0:.2} ", c);
            if sa < sb {
                compared.push((c, sa.clone(), sb.clone()));
            }
        }
        println!()
    }

    for (ra, sa) in reprs.iter().zip(symbols.iter()) {
        use std::cmp::Ordering;
        let (c, _sb) = reprs
            .iter()
            .zip(symbols.iter())
            .filter(|(_rb, sb)| sb != &sa)
            .map(|(rb, sb)| (dot(ra, rb), sb))
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal))
            .unwrap();

        if c > 0.2 {
            let sbs: Vec<_> = reprs
                .iter()
                .zip(symbols.iter())
                .filter(|(_rb, sb)| sb != &sa)
                .map(|(rb, sb)| (dot(ra, rb), sb))
                .filter(|(cb, _sb)| *cb > 0.8 * c)
                .map(|(cb, sb)| (cb, symbols_to_word(&[sb.clone()], false)))
                .collect();
            println!("{} {:?}", symbols_to_word(&[sa.clone()], false), sbs);
        }
    }

    compared.sort_by_key(|e| (-e.0 * 1000.0) as i64);
    for x in &compared[0..100] {
        println! {"{} ~ {} : {}", symbolrefs_to_word(&[&x.1], false), symbolrefs_to_word(&[&x.2], false), x.0}
    }
}

fn main() {
    use log::Level::Info;

    let opt = Opt::from_args();
    debug!("{:?}", opt);

    let verbose: i32 = opt.verbose;
    setup_logging(verbose);

    let input_names_raw: Vec<String> = opt
        .name_file
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

    //let input_names_raw = std::fs::read_to_string(opt.name_file).unwrap();

    let order = 3;
    let print_sep = verbose >= 1 || opt.print_separators;

    info!("Loading word list...");

    let input_names: Vec<Vec<Symbol>> = input_names_raw
        .iter()
        .map(|s| raw_symbolify_word(s))
        .collect();
    //println!("{:?}", symbolify_word(input_names[0], order));

    info!("{:?} raw entries", input_names_raw.len());
    info!("{:?} entries", input_names.len());

    if log_enabled!(Info) {
        log_symbol_counts(&input_names);
        log_bigram_counts(&input_names);
        log_trigram_counts(&input_names);
    }

    //Good to get rid of the rare cases well before we hit any other optimisations.
    let input_names = combine_rare_symbols(input_names);
    let input_names = convert_common_bigrams_to_symbols(input_names, opt.bigram_reduce_count);
    let input_names = combine_rare_symbols(input_names);

    if opt.print_symbol_similarity {
        print_similar_symbols(&input_names);
    }

    let mut model = MarkovModel::new(order);

    info!("Populating model...");
    for mut name in input_names {
        if opt.reverse {
            name.reverse()
        }
        model.add(&name);
    }

    print_model_summary(&model);

    let mut rng = rand::thread_rng();
    info!("Sampling model...");

    for _ in 0..opt.count {
        let mut symbs = if let Some(prefix) = opt.prefix.as_ref() {
            let mut p = model.convert_string_to_symbols(prefix);
            if opt.reverse {
                p.reverse();
            }
            model.sample_starting_with(&p, &mut rng)
        } else {
            model.sample_starting_with(&[], &mut rng)
        };
        if opt.reverse {
            symbs.reverse();
        }
        let result = symbols_to_word(&symbs, print_sep);
        if print_sep {
            println!("{:?}", result);
        } else {
            println!("{}", result);
        }
    }
}

pub fn normalize(v: Vec<f32>) -> Vec<f32> {
    let norm_v = v.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_v == 0.0 {
        return v;
    }
    v.into_iter().map(|v| v / norm_v).collect()
}

fn repr_for_symbol(
    s: &Symbol,
    symbols: &[Symbol],
    bigrams: &BTreeMap<(Symbol, Symbol), usize>,
) -> Vec<f32> {
    let mut v: Vec<usize> = vec![];
    for ss in symbols {
        v.push(*bigrams.get(&(ss.clone(), s.clone())).unwrap_or(&0))
    }
    for ss in symbols {
        v.push(*bigrams.get(&(s.clone(), ss.clone())).unwrap_or(&0))
    }
    let v: Vec<f32> = v
        .into_iter()
        .map(|v| if v >= 4 { v as f32 } else { 0.0 })
        .collect();
    normalize(v)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(aa, bb)| aa * bb).sum()
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_replace() {
        let v = vec!["A", "B"];
        let r = replace(&v, &v, &["X"]);
        assert_eq!(r, vec!["X"]);
    }

    #[test]
    fn test_symbol_to_vector_from_bigram_map() {
        let a = || Symbol::Char(b'a');
        let b = || Symbol::Char(b'b');
        let c = || Symbol::Char(b'c');
        // The vector for a symbols identity is the normalized
        // count of how often it appears after a given character, and how many times
        // it appears before a given character.

        let mut bigrams: BTreeMap<(Symbol, Symbol), usize> = BTreeMap::new();
        bigrams.insert((a(), a()), 20);
        bigrams.insert((a(), b()), 10);
        bigrams.insert((c(), a()), 30);
        bigrams.insert((c(), b()), 10);

        let v = normalize(vec![2., 0., 3., 2., 1., 0.]);
        let symbols = vec![a(), b(), c()];

        fn cmp_eq(a: f32, b: f32) -> bool {
            (a - b).abs() <= 1e-5 * 0.5 * (a.abs() + b.abs())
        }
        fn cmp_vec_eq(aa: &[f32], bb: &[f32]) -> bool {
            aa.iter().zip(bb.iter()).all(|(a, b)| cmp_eq(*a, *b))
        }
        let r = repr_for_symbol(&a(), &symbols, &bigrams);
        assert!(cmp_vec_eq(&v, &r), "{:?} != {:?}", v, r);

        let rr: Vec<_> = symbols
            .iter()
            .map(|s| repr_for_symbol(s, &symbols, &bigrams))
            .collect();

        for ra in &rr {
            println!("{:?}", ra);
        }

        for ra in &rr {
            for rb in &rr {
                print!("{} ", dot(ra, rb))
            }
            println!()
        }
    }
}
