use std::collections::{BTreeMap, BTreeSet, HashMap};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use markovian_core::symbol::{SymbolTable, SymbolTableEntry};
use markovian_core::symbolize_new::{
    symbolize_new,
    symbolize_new2,
    symbolize_new3,
    BackBacking,
    FwdBacking,
    ProgressiveIndex,
    ProgressiveIndexState,
    SymbolId,
    SymbolizeState,
};

pub fn symbolize_10as(c: &mut Criterion) {
    let to_symbolize_40as = &[
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a',
    ];

    let to_symbolize_20as = &[
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a',
    ];

    let to_symbolize_10as = &['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'];
    let to_symbolize_5as = &['a', 'a', 'a', 'a', 'a'];

    let mut symbol_index = HashMap::new();
    symbol_index.insert(&['a', 'a'][..], SymbolId(1));
    symbol_index.insert(&['a'][..], SymbolId(2));

    c.bench_function("symbolize 40a", |b| {
        b.iter(|| {
            symbolize_new(black_box(to_symbolize_40as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 20a", |b| {
        b.iter(|| {
            symbolize_new(black_box(to_symbolize_20as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 10a", |b| {
        b.iter(|| {
            symbolize_new(black_box(to_symbolize_10as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 5a", |b| {
        b.iter(|| {
            symbolize_new(black_box(to_symbolize_5as), black_box(&symbol_index));
        })
    });
}

pub fn symbolize_10as_trie(c: &mut Criterion) {
    let to_symbolize_40as = &[
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a', 'a', 'a',
    ];

    let to_symbolize_20as = &[
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
        'a', 'a',
    ];

    let to_symbolize_10as = &['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'];
    let to_symbolize_5as = &['a', 'a', 'a', 'a', 'a'];

    let mut symbol_index = HashMap::new();
    symbol_index.insert(&['a', 'a'][..], ProgressiveIndexState::Present(SymbolId(1)));
    symbol_index.insert(&['a'][..], ProgressiveIndexState::Present(SymbolId(2)));

    c.bench_function("symbolize 40a T", |b| {
        b.iter(|| {
            symbolize_new2(black_box(to_symbolize_40as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 20a T", |b| {
        b.iter(|| {
            symbolize_new2(black_box(to_symbolize_20as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 10a T", |b| {
        b.iter(|| {
            symbolize_new2(black_box(to_symbolize_10as), black_box(&symbol_index));
        })
    });

    c.bench_function("symbolize 5a T", |b| {
        b.iter(|| {
            symbolize_new2(black_box(to_symbolize_5as), black_box(&symbol_index));
        })
    });
}

pub fn symbolize_as_old(c: &mut Criterion) {
    let mut symbol_table = SymbolTable::new();

    // let start = result.add(SymbolTableEntry::Start).unwrap();
    // let end = result.add(SymbolTableEntry::End).unwrap();
    symbol_table.add(SymbolTableEntry::Single('a')).unwrap();
    symbol_table
        .add(SymbolTableEntry::Compound(vec!['a', 'a']))
        .unwrap();

    let to_symbolize_5as = &['a', 'a', 'a', 'a', 'a'];
    let to_symbolize_10as = &['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'];

    c.bench_function("symbolize 10a OLD", |b| {
        b.iter(|| symbol_table.symbolifications(black_box(to_symbolize_10as)))
    });

    c.bench_function("symbolize 5a OLD", |b| {
        b.iter(|| symbol_table.symbolifications(black_box(to_symbolize_5as)))
    });
}

pub fn generate_common_symbols() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    //let s = include_str!("../../resources/web2.txt");
    let s = include_str!("../../resources/Moby_Names_M_lc.txt");
    let mut single_symbols: BTreeMap<u8, usize> = BTreeMap::new();
    let mut bigrams: BTreeMap<(u8, u8), usize> = BTreeMap::new();
    let mut trigrams: BTreeMap<(u8, u8, u8), usize> = BTreeMap::new();
    let mut words: Vec<Vec<u8>> = vec![];
    for l in s.lines() {
        let b = l.as_bytes();
        for bb in b {
            *single_symbols.entry(*bb).or_default() += 1;
        }
        if b.len() > 1 {
            for i in 0..b.len() - 1 {
                *bigrams.entry((b[i], b[i + 1])).or_default() += 1;
            }
        }
        if b.len() > 2 {
            for i in 0..b.len() - 2 {
                *trigrams.entry((b[i], b[i + 1], b[i + 2])).or_default() += 1;
            }
        }
        words.push(b.to_vec());
    }

    // println!("symbols = {:?}", single_symbols);
    // println!("bigrams = {:?}", bigrams);
    println!("|words| = {}", words.len());
    println!("|symbols| = {}", single_symbols.len());
    println!("|bigrams| = {}", bigrams.len());
    println!("|trigrams| = {}", trigrams.len());

    // Keep anything with a count higher than 40
    let single_symbols = single_symbols
        .into_iter()
        .filter(|(s, c)| *c > 40)
        .map(|(s, c)| vec![s])
        .collect::<Vec<_>>();

    // Keep the 20 best bigrams
    let mut bigrams = bigrams.into_iter().collect::<Vec<_>>();
    bigrams.sort_by_key(|(s, c)| *c);
    bigrams.reverse();
    let mut bigrams = bigrams[0..20]
        .iter()
        .map(|((a, b), c)| vec![*a, *b])
        .collect::<Vec<_>>();

    // Keep the 20 best trigrams
    let mut trigrams = trigrams.into_iter().collect::<Vec<_>>();
    trigrams.sort_by_key(|(s, c)| *c);
    trigrams.reverse();
    let mut trigrams = trigrams[0..20]
        .iter()
        .map(|((p, q, r), c)| vec![*p, *q, *r])
        .collect::<Vec<_>>();

    let mut symbols: Vec<Vec<u8>> = single_symbols;
    symbols.append(&mut bigrams);
    symbols.append(&mut trigrams);

    (symbols, words)
}

pub fn symbolize_realistic(c: &mut Criterion) {
    let (symbols, words) = generate_common_symbols();

    // Create the "usual" map
    let mut symbol_index = HashMap::new();
    for (i, s) in symbols.iter().enumerate() {
        symbol_index.insert(&s[..], SymbolId(i + 1));
    }

    c.bench_function("realistic NEW", |b| {
        b.iter(|| {
            for w in &words {
                black_box(symbolize_new(&w[..], &symbol_index));
            }
        })
    });

    let mut symbol_table = SymbolTable::new();
    symbol_table.add_symbols(symbols.clone()).unwrap();

    c.bench_function("realistic OLD", |b| {
        b.iter(|| {
            for w in &words {
                black_box(symbol_table.symbolifications(&w[..]));
            }
        })
    });

    // TODO: This is not strictly correct!
    let mut symbol_index = HashMap::new();
    for (i, s) in symbols.iter().enumerate() {
        symbol_index.insert(&s[..], ProgressiveIndexState::Present(SymbolId(i + 1)));
    }

    c.bench_function("realistic T", |b| {
        b.iter(|| {
            for w in &words {
                black_box(symbolize_new2(&w[..], &symbol_index));
            }
        })
    });

    let max_length = words.iter().map(|w| w.len()).max().unwrap();
    let mut fwd = FwdBacking::allocate(max_length);
    let mut back = BackBacking::allocate(max_length);

    let mut state = SymbolizeState {
        forward: fwd.state(),
        back: back.state(),
    };

    c.bench_function("realistic T3", |b| {
        b.iter(|| {
            for w in &words {
                black_box(symbolize_new3(&w[..], &symbol_index, &mut state, |x| {
                    black_box(x);
                }));
            }
        })
    });
}

pub fn core_only(c: &mut Criterion) {
    let (symbols, words) = generate_common_symbols();

    // TODO: This is not strictly correct!
    let mut symbol_index = HashMap::new();
    for (i, s) in symbols.iter().enumerate() {
        symbol_index.insert(&s[..], ProgressiveIndexState::Present(SymbolId(i + 1)));
    }

    c.bench_function("realistic T", |b| {
        b.iter(|| {
            for w in &words {
                black_box(symbolize_new2(&w[..], &symbol_index));
            }
        })
    });
}

criterion_group!(benches, symbolize_realistic);
// criterion_group!(benches, symbolize_10as, symbolize_as_old, symbolize_10as_trie);
criterion_main!(benches);
