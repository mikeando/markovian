use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::language;

mod language_manipulation {
    use crate::language::raw::*;
    use super::*;

    #[derive(Debug)]
    pub struct ExtractSequence<T> {
        sequence:Vec<SymbolOrLiteral<T>>
    }

    #[derive(Debug)]
    pub struct FactorPrefix<T> {
        symbol:Symbol,
        prefix:Vec<SymbolOrLiteral<T>>
    }

    #[derive(Debug)]
    pub struct FactorSuffix<T> {
        symbol:Symbol,
        suffix:Vec<SymbolOrLiteral<T>>
    }

    #[derive(Debug)]
    pub enum LanguageManipulation<T> {
        ExtractSequence( ExtractSequence<T> ),
        FactorPrefix( FactorPrefix<T> ),
        FactorSuffix( FactorSuffix<T> ),
    }

    #[derive(Debug)]
    pub struct Proposal<T> {
        pub action: LanguageManipulation<T>,
        pub expected_improvement: i64,
    }

    pub trait Proposer<T> {
        fn get_proposal(&self, l: &Language<T>) -> Option<Proposal<T>>;
    }

    impl <T> LanguageManipulation<T> 
    where 
        T: Clone + PartialEq
    {
        pub fn apply(&self, l: &Language<T>) -> Language<T> {
            match &self {
                LanguageManipulation::ExtractSequence( action ) => { action.apply(l) }
                LanguageManipulation::FactorPrefix( action ) => { action.apply(l) }
                LanguageManipulation::FactorSuffix( action ) => { action.apply(l) }
            }
        }
    }

    impl <T> ExtractSequence<T> 
    where 
        T: Clone + PartialEq
    {
        pub fn apply(&self, l: &Language<T>) -> Language<T> {
            let s = new_symbol(l);
        
            let mut entries: Vec<Production<T>> = vec![];
            //First add the new rule
            entries.push(Production {
                from: s.clone(),
                to: self.sequence.clone(),
                weight: 1,
            });
        
            // Now we need to find and replace occurrences of this in
            // all the productions
            for p in &l.entries {
                entries.push(Production {
                    from: p.from.clone(),
                    weight: p.weight,
                    to: replace_subsequence(&p.to, &self.sequence, SymbolOrLiteral::Symbol(s.clone())),
                })
            }
        
            Language { entries }
        }
    }

    impl <T> FactorPrefix<T> 
        where 
            T: Clone + PartialEq
    {
        pub fn apply(&self, l: &Language<T>) -> Language<T> {
            let s = new_symbol(l);
            let mut unchanged = vec![];
            let mut with_prefix = vec![];
            let mut weight: u32 = 0;
            for e in &l.entries {
                if e.from != self.symbol {
                    unchanged.push(e.clone());
                    continue;
                }
                if ! e.to.starts_with(&self.prefix) {
                    unchanged.push(e.clone());
                    continue;
                }
                with_prefix.push(
                    Production {
                        from: s.clone(),
                        weight: e.weight,
                        to: e.to[self.prefix.len()..].to_vec()
                    }
                );
                weight += e.weight;
            }
            let p = Production {
                from: self.symbol.clone(),
                weight,
                to: [self.prefix.clone(), vec![SymbolOrLiteral::Symbol(s)]].concat(),
            };
            let mut entries = unchanged;
            entries.push(p);
            entries.append(&mut with_prefix);
            Language {
                entries
            }
        }
    }

    impl <T> FactorSuffix<T> 
        where T: Clone + PartialEq
    {
        pub fn apply(&self, l: &Language<T>) -> Language<T> {
            let s = new_symbol(l);
            let mut unchanged = vec![];
            let mut with_prefix = vec![];
            let mut weight: u32 = 0;
            for e in &l.entries {
                if e.from != self.symbol {
                    unchanged.push(e.clone());
                    continue;
                }
                if ! e.to.ends_with(&self.suffix) {
                    unchanged.push(e.clone());
                    continue;
                }
                with_prefix.push(
                    Production {
                        from: s.clone(),
                        weight: e.weight,
                        to: e.to[..(e.to.len() - self.suffix.len())].to_vec()
                    }
                );
                weight += e.weight;
            }
            let p = Production {
                from: self.symbol.clone(),
                weight,
                to: [vec![SymbolOrLiteral::Symbol(s)], self.suffix.clone()].concat()
            };
            let mut entries = unchanged;
            entries.push(p);
            entries.append(&mut with_prefix);
            Language {
                entries
            }
        }
    }

    pub struct ExtractSequenceProposer;

    impl <T> Proposer<T> for ExtractSequenceProposer 
        where T: Ord + Clone
    {
        fn get_proposal(&self, l: &Language<T>) -> Option<Proposal<T>> {
            let mut m: BTreeMap<&[SymbolOrLiteral<T>], usize> = BTreeMap::new();
            for p in &l.entries {
                let mm: BTreeMap<&[SymbolOrLiteral<T>], usize> = substring_count(&p.to);
                merge(&mut m, mm)
            }
        
            // Now find the best subsequence
            let m = substring_value(m);
            //Find the one with the highest value.
            m.into_iter()
                .max_by_key(|e| e.1)
                .filter(|v| v.1 > 0)
                .map(|(a,b)| 
                    Proposal {
                        action: LanguageManipulation::ExtractSequence(
                            ExtractSequence {
                                sequence:a.to_vec()
                            }
                        ),
                        expected_improvement: b as i64
                    }
                )
        }

    }

    pub struct FactorPrefixProposer;

    impl<T> Proposer<T> for FactorPrefixProposer
        where T: Ord + Clone
    {
        fn get_proposal(&self, l: &Language<T>) -> Option<Proposal<T>> {
            let mut m:BTreeMap<(&Symbol,&[SymbolOrLiteral<T>]), usize> = BTreeMap::new();
            for e in &l.entries {
                for i in 0..e.to.len() {
                    *m.entry((&e.from, &e.to[..i+1]) ).or_insert(0) += 1;
                }
            }
            // Replacing rules i=1..N
            //   A -> w_i : [P] [S_i]
            //  total cost = N * len(P) + sum( len(S_i) )
            //with
            //  A -> sum(w_i) : [P] [S]
            //  S -> w_i : [S_i]
            // total cost = len(P) + 1 + sum( len(S_i) )
            //
            // so total saving is (N-1) * len(P) - 1
            fn score<T>(suffix:&[SymbolOrLiteral<T>], count:usize) -> i64 {
                (suffix.len() as i64) * (count as i64 - 1) - 1
            }

            m.into_iter()
                .map(|(k,v)| (k, score(k.1, v)) )
                .filter( |(k,v)| *v>0 )
                .max_by_key(|e| e.1)
                .map( |((symbol, prefix), s)|
                    Proposal {
                        action: LanguageManipulation::FactorPrefix(
                            FactorPrefix { symbol: symbol.clone(), prefix: prefix.to_vec() }, ),
                        expected_improvement: s,
                    }
                )
        }
    }

    pub struct FactorSuffixProposer;

    impl<T> Proposer<T> for FactorSuffixProposer
        where T: Ord + Clone
    {
        fn get_proposal(&self, l: &Language<T>) -> Option<Proposal<T>> {
            let mut m:BTreeMap<(&Symbol,&[SymbolOrLiteral<T>]), usize> = BTreeMap::new();
            for e in &l.entries {
                for i in 0..e.to.len() {
                    *m.entry((&e.from, &e.to[i..]) ).or_insert(0) += 1;
                }
            }
            // Replacing rules i=1..N
            //   A -> w_i : [P] [S_i]
            //  total cost = N * len(P) + sum( len(S_i) )
            //with
            //  A -> sum(w_i) : [P] [S]
            //  S -> w_i : [S_i]
            // total cost = len(P) + 1 + sum( len(S_i) )
            //
            // so total saving is (N-1) * len(P) - 1
            fn score<T>(suffix:&[SymbolOrLiteral<T>], count:usize) -> i64 {
                (suffix.len() as i64) * (count as i64 - 1) - 1
            }

            m.into_iter()
                .map(|(k,v)| (k, score(k.1, v)) )
                .filter( |(k,v)| *v>0 )
                .max_by_key(|e| e.1)
                .map( |((symbol, suffix), s)|
                    Proposal {
                        action: LanguageManipulation::FactorSuffix(
                            FactorSuffix { symbol: symbol.clone(), suffix: suffix.to_vec() }, ),
                        expected_improvement: s,
                    }
                )
        }
    }

}




// The aim of this module it to provide tools to
// find the most common sub sequences of a set of sequences.
// The idea being that if we consider a language, then the
// each production is a list of symbols.
// We should consider extracting a subsequence from all productions
// if it reduces the total size of the language. And we'll approach this
// from a greedy manner, iteratively extracting the common sub-sequence
// that reduces our language size by the most.

pub fn substring_count<T>(v: &[T]) -> BTreeMap<&[T], usize>
where
    T: Ord,
{
    let mut m: BTreeMap<&[T], usize> = BTreeMap::new();
    for i in 0..v.len() {
        for j in (i + 1)..=v.len() {
            *m.entry(&v[i..j]).or_insert(0) += 1
        }
    }
    m
}

pub fn substring_value<T>(m: BTreeMap<&[T], usize>) -> BTreeMap<&[T], usize>
where
    T: Ord,
{
    m.into_iter()
        .map(|(k, v)| (k, (v - 1) * (k.len() - 1)))
        .collect()
}

fn shatter_literal(
    e: language::raw::SymbolOrLiteral<String>,
) -> Vec<language::raw::SymbolOrLiteral<char>> {
    match e {
        language::raw::SymbolOrLiteral::Symbol(s) => {
            vec![language::raw::SymbolOrLiteral::Symbol(s)]
        }
        language::raw::SymbolOrLiteral::Literal(l) => {
            l.0.chars()
                .map(language::raw::SymbolOrLiteral::literal)
                .collect()
        }
    }
}

fn unshatter_vec_symbols(
    v: &[language::raw::SymbolOrLiteral<char>],
) -> Vec<language::raw::SymbolOrLiteral<String>> {
    use language::raw::SymbolOrLiteral;

    let mut result = vec![];
    let mut pending_string: Option<String> = None;
    for cs in v {
        match cs {
            SymbolOrLiteral::Symbol(s) => {
                if let Some(p) = pending_string {
                    result.push(SymbolOrLiteral::literal(p));
                    pending_string = None;
                }
                result.push(SymbolOrLiteral::Symbol(s.clone()));
            }
            SymbolOrLiteral::Literal(c) => {
                if pending_string.is_none() {
                    pending_string = Some(format!("{}", c.0));
                } else {
                    pending_string = Some(format!("{}{}", pending_string.unwrap(), c.0));
                }
            }
        }
    }
    if let Some(p) = pending_string {
        result.push(SymbolOrLiteral::literal(p));
    }

    result
}

fn shatter_production(p: language::raw::Production<String>) -> language::raw::Production<char> {
    language::raw::Production {
        to: p.to.into_iter().flat_map(shatter_literal).collect(),
        weight: p.weight,
        from: p.from,
    }
}

fn unshatter_production(p: language::raw::Production<char>) -> language::raw::Production<String> {
    language::raw::Production {
        to: unshatter_vec_symbols(&p.to),
        weight: p.weight,
        from: p.from,
    }
}

// Take a language that uses strings and convert it to a
// language that uses characters.
pub fn shatter_language(m: language::raw::Language<String>) -> language::raw::Language<char> {
    language::raw::Language {
        entries: m.entries.into_iter().map(shatter_production).collect(),
    }
}

pub fn unshatter_language(m: language::raw::Language<char>) -> language::raw::Language<String> {
    language::raw::Language {
        entries: m.entries.into_iter().map(unshatter_production).collect(),
    }
}

fn merge<A>(m: &mut BTreeMap<A, usize>, mm: BTreeMap<A, usize>)
where
    A: Ord,
{
    for e in mm {
        *m.entry(e.0).or_insert(0) += e.1
    }
}

pub fn new_symbol<T>(l: &language::raw::Language<T>) -> language::raw::Symbol {
    let mut all_symbols = BTreeSet::new();
    for p in &l.entries {
        all_symbols.insert(p.from.clone());
        for s in &p.to {
            if let Some(s) = s.as_symbol() {
                all_symbols.insert(s.clone());
            }
        }
    }

    let mut c: usize = 0;
    loop {
        let s = language::raw::Symbol(format!("s_{}", c));
        if !all_symbols.contains(&s) {
            return s;
        }
        c += 1;
    }
}

pub fn replace_subsequence<T>(input: &[T], needle: &[T], replacement: T) -> Vec<T>
where
    T: Clone + PartialEq,
{
    if needle.is_empty() {
        return input.to_vec();
    }
    if input.len() < needle.len() {
        return input.to_vec();
    }

    let mut result: Vec<T> = vec![];

    let mut s: usize = 0;
    while s + needle.len() <= input.len() {
        let sl = &input[s..s + needle.len()];
        if sl != needle {
            result.push(sl[0].clone());
            s += 1;
        } else {
            result.push(replacement.clone());
            s += needle.len()
        }
    }
    while s < input.len() {
        result.push(input[s].clone());
        s += 1;
    }
    result
}

pub fn propose_extract_sequence<T>(
    l: &language::raw::Language<T>
) -> Option<(Vec<language::raw::SymbolOrLiteral<T>>, usize)>
where
    T: Ord + Clone + std::fmt::Debug,
{
    use language::raw::SymbolOrLiteral;
    let mut m: BTreeMap<&[SymbolOrLiteral<T>], usize> = BTreeMap::new();
    for p in &l.entries {
        let mm: BTreeMap<&[SymbolOrLiteral<T>], usize> = substring_count(&p.to);
        merge(&mut m, mm)
    }

    // Now find the best subsequence
    let m = substring_value(m);
    //Find the one with the highest value.
    m.into_iter()
        .max_by_key(|e| e.1)
        .filter(|v| v.1 > 0)
        .map(|(a,b)| (a.to_vec(), b) )
}

pub fn remove_best_subseq_from_language<T>(
    l: &language::raw::Language<T>,
) -> Option<language::raw::Language<T>>
where
    T: Ord + Clone + std::fmt::Debug,
{
    let proposer = language_manipulation::FactorPrefixProposer;
    let p1 = proposer.get_proposal(l);
    println!("Suggested prefix removal {:?}", p1);

    let proposer = language_manipulation::FactorSuffixProposer;
    let p2 = proposer.get_proposal(l);
    println!("Suggested suffix removal {:?}", p2);

    use language_manipulation::Proposer;
    let proposer = language_manipulation::ExtractSequenceProposer;
    let p3 = proposer.get_proposal(l);
    println!("Suggested sequence removal {:?}", p3);

    [p1,p2,p3].into_iter()
            .filter_map(|f| f.as_ref())
            .max_by_key(|p| p.expected_improvement)
            .map( |p| { println!("Applying {:?}", p); p.action.apply(l) } )
}

pub fn extract_sequence<T>(
    l: &language::raw::Language<T>,
    seq: &[language::raw::SymbolOrLiteral<T>],
) -> language::raw::Language<T>
where 
    T: Ord + Clone + std::fmt::Debug,
{
    use language::raw::Language;
    use language::raw::Production;
    use language::raw::SymbolOrLiteral;

    let s = new_symbol(l);

    let mut entries: Vec<Production<T>> = vec![];
    //First add the new rule
    entries.push(Production {
        from: s.clone(),
        to: seq.to_vec(),
        weight: 1,
    });

    // Now we need to find and replace occurrences of this in
    // all the productions
    for p in &l.entries {
        entries.push(Production {
            from: p.from.clone(),
            weight: p.weight,
            to: replace_subsequence(&p.to, seq, SymbolOrLiteral::Symbol(s.clone())),
        })
    }

    Language { entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn simple_substring_count() {
        let v = "abc";
        let n = substring_count(v.as_bytes());
        let r: Vec<(&str, usize)> = n
            .iter()
            .map(|(k, v)| (std::str::from_utf8(k).unwrap(), *v))
            .collect();
        assert_eq!(
            r,
            vec![
                ("a", 1),
                ("ab", 1),
                ("abc", 1),
                ("b", 1),
                ("bc", 1),
                ("c", 1),
            ]
        )
    }

    #[test]
    pub fn substring_count_repeat() {
        let v = "ababc";
        let n = substring_count(v.as_bytes());
        let r: Vec<(&str, usize)> = n
            .iter()
            .map(|(k, v)| (std::str::from_utf8(k).unwrap(), *v))
            .collect();
        assert_eq!(
            r,
            vec![
                ("a", 2),
                ("ab", 2),
                ("aba", 1),
                ("abab", 1),
                ("ababc", 1),
                ("abc", 1),
                ("b", 2),
                ("ba", 1),
                ("bab", 1),
                ("babc", 1),
                ("bc", 1),
                ("c", 1),
            ]
        )
    }

    #[test]
    pub fn test_substring_value() {
        let v = "ababc";
        let n = substring_count(v.as_bytes());
        let n = substring_value(n);
        let r: Vec<(&str, usize)> = n
            .iter()
            .map(|(k, v)| (std::str::from_utf8(k).unwrap(), *v))
            .collect();
        assert_eq!(
            r,
            vec![
                ("a", 0),
                ("ab", 1),
                ("aba", 0),
                ("abab", 0),
                ("ababc", 0),
                ("abc", 0),
                ("b", 0),
                ("ba", 0),
                ("bab", 0),
                ("babc", 0),
                ("bc", 0),
                ("c", 0),
            ]
        )
    }

    fn prod<T>(
        from: &str,
        weight: u32,
        to: Vec<language::raw::SymbolOrLiteral<T>>,
    ) -> language::raw::Production<T> {
        language::raw::Production {
            from: language::raw::Symbol::new(from),
            weight,
            to,
        }
    }

    fn s<T>(k: &str) -> language::raw::SymbolOrLiteral<T> {
        language::raw::SymbolOrLiteral::symbol(k)
    }

    fn l(k: char) -> language::raw::SymbolOrLiteral<char> {
        language::raw::SymbolOrLiteral::literal(k)
    }

    fn sl(k: &str) -> language::raw::SymbolOrLiteral<String> {
        language::raw::SymbolOrLiteral::literal(k)
    }

    #[test]
    pub fn test_shatter_language() {
        use language::raw::Language;
        let lang: Language<String> = Language {
            entries: vec![
                prod("A", 1, vec![s("P1"), s("X"), sl("abc"), sl("def")]),
                prod("B", 1, vec![s("X"), sl("pqr"), s("S1")]),
                prod("C", 1, vec![sl("ghi")]),
            ],
        };
        let lang2 = shatter_language(lang);
        assert_eq!(
            lang2,
            Language {
                entries: vec![
                    prod(
                        "A",
                        1,
                        vec![
                            s("P1"),
                            s("X"),
                            l('a'),
                            l('b'),
                            l('c'),
                            l('d'),
                            l('e'),
                            l('f')
                        ]
                    ),
                    prod("B", 1, vec![s("X"), l('p'), l('q'), l('r'), s("S1")]),
                    prod("C", 1, vec![l('g'), l('h'), l('i')]),
                ]
            }
        )
    }

    #[test]
    pub fn test_unshatter_language() {
        use language::raw::Language;
        let lang: Language<char> = Language {
            entries: vec![
                prod(
                    "A",
                    1,
                    vec![
                        s("P1"),
                        s("X"),
                        l('a'),
                        l('b'),
                        l('c'),
                        l('d'),
                        l('e'),
                        l('f'),
                    ],
                ),
                prod("B", 1, vec![s("X"), l('p'), l('q'), l('r'), s("S1")]),
                prod("C", 1, vec![l('g'), l('h'), l('i')]),
            ],
        };
        let lang2 = unshatter_language(lang);
        assert_eq!(
            lang2,
            Language {
                entries: vec![
                    prod("A", 1, vec![s("P1"), s("X"), sl("abcdef")]),
                    prod("B", 1, vec![s("X"), sl("pqr"), s("S1")]),
                    prod("C", 1, vec![sl("ghi")]),
                ]
            }
        )
    }

    #[test]
    pub fn test_remove_from_language() {
        use language::raw::Language;
        let lang: Language<char> = Language {
            entries: vec![
                prod("A", 1, vec![s("P1"), s("X"), l('a'), l('b')]),
                prod("B", 1, vec![s("X"), l('a'), l('b'), s("S1")]),
                prod("C", 1, vec![s("P2"), s("X"), l('a'), l('b'), s("S2")]),
            ],
        };
        let mod_lang = remove_best_subseq_from_language(&lang).unwrap();
        assert_eq!(
            mod_lang,
            Language {
                entries: vec![
                    prod("s_0", 1, vec![s("X"), l('a'), l('b')]),
                    prod("A", 1, vec![s("P1"), s("s_0")]),
                    prod("B", 1, vec![s("s_0"), s("S1")]),
                    prod("C", 1, vec![s("P2"), s("s_0"), s("S2")]),
                ]
            }
        )
    }

    mod subsequence {
        use super::super::*;

        #[test]
        pub fn test_replace_subsequence() {
            let v: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
            let w = replace_subsequence(&v, &[3, 4, 5], 9);
            assert_eq!(w, vec![1, 2, 9, 6]);
        }

        #[test]
        pub fn test_empty_input() {
            let v: Vec<i32> = vec![];
            let w = replace_subsequence(&v, &[3, 4, 5], 9);
            assert_eq!(w, vec![]);
        }

        #[test]
        pub fn test_empty_needle() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[], 9);
            assert_eq!(w, vec![1, 2, 3]);
        }

        #[test]
        pub fn test_short_input() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[1, 2, 3, 4], 9);
            assert_eq!(w, vec![1, 2, 3]);
        }

        #[test]
        pub fn test_equal() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[1, 2, 3], 9);
            assert_eq!(w, vec![9]);
        }

        #[test]
        pub fn test_no_replace() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[9, 9, 9], 9);
            assert_eq!(w, vec![1, 2, 3]);
        }

        #[test]
        pub fn test_replace_at_end() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[2, 3], 9);
            assert_eq!(w, vec![1, 9]);
        }

        #[test]
        pub fn test_replace_at_start() {
            let v: Vec<i32> = vec![1, 2, 3];
            let w = replace_subsequence(&v, &[1, 2], 9);
            assert_eq!(w, vec![9, 3]);
        }

        #[test]
        pub fn test_replace_multiple() {
            let v: Vec<i32> = vec![1, 2, 1, 2, 1, 2];
            let w = replace_subsequence(&v, &[1, 2], 9);
            assert_eq!(w, vec![9, 9, 9]);
        }

        #[test]
        pub fn test_replace_overlapping() {
            let v: Vec<i32> = vec![1, 2, 1, 2, 1, 2, 1];
            let w = replace_subsequence(&v, &[1, 2, 1], 9);
            assert_eq!(w, vec![9, 2, 9]);
        }
    }

    pub fn language_size<T>(l:&language::raw::Language<T>) -> usize {
        l.entries.iter().map(|e| e.to.len()).sum()
    }

    #[test]
    pub fn test_derive_names() {
        println!("CARGO_MANIFEST_DIR={}", env!("CARGO_MANIFEST_DIR"));
        let names = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), "../resources/all_names_lc.text");
        println!("names={}", names);
        let names = std::fs::read_to_string(names).unwrap();
        for name in names.lines() {
            println!("{}", name);
        }
        let s = language::raw::Symbol::new("N");
        let entries:Vec<language::raw::Production<String>> = 
            names.lines().map( |name| 
                language::raw::Production {
                    from:s.clone(),
                    weight:1,
                    to:vec!{language::raw::SymbolOrLiteral::literal(name),
                }
        }).collect();
        let l = language::raw::Language {
            entries
        };
        let mut ll = shatter_language(l);
        for _i in 0..500 {
            println!("==== language size = {} ====", language_size(&ll));
            ll = remove_best_subseq_from_language(&ll).unwrap();
        }
        println!("==== language size = {} ====", language_size(&ll));

        let l = unshatter_language(ll);
        for e in l.entries {
            println!("{:?}",e);
        }
    }
}
