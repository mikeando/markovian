use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use crate::renderer::{SymbolIdRenderer, SymbolIdRendererChar, SymbolIdRendererU8};
use crate::symbol::{
    shortest_symbolifications,
    SymbolTable,
    SymbolTableEntry,
    SymbolTableEntryId,
    SymbolTableWrapper,
};
use crate::tombstone::{self, TombstoneList};
use crate::vecutils::select_by_lowest_value;

struct WordEntry<T> {
    raw_word: String,
    word: Vec<T>,
    symbolifications: Vec<Vec<SymbolTableEntryId>>,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
struct WordId(tombstone::Token);

struct A2<T> {
    words: TombstoneList<WordEntry<T>>,
    symbol_to_word_map: BTreeMap<SymbolTableEntryId, BTreeSet<WordId>>,
    symbol_counts: BTreeMap<SymbolTableEntryId, (usize, f64)>,
    bigram_counts: BTreeMap<(SymbolTableEntryId, SymbolTableEntryId), (usize, f64)>,
}

impl<T> A2<T> {
    pub fn new() -> A2<T> {
        A2 {
            words: TombstoneList::new(),
            symbol_to_word_map: BTreeMap::new(),
            symbol_counts: BTreeMap::new(),
            bigram_counts: BTreeMap::new(),
        }
    }

    pub fn add_word(&mut self, w: WordEntry<T>) -> WordId {
        let id = WordId(self.words.add(w));
        // We've just inserted it, so we know its there.
        let w = self.words.get(&id.0).unwrap();
        let weight = 1.0 / w.symbolifications.len() as f64;
        for s in &w.symbolifications {
            for ss in s {
                let e = self.symbol_counts.entry(*ss).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += weight;
                self.symbol_to_word_map.entry(*ss).or_default().insert(id);
            }
            let n = s.len();
            if n < 1 {
                continue;
            }
            for i in 0..(n - 1) {
                let t = (s[i], s[i + 1]);
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += weight;
            }
            {
                let t = (SymbolTableEntryId(0), s[0]);
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += weight;
            }
            {
                let t = (s[n - 1], SymbolTableEntryId(1));
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += weight;
            }
        }
        id
    }

    pub fn sub_word(&mut self, id: &WordId) -> WordEntry<T> {
        let w = self.words.remove(&id.0).unwrap();
        let weight = 1.0 / w.symbolifications.len() as f64;
        for s in &w.symbolifications {
            for ss in s {
                self.symbol_to_word_map.entry(*ss).or_default().remove(id);
                let e = self.symbol_counts.entry(*ss).or_insert((0, 0.0));
                e.0 -= 1;
                e.1 -= weight;
                if e.0 == 0 {
                    self.symbol_counts.remove(ss);
                }
            }
            let n = s.len();
            if n <= 1 {
                continue;
            }
            for i in 0..(n - 1) {
                let t = (s[i], s[i + 1]);
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 -= 1;
                e.1 -= weight;
                if e.0 == 0 {
                    self.bigram_counts.remove(&t);
                }
            }
            {
                let t = (SymbolTableEntryId(0), s[0]);
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 -= 1;
                e.1 -= weight;
                if e.0 == 0 {
                    self.bigram_counts.remove(&t);
                }
            }
            {
                let t = (s[n - 1], SymbolTableEntryId(1));
                let e = self.bigram_counts.entry(t).or_insert((0, 0.0));
                e.0 -= 1;
                e.1 -= weight;
                if e.0 == 0 {
                    self.bigram_counts.remove(&t);
                }
            }
        }
        w
    }
}

// TODO: Would be nice to be able to return a slice when we can rather than
// allocating up a complete buffer, but the lifetimes on the required associated types
// are tricky.
pub trait WordToToken<T> {
    fn convert(&self, s: &'_ str) -> Vec<T>;
}

pub struct Analyser<T, Tokenizer>
where
    T: Ord + Clone,
{
    pub symbol_table: SymbolTable<T>,
    tokenizer: Tokenizer,
    a: A2<T>,
}

impl<T, Tokenizer> Analyser<T, Tokenizer>
where
    Tokenizer: WordToToken<T>,
    T: Eq + Clone + Ord + Debug,
{
    //TODO: Should this take ownership of the list of strings too?
    pub fn new(
        symbol_table: SymbolTable<T>,
        tokenizer: Tokenizer,
        word_list: Vec<String>,
    ) -> Analyser<T, Tokenizer> {
        let mut a = A2::new();
        for raw_word in word_list {
            let word = tokenizer.convert(&raw_word);
            let symbolifications = shortest_symbolifications(&symbol_table, &word);
            let entry = WordEntry {
                raw_word,
                word,
                symbolifications,
            };
            a.add_word(entry);
        }

        Analyser {
            symbol_table,
            tokenizer,
            a,
        }
    }

    pub fn word_list(&self) -> Vec<&str> {
        self.a.words.iter().map(|v| v.raw_word.as_str()).collect()
    }

    pub fn shortest_symbolifications_str(&self, word: &str) -> Vec<Vec<SymbolTableEntryId>> {
        self.shortest_symbolifications(self.tokenizer.convert(word).as_slice())
    }

    pub fn shortest_symbolifications(&self, v: &[T]) -> Vec<Vec<SymbolTableEntryId>> {
        let v = self.symbol_table.symbolifications(v);
        select_by_lowest_value(&v, &|s: &Vec<SymbolTableEntryId>| s.len())
    }

    pub fn get_symbolisations(&self) -> Vec<(&str, Vec<Vec<SymbolTableEntryId>>)> {
        self.a
            .words
            .iter()
            .map(|entry| {
                (
                    entry.raw_word.as_str(),
                    self.shortest_symbolifications_str(&entry.raw_word),
                )
            })
            .collect()
    }

    pub fn get_symbol_counts(&self) -> Vec<(SymbolTableEntryId, (usize, f64))> {
        self.a.symbol_counts.iter().map(|(&k, &v)| (k, v)).collect()
    }

    pub fn get_bigram_counts(
        &self,
    ) -> &BTreeMap<(SymbolTableEntryId, SymbolTableEntryId), (usize, f64)> {
        &self.a.bigram_counts
    }

    pub fn concatenate_symbols(
        &mut self,
        a: SymbolTableEntryId,
        b: SymbolTableEntryId,
    ) -> SymbolTableEntryId {
        let mut v = Vec::<T>::new();

        // Take the commonest bigram and remove it
        self.symbol_table.append_to_vec(a, &mut v).unwrap();
        self.symbol_table.append_to_vec(b, &mut v).unwrap();
        let symbol_table_entry = SymbolTableEntry::Compound(v);
        let new_id = self.symbol_table.add(symbol_table_entry).unwrap();

        let words_with_a = self.a.symbol_to_word_map.get(&a);
        let words_with_b = self.a.symbol_to_word_map.get(&b);
        let word_ids: Vec<WordId> = match (words_with_a, words_with_b) {
            (Some(wa), Some(wb)) => wa.intersection(wb).cloned().collect(),
            _ => vec![],
        };

        for word_id in &word_ids {
            let entry = self.a.words.get(&word_id.0).unwrap();
            // Check the word contains at least one tokenization that contains this bigram
            let mut found = false;
            for ss in &entry.symbolifications {
                if ss.windows(2).any(|x| x[0] == a && x[1] == b) {
                    found = true;
                    break;
                }
            }
            if found {
                let mut entry = self.a.sub_word(word_id);
                entry.symbolifications = self.shortest_symbolifications(&entry.word);
                self.a.add_word(entry);
            }
        }
        new_id
    }
}

pub struct CharTokenizer;

impl WordToToken<char> for CharTokenizer {
    fn convert(&self, s: &'_ str) -> Vec<char> {
        s.chars().collect()
    }
}

pub struct ByteTokenizer;

impl WordToToken<u8> for ByteTokenizer {
    fn convert(&self, s: &'_ str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }
}

pub enum AnalyserWrapper {
    Bytes(Analyser<u8, ByteTokenizer>),
    String(Analyser<char, CharTokenizer>),
}

impl AnalyserWrapper {
    pub fn new(table: SymbolTableWrapper, word_list: Vec<String>) -> AnalyserWrapper {
        match table {
            SymbolTableWrapper::Bytes(table) => {
                AnalyserWrapper::Bytes(Analyser::new(table, ByteTokenizer, word_list))
            }
            SymbolTableWrapper::String(table) => {
                AnalyserWrapper::String(Analyser::new(table, CharTokenizer, word_list))
            }
        }
    }

    pub fn get_symbol_table(&self) -> SymbolTableWrapper {
        match self {
            AnalyserWrapper::Bytes(v) => SymbolTableWrapper::Bytes(v.symbol_table.clone()),
            AnalyserWrapper::String(v) => SymbolTableWrapper::String(v.symbol_table.clone()),
        }
    }

    pub fn word_list(&self) -> Vec<&str> {
        match self {
            AnalyserWrapper::Bytes(v) => v.word_list(),
            AnalyserWrapper::String(v) => v.word_list(),
        }
    }

    pub fn symbolifications(&self, word: &str) -> Vec<Vec<SymbolTableEntryId>> {
        match self {
            AnalyserWrapper::Bytes(v) => v.shortest_symbolifications_str(word),
            AnalyserWrapper::String(v) => v.shortest_symbolifications_str(word),
        }
    }

    pub fn get_all_symbolifications(&self) -> Vec<Vec<Vec<SymbolTableEntryId>>> {
        self.word_list()
            .iter()
            .map(|s| self.symbolifications(s))
            .collect::<Vec<_>>()
    }

    pub fn get_symbolization_ways_counts(&self) -> BTreeMap<usize, usize> {
        let mut symbolization_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for word in self.word_list() {
            let n_symbolizations = self.symbolifications(word).len();
            *symbolization_counts.entry(n_symbolizations).or_insert(0) += 1;
        }
        symbolization_counts
    }

    pub fn get_ordered_symbol_counts(&self) -> Vec<(SymbolTableEntryId, usize)> {
        let mut symbol_counts: BTreeMap<SymbolTableEntryId, usize> = BTreeMap::new();
        for x in self.get_all_symbolifications().iter().flatten().flatten() {
            *symbol_counts.entry(*x).or_insert(0) += 1
        }

        let mut symbol_counts: Vec<_> = symbol_counts.into_iter().collect();
        symbol_counts.sort_by_key(|e| e.1);
        symbol_counts.reverse();
        symbol_counts
    }

    pub fn get_bigram_counts(
        &self,
    ) -> &BTreeMap<(SymbolTableEntryId, SymbolTableEntryId), (usize, f64)> {
        match self {
            AnalyserWrapper::Bytes(v) => v.get_bigram_counts(),
            AnalyserWrapper::String(v) => v.get_bigram_counts(),
        }
    }

    pub fn get_symbol_renderer<'a>(
        &'a self,
        start: &'a str,
        end: &'a str,
    ) -> Box<dyn SymbolIdRenderer + 'a> {
        match self {
            AnalyserWrapper::Bytes(v) => Box::new(SymbolIdRendererU8 {
                table: &v.symbol_table,
                start,
                end,
            }),
            AnalyserWrapper::String(v) => Box::new(SymbolIdRendererChar {
                table: &v.symbol_table,
                start,
                end,
            }),
        }
    }

    pub fn concatenate_symbols(
        &mut self,
        a: SymbolTableEntryId,
        b: SymbolTableEntryId,
    ) -> SymbolTableEntryId {
        match self {
            AnalyserWrapper::Bytes(analyser) => analyser.concatenate_symbols(a, b),
            AnalyserWrapper::String(analyser) => analyser.concatenate_symbols(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_create() {
        let mut u = SymbolTable::<char>::new();
        u.add(SymbolTableEntry::Single('a')).unwrap();
        u.add(SymbolTableEntry::Compound("bb".chars().collect()))
            .unwrap();
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;
        let _a: Analyser<char, CharTokenizer> = Analyser::new(u, tokenizer, wordlist);
    }

    #[test]
    pub fn test_get_best_tokenisations() {
        let mut u = SymbolTable::<char>::new();
        let id_a = u.add(SymbolTableEntry::Single('a')).unwrap();
        let id_bb = u
            .add(SymbolTableEntry::Compound("bb".chars().collect()))
            .unwrap();
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;

        let a = Analyser::new(u, tokenizer, wordlist);

        let v: Vec<(&str, Vec<Vec<SymbolTableEntryId>>)> = a.get_symbolisations();
        assert_eq!(
            v,
            vec![("aa", vec![vec![id_a, id_a]]), ("bb", vec![vec![id_bb]])]
        );
    }

    #[test]
    pub fn test_get_token_counts() {
        let mut u = SymbolTable::<char>::new();
        let _id_start = u.start_symbol_id();
        let _id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a')).unwrap();
        let id_bb = u
            .add(SymbolTableEntry::Compound("bb".chars().collect()))
            .unwrap();
        let _id_c = u.add(SymbolTableEntry::Single('c'));
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;

        let a = Analyser::new(u, tokenizer, wordlist);
        let v: Vec<(SymbolTableEntryId, (usize, f64))> = a.get_symbol_counts();
        assert_eq!(v, vec![(id_a, (2, 2.0)), (id_bb, (1, 1.0)),]);
    }

    #[test]
    pub fn test_can_concatenate_symbol() {
        let mut u = SymbolTable::<char>::new();
        let _id_start = u.start_symbol_id();
        let _id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a')).unwrap();
        let id_bb = u
            .add(SymbolTableEntry::Compound("bb".chars().collect()))
            .unwrap();
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;
        let mut a = Analyser::new(u, tokenizer, wordlist);

        let id_aa = a.concatenate_symbols(id_a, id_a);

        let v: Vec<(&str, Vec<Vec<SymbolTableEntryId>>)> = a.get_symbolisations();
        assert_eq!(
            v,
            vec![("aa", vec![vec![id_aa]]), ("bb", vec![vec![id_bb]])]
        );

        let v: Vec<(SymbolTableEntryId, (usize, f64))> = a.get_symbol_counts();
        assert_eq!(v, vec![(id_bb, (1, 1.0)), (id_aa, (1, 1.0))]);
    }

    #[test]
    pub fn test_handles_multiple_best_symbolifications() {
        use crate::vecutils::Sortable;

        let mut u = SymbolTable::<char>::new();
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a')).unwrap();
        let id_aa = u
            .add(SymbolTableEntry::Compound("aa".chars().collect()))
            .unwrap();
        let wordlist = vec!["aaa".to_string()];
        let tokenizer = CharTokenizer;
        let a = Analyser::new(u, tokenizer, wordlist);

        let symbolisations = a.get_symbolisations();
        assert_eq!(1, symbolisations.len());
        assert_eq!("aaa", symbolisations[0].0);
        assert_eq!(
            vec![vec![id_a, id_aa], vec![id_aa, id_a]].sorted(),
            symbolisations[0].1.sorted()
        );

        let v: Vec<(SymbolTableEntryId, (usize, f64))> =
            a.get_symbol_counts().into_iter().collect();
        assert_eq!(v, vec![(id_a, (2, 1.0)), (id_aa, (2, 1.0))]);

        let bigrams = a.get_bigram_counts();

        eprintln!("bigrams = {:?}", bigrams.iter().collect::<Vec<_>>());

        // We have two tokenizations ^ A AA $ and ^ AA A $ both with weights 0.5
        assert_eq!(bigrams.get(&(id_start, id_a)).cloned(), Some((1, 0.5)));
        assert_eq!(bigrams.get(&(id_a, id_aa)).cloned(), Some((1, 0.5)));
        assert_eq!(bigrams.get(&(id_aa, id_end)).cloned(), Some((1, 0.5)));

        assert_eq!(bigrams.get(&(id_start, id_aa)).cloned(), Some((1, 0.5)));
        assert_eq!(bigrams.get(&(id_aa, id_a)).cloned(), Some((1, 0.5)));
        assert_eq!(bigrams.get(&(id_a, id_end)).cloned(), Some((1, 0.5)));
    }

    #[test]
    pub fn test_basic_ops_bytes() {
        let mut u = SymbolTable::<u8>::new();
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single(b'a')).unwrap();
        let id_bb = u.add(SymbolTableEntry::Compound(b"bb".to_vec())).unwrap();
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = ByteTokenizer;
        let a: Analyser<u8, ByteTokenizer> = Analyser::new(u, tokenizer, wordlist);

        assert_eq!(
            a.get_symbol_counts(),
            vec![(id_a, (2, 2.0)), (id_bb, (1, 1.0))]
        );

        let bigrams = a.get_bigram_counts();
        assert_eq!(bigrams.get(&(id_start, id_a)).cloned(), Some((1, 1.0)));
        assert_eq!(bigrams.get(&(id_a, id_a)).cloned(), Some((1, 1.0)));
        assert_eq!(bigrams.get(&(id_a, id_end)).cloned(), Some((1, 1.0)));

        assert_eq!(bigrams.get(&(id_start, id_bb)).cloned(), Some((1, 1.0)));
        assert_eq!(bigrams.get(&(id_bb, id_end)).cloned(), Some((1, 1.0)));
    }
}
