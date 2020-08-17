use crate::{
    ngram::BigramCount,
    renderer::{SymbolIdRenderer, SymbolIdRendererChar, SymbolIdRendererU8},
    symbol::{SymbolTable, SymbolTableEntry, SymbolTableEntryId, SymbolTableWrapper},
};
use std::collections::BTreeMap;

// TODO: Would be nice to be able to return a slice when we can rather than
// allocating up a complete buffer, but the lifetimes on the required associated types
// are tricky.
pub trait WordToToken<T> {
    fn convert<'a>(&self, s: &'a str) -> Vec<T>;
}

pub struct Analyser<T, Tokenizer> {
    pub symbol_table: SymbolTable<T>,
    tokenizer: Tokenizer,
    word_list: Vec<String>,
}

pub fn select_by_lowest_value<T, F, V>(s: &[T], f: &F) -> Vec<T>
where
    V: Ord,
    F: Fn(&T) -> V,
    T: Clone,
{
    let min = s.iter().map(f).min();
    match min {
        None => vec![],
        Some(min) => s.iter().filter(|v| f(v) == min).cloned().collect(),
    }
}

impl<T, Tokenizer> Analyser<T, Tokenizer>
where
    Tokenizer: WordToToken<T>,
    T: Eq + Clone,
{
    //TODO: Should this take ownership of the list of strings too?
    pub fn new(
        symbol_table: SymbolTable<T>,
        tokenizer: Tokenizer,
        word_list: Vec<String>,
    ) -> Analyser<T, Tokenizer> {
        Analyser {
            symbol_table,
            tokenizer,
            word_list,
        }
    }

    pub fn word_list(&self) -> Vec<&str> {
        self.word_list.iter().map(|v| v.as_str()).collect()
    }

    pub fn shortest_symbolifications(&self, word: &str) -> Vec<Vec<SymbolTableEntryId>> {
        let v = self
            .symbol_table
            .symbolifications(self.tokenizer.convert(word).as_slice());
        select_by_lowest_value(&v, &|s: &Vec<SymbolTableEntryId>| s.len())
    }

    pub fn get_symbolisations(&self) -> Vec<(&str, Vec<Vec<SymbolTableEntryId>>)> {
        self.word_list
            .iter()
            .map(|word| (word.as_str(), self.shortest_symbolifications(word)))
            .collect()
    }

    pub fn get_symbol_counts(&self) -> Vec<(SymbolTableEntryId, f32)> {
        // A symboltable guarantees that its symbolids are contiguous and up to
        // symboltable.len()
        // So we can count directly into an array.
        let mut counts: Vec<f32> = vec![0.0; self.symbol_table.len()];

        let s = self.get_symbolisations();
        for (_k, v) in s {
            let l = v.len();
            for ss in v {
                for s in ss {
                    counts[s.0 as usize] += 1.0 / (l as f32);
                }
            }
        }
        let result: Vec<(SymbolTableEntryId, f32)> = counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| (SymbolTableEntryId(i as u64), count))
            .collect();
        result
    }

    pub fn add_symbol(&mut self, s: SymbolTableEntry<T>) -> SymbolTableEntryId {
        self.symbol_table.add(s)
    }

    pub fn remove_symbol(&mut self, s: SymbolTableEntryId) {
        self.symbol_table.remove(s)
    }

    pub fn get_symbol_id(&self, s: &SymbolTableEntry<T>) -> Option<SymbolTableEntryId> {
        self.symbol_table.find(s)
    }

    pub fn get_bigram_counts(&self) -> BigramCount<SymbolTableEntryId, f32> {
        let id_start = self.symbol_table.start_symbol_id();
        let id_end = self.symbol_table.end_symbol_id();
        let s = self.get_symbolisations();
        // Convert Vec(&str, Vec<Vec<SymbolTableEntryId>>) to Vec<(Vec<SymbolTableEntryId>, f32)
        let v: Vec<(Vec<SymbolTableEntryId>, f32)> = s
            .into_iter()
            .flat_map(|(_k, ss)| {
                let l = ss.len();
                ss.into_iter().map(move |v| (v, 1.0 / (l as f32)))
            })
            .collect();
        // Append start and end symbols.
        let v: Vec<(Vec<SymbolTableEntryId>, f32)> = v
            .into_iter()
            .map(|(mut s, w)| {
                s.insert(0, id_start);
                s.push(id_end);
                (s, w)
            })
            .collect();

        v.iter().map(|(s, w)| (s.as_slice(), *w)).collect()
    }

    pub fn concatenate_symbols(
        &mut self,
        a: SymbolTableEntryId,
        b: SymbolTableEntryId,
    ) -> SymbolTableEntryId {
        let mut v = Vec::<T>::new();
        self.symbol_table.append_to_vec(a, &mut v).unwrap();
        self.symbol_table.append_to_vec(b, &mut v).unwrap();
        let symbol_table_entry = SymbolTableEntry::Compound(v);
        self.symbol_table.add(symbol_table_entry)
    }
}

pub struct CharTokenizer;

impl WordToToken<char> for CharTokenizer {
    fn convert<'a>(&self, s: &'a str) -> Vec<char> {
        s.chars().collect()
    }
}

pub struct ByteTokenizer;

impl WordToToken<u8> for ByteTokenizer {
    fn convert<'a>(&self, s: &'a str) -> Vec<u8> {
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

    pub fn word_list(&self) -> Vec<&str> {
        match self {
            AnalyserWrapper::Bytes(v) => v.word_list(),
            AnalyserWrapper::String(v) => v.word_list(),
        }
    }

    pub fn symbolifications(&self, word: &str) -> Vec<Vec<SymbolTableEntryId>> {
        match self {
            AnalyserWrapper::Bytes(v) => v.shortest_symbolifications(word),
            AnalyserWrapper::String(v) => v.shortest_symbolifications(word),
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

    pub fn get_bigram_counts(&self) -> BigramCount<SymbolTableEntryId, usize> {
        self.get_all_symbolifications()
            .iter()
            .flatten()
            .map(|v| -> &[SymbolTableEntryId] { &v })
            .collect()
    }

    pub fn get_ordered_bigram_counts(
        &self,
    ) -> Vec<((SymbolTableEntryId, SymbolTableEntryId), usize)> {
        let mut bigram_counts: Vec<_> = self
            .get_bigram_counts()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        bigram_counts.sort_by_key(|e| e.1);
        bigram_counts.reverse();
        bigram_counts
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
    use crate::ngram::BigramCount;

    #[test]
    pub fn test_create() {
        let mut u = SymbolTable::<char>::new();
        u.add(SymbolTableEntry::Single('a'));
        u.add(SymbolTableEntry::Compound("bb".chars().collect()));
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;
        let _a: Analyser<char, CharTokenizer> = Analyser::new(u, tokenizer, wordlist);
    }

    #[test]
    pub fn test_get_best_tokenisations() {
        let mut u = SymbolTable::<char>::new();
        let id_a = u.add(SymbolTableEntry::Single('a'));
        let id_bb = u.add(SymbolTableEntry::Compound("bb".chars().collect()));
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
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a'));
        let id_bb = u.add(SymbolTableEntry::Compound("bb".chars().collect()));
        let id_c = u.add(SymbolTableEntry::Single('c'));
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;

        let a = Analyser::new(u, tokenizer, wordlist);
        let v: Vec<(SymbolTableEntryId, f32)> = a.get_symbol_counts();
        assert_eq!(
            v,
            vec![
                (id_start, 0.0),
                (id_end, 0.0),
                (id_a, 2.0),
                (id_bb, 1.0),
                (id_c, 0.0)
            ]
        );
    }

    #[test]
    pub fn test_can_add_symbol() {
        let mut u = SymbolTable::<char>::new();
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a'));
        let id_bb = u.add(SymbolTableEntry::Compound("bb".chars().collect()));
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = CharTokenizer;
        let mut a = Analyser::new(u, tokenizer, wordlist);

        let id_aa = a.add_symbol(SymbolTableEntry::Compound("aa".chars().collect()));

        let v: Vec<(&str, Vec<Vec<SymbolTableEntryId>>)> = a.get_symbolisations();
        assert_eq!(
            v,
            vec![("aa", vec![vec![id_aa]]), ("bb", vec![vec![id_bb]])]
        );

        let v: Vec<(SymbolTableEntryId, f32)> = a.get_symbol_counts();
        assert_eq!(
            v,
            vec![
                (id_start, 0.0),
                (id_end, 0.0),
                (id_a, 0.0),
                (id_bb, 1.0),
                (id_aa, 1.0)
            ]
        );
    }

    #[test]
    pub fn test_can_remove_symbol() {
        let mut u = SymbolTable::<char>::new();
        let id_a = u.add(SymbolTableEntry::Single('a'));
        let id_aa = u.add(SymbolTableEntry::Compound("aa".chars().collect()));
        let id_c = u.add(SymbolTableEntry::Single('c'));

        let wordlist = vec!["aa".to_string(), "c".to_string()];
        let tokenizer = CharTokenizer;
        let mut a = Analyser::new(u, tokenizer, wordlist);

        assert_eq!(
            a.get_symbolisations(),
            vec![("aa", vec![vec![id_aa]]), ("c", vec![vec![id_c]])]
        );

        a.remove_symbol(id_aa);

        assert_eq!(
            a.get_symbolisations(),
            vec![("aa", vec![vec![id_a, id_a]]), ("c", vec![vec![id_c]])]
        );
    }

    #[test]
    pub fn test_handles_multiple_best_symbolifications() {
        let mut u = SymbolTable::<char>::new();
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single('a'));
        let id_aa = u.add(SymbolTableEntry::Compound("aa".chars().collect()));
        let wordlist = vec!["aaa".to_string()];
        let tokenizer = CharTokenizer;
        let a = Analyser::new(u, tokenizer, wordlist);

        assert_eq!(
            a.get_symbolisations(),
            vec![("aaa", vec![vec![id_a, id_aa], vec![id_aa, id_a]])]
        );

        let v: Vec<(SymbolTableEntryId, f32)> = a
            .get_symbol_counts()
            .into_iter()
            .filter(|(_k, v)| *v != 0.0)
            .collect();
        assert_eq!(v, vec![(id_a, 1.0), (id_aa, 1.0)]);

        let bigrams: BigramCount<SymbolTableEntryId, f32> = a.get_bigram_counts();

        // We have two tokenizations ^ A AA $ and ^ AA A $ both with weights 0.5
        assert_eq!(bigrams.count(&(id_start, id_a)), 0.5);
        assert_eq!(bigrams.count(&(id_a, id_aa)), 0.5);
        assert_eq!(bigrams.count(&(id_aa, id_end)), 0.5);

        assert_eq!(bigrams.count(&(id_start, id_aa)), 0.5);
        assert_eq!(bigrams.count(&(id_aa, id_a)), 0.5);
        assert_eq!(bigrams.count(&(id_a, id_end)), 0.5);
    }

    #[test]
    pub fn test_basic_ops_bytes() {
        let mut u = SymbolTable::<u8>::new();
        let id_start = u.start_symbol_id();
        let id_end = u.end_symbol_id();
        let id_a = u.add(SymbolTableEntry::Single(b'a'));
        let id_bb = u.add(SymbolTableEntry::Compound(b"bb".to_vec()));
        let wordlist = vec!["aa".to_string(), "bb".to_string()];
        let tokenizer = ByteTokenizer;
        let a: Analyser<u8, ByteTokenizer> = Analyser::new(u, tokenizer, wordlist);

        assert_eq!(
            a.get_symbol_counts(),
            vec![(id_start, 0.0), (id_end, 0.0), (id_a, 2.0), (id_bb, 1.0)]
        );

        let bigrams: BigramCount<SymbolTableEntryId, f32> = a.get_bigram_counts();
        assert_eq!(bigrams.count(&(id_start, id_a)), 1.0);
        assert_eq!(bigrams.count(&(id_a, id_a)), 1.0);
        assert_eq!(bigrams.count(&(id_a, id_end)), 1.0);

        assert_eq!(bigrams.count(&(id_start, id_bb)), 1.0);
        assert_eq!(bigrams.count(&(id_bb, id_end)), 1.0);
    }
}
