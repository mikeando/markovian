use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Serialize, Deserialize)]
pub struct SymbolTableEntryId(pub u64);

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub enum SymbolTableEntryType {
    Start,
    End,
    Single,
    Compound,
    Dead,
}

// TODO: Is there value in having multiple Start / End symbols
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Serialize, Deserialize)]
pub enum SymbolTableEntry<T> {
    Start,
    End,
    Single(T),
    Compound(Vec<T>),
    Dead(Option<usize>), // Index of next dead entry
}

impl<T> SymbolTableEntry<T>
where
    T: PartialEq,
{
    pub fn matches_start(&self, v: &[T]) -> bool {
        if v.is_empty() {
            return false;
        }
        match self {
            SymbolTableEntry::Single(s) => v[0] == *s,
            SymbolTableEntry::Compound(ss) => v.starts_with(&ss),
            _ => false,
        }
    }

    // the entry must match the start of v, but can go past the
    // end of v
    pub fn matches_start_prefix(&self, v: &[T]) -> bool {
        if v.is_empty() {
            return false;
        }
        match self {
            SymbolTableEntry::Single(s) => v[0] == *s,
            SymbolTableEntry::Compound(ss) => {
                if v.len() >= ss.len() {
                    v.starts_with(&ss)
                } else {
                    v == &ss[..v.len()]
                }
            }
            _ => false,
        }
    }

    pub fn matches_end_suffix(&self, v: &[T]) -> bool {
        if v.is_empty() {
            return false;
        }
        match self {
            SymbolTableEntry::Single(s) => v.last().unwrap() == s,
            SymbolTableEntry::Compound(ss) => {
                if v.len() >= ss.len() {
                    v.ends_with(&ss)
                } else {
                    v == &ss[(ss.len() - v.len())..]
                }
            }
            _ => false,
        }
    }

    pub fn first(&self) -> Option<&T> {
        match self {
            SymbolTableEntry::Single(s) => Some(s),
            SymbolTableEntry::Compound(ss) => ss.first(),
            _ => None,
        }
    }
}

impl<T> SymbolTableEntry<T> {
    pub fn length(&self) -> usize {
        match self {
            SymbolTableEntry::Single(_) => 1,
            SymbolTableEntry::Compound(ss) => ss.len(),
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTableLite<T> {
    index: Vec<SymbolTableEntry<T>>,
    dead_chain_head: Option<usize>,
    n_dead: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SymbolTableLite<T>", into = "SymbolTableLite<T>")]
pub struct SymbolTable<T>
where
    T: Ord + Clone,
{
    index: Vec<SymbolTableEntry<T>>,
    dead_chain_head: Option<usize>,
    n_dead: usize,
    by_first_character: BTreeMap<T, Vec<usize>>,
}

impl<T> From<SymbolTableLite<T>> for SymbolTable<T>
where
    T: Ord + Clone,
{
    fn from(v: SymbolTableLite<T>) -> Self {
        let SymbolTableLite {
            index,
            dead_chain_head,
            n_dead,
        } = v;
        let mut by_first_character: BTreeMap<T, Vec<usize>> = BTreeMap::new();

        for (i, e) in index.iter().enumerate() {
            if let Some(t) = e.first().cloned() {
                by_first_character.entry(t).or_default().push(i);
            }
        }

        SymbolTable {
            index,
            dead_chain_head,
            n_dead,
            by_first_character,
        }
    }
}

impl<T> From<SymbolTable<T>> for SymbolTableLite<T>
where
    T: Ord + Clone,
{
    fn from(v: SymbolTable<T>) -> Self {
        let SymbolTable {
            index,
            dead_chain_head,
            n_dead,
            ..
        } = v;
        SymbolTableLite {
            index,
            dead_chain_head,
            n_dead,
        }
    }
}

impl<T> SymbolTable<T>
where
    T: Ord + Clone,
{
    pub fn new() -> SymbolTable<T> {
        SymbolTable {
            index: vec![],
            dead_chain_head: None,
            n_dead: 0,
            by_first_character: BTreeMap::new(),
        }
    }

    // Includes dead entries.
    pub fn max_symbol_id(&self) -> usize {
        self.index.len() + 2
    }

    pub fn get_by_id(&self, id: SymbolTableEntryId) -> Option<&SymbolTableEntry<T>> {
        match id.0 {
            0 => Some(&SymbolTableEntry::Start),
            1 => Some(&SymbolTableEntry::End),
            _ => self.index.get((id.0 - 2) as usize),
        }
    }

    pub fn start_symbol_id(&self) -> SymbolTableEntryId {
        SymbolTableEntryId(0)
    }

    pub fn end_symbol_id(&self) -> SymbolTableEntryId {
        SymbolTableEntryId(1)
    }

    // Includes dead entries
    pub fn iter(&self) -> impl Iterator<Item = (SymbolTableEntryId, &SymbolTableEntry<T>)> {
        vec![
            (SymbolTableEntryId(0), &SymbolTableEntry::Start),
            (SymbolTableEntryId(1), &SymbolTableEntry::End),
        ]
        .into_iter()
        .chain(
            self.index
                .iter()
                .enumerate()
                .map(|(i, v)| (SymbolTableEntryId(i as u64), v)),
        )
    }
}

impl<T> SymbolTable<T>
where
    T: PartialEq + Ord + Clone,
{
    pub fn add(&mut self, value: SymbolTableEntry<T>) -> SymbolTableEntryId {
        if value == SymbolTableEntry::Start {
            return self.start_symbol_id();
        }
        if value == SymbolTableEntry::End {
            return self.end_symbol_id();
        }

        if let Some(id) = self.find(&value) {
            return id;
        }

        let first = value.first().cloned();

        let id = match self.dead_chain_head {
            Some(last_dead) => {
                let e = &self.index[last_dead];
                match e {
                    SymbolTableEntry::Dead(next_dead) => {
                        self.dead_chain_head = *next_dead;
                        self.n_dead -= 1;
                        self.index[last_dead] = value;
                        SymbolTableEntryId((last_dead + 2) as u64)
                    }
                    _ => panic!("SymbolTable.dead_chain_head pointing to a live entry"),
                }
            }
            None => {
                self.index.push(value);
                SymbolTableEntryId((self.index.len() + 1) as u64)
            }
        };

        if let Some(first) = first {
            self.by_first_character
                .entry(first)
                .or_default()
                .push((id.0 - 2) as usize)
        }

        id
    }

    pub fn find(&self, value: &SymbolTableEntry<T>) -> Option<SymbolTableEntryId> {
        match value {
            SymbolTableEntry::Start => return Some(SymbolTableEntryId(0)),
            SymbolTableEntry::End => return Some(SymbolTableEntryId(1)),
            SymbolTableEntry::Dead(_) => panic!("Cant find dead entries"),
            _ => {}
        };
        let first = value.first()?;

        let ids = self.by_first_character.get(first)?;
        for id in ids {
            if &self.index[*id] == value {
                return Some(SymbolTableEntryId((id + 2) as u64));
            }
        }
        None
    }

    // We don't actually remove the entry, just mark it as dead, so that we dont invalidate
    // existing ids.
    pub fn remove(&mut self, id: SymbolTableEntryId) {
        //TODO: Handle error cases - like trying to remove a symbol that is out-of-bounds
        // and removing the start or end symbols.
        assert!(id.0 >= 2);

        let first = self.index[(id.0 - 2) as usize].first();

        if let Some(first) = first {
            if let Some(ids) = self.by_first_character.get_mut(first) {
                *ids = ids
                    .iter()
                    .filter(|iid| **iid != (id.0 - 2) as usize)
                    .cloned()
                    .collect();
            }
        }

        self.index[(id.0 - 2) as usize] = SymbolTableEntry::Dead(self.dead_chain_head);
        self.dead_chain_head = Some((id.0 - 2) as usize);
        self.n_dead += 1;
    }

    // Remove all the dead entries. Potentially invalidating any exitsing ids.
    pub fn compact(&mut self) {
        self.dead_chain_head = None;
        self.n_dead = 0;
        self.index
            .retain(|e| !matches!(e, SymbolTableEntry::Dead(_)));

        self.by_first_character = BTreeMap::new();
        for (i, e) in self.index.iter().enumerate() {
            if let Some(t) = e.first().cloned() {
                self.by_first_character.entry(t).or_default().push(i);
            }
        }
    }
}

impl<T> Default for SymbolTable<T>
where
    T: Ord + Clone,
{
    fn default() -> Self {
        SymbolTable::new()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    InvalidSymbolify,
    InvalidSymbolType(SymbolTableEntryType),
}

type Result<T> = std::result::Result<T, Error>;

impl<T> SymbolTable<T>
where
    T: Ord + Clone,
{
    /// *NOTE* This only returns the first matching decomposition.
    /// Use `symbolifications` instead if you want all of them.
    pub fn symbolify(&self, v: &[T]) -> Result<Vec<SymbolTableEntryId>>
    where
        T: Eq,
    {
        let mut b: &[T] = v;
        let mut result: Vec<SymbolTableEntryId> = vec![];
        'outer: while !b.is_empty() {
            // We dont need to consider START and END as they can never match
            for (k, s) in self.index.iter().enumerate() {
                if s.matches_start(b) {
                    b = &b[s.length()..];
                    result.push(SymbolTableEntryId((k + 2) as u64));
                    continue 'outer;
                }
            }
            return Err(Error::InvalidSymbolify);
        }
        Ok(result)
    }

    // TODO: Would be better to do this using an iterator
    pub fn possible_symbols<'a>(&self, v: &'a [T]) -> Vec<(SymbolTableEntryId, &'a [T])>
    where
        T: Eq,
    {
        let mut result = vec![];
        if v.is_empty() {
            return result;
        }

        let symbol_indices = self.by_first_character.get(&v[0]);
        match symbol_indices {
            None => {}
            Some(symbol_indices) => {
                for index in symbol_indices {
                    let s = &self.index[*index];
                    if s.matches_start(v) {
                        result.push((SymbolTableEntryId((index + 2) as u64), &v[s.length()..]));
                    }
                }
            }
        }
        result
    }

    pub fn possible_symbols_prefix<'a>(&self, v: &'a [T]) -> Vec<(SymbolTableEntryId, &'a [T])>
    where
        T: Eq,
    {
        let mut result = vec![];
        for (id, s) in self.index.iter().enumerate() {
            if s.matches_start_prefix(v) {
                let slice: &[T] = if s.length() >= v.len() {
                    &[]
                } else {
                    &v[s.length()..]
                };
                result.push((SymbolTableEntryId((id + 2) as u64), slice));
            }
        }
        result
    }

    pub fn possible_symbols_suffix<'a>(&self, v: &'a [T]) -> Vec<(SymbolTableEntryId, &'a [T])>
    where
        T: Eq,
    {
        let mut result = vec![];
        for (id, s) in self.index.iter().enumerate() {
            if s.matches_end_suffix(v) {
                let slice: &[T] = if s.length() >= v.len() {
                    &[]
                } else {
                    &v[0..(v.len() - s.length())]
                };
                result.push((SymbolTableEntryId((id + 2) as u64), slice));
            }
        }
        result
    }

    // Returns all symbol sequences that match exactly the value v
    pub fn symbolifications(&self, v: &[T]) -> Vec<Vec<SymbolTableEntryId>>
    where
        T: Eq,
    {
        if v.is_empty() {
            return vec![vec![]];
        }

        let mut result = Vec::with_capacity(1);

        // parent_index, processed, current symbol, rest
        let mut stack: Vec<(i32, bool, SymbolTableEntryId, &[T])> = Vec::with_capacity(32);
        stack.push((-1, false, SymbolTableEntryId(0), v));

        while !stack.is_empty() {
            let current_index = stack.len() as i32 - 1;
            let (_parent_id, vistied, _symbol, rest) = stack[current_index as usize];
            if vistied {
                stack.pop();
            } else {
                stack[current_index as usize].1 = true;
                assert!(!rest.is_empty());
                for (id, child_rest) in self.possible_symbols(rest) {
                    if child_rest.is_empty() {
                        // Walk back up through the stack adding the parents at each stage
                        // TODO: If each node traced its depth we could create the right size the first time!
                        let mut vv = Vec::with_capacity(v.len());
                        vv.push(id);
                        let mut idx = current_index;
                        loop {
                            let parent = stack[idx as usize];
                            if parent.0 == -1 {
                                break;
                            }
                            vv.push(parent.2);
                            idx = parent.0;
                        }
                        vv.reverse();
                        result.push(vv)
                    } else {
                        stack.push((current_index, false, id, child_rest))
                    }
                }
            }
        }
        result
    }

    // Returns all symbol sequences SS such that
    // render(SS).startswith(v) but !render(&SS[..SS.len()-1]).startswith(v)
    // This means in practice that the last symbol of an allowed sequence can overhang the
    // end of v slightly. For example if our symbols are "ab" and "ac", "a"
    // then a prefix symbolification of "aba" will contain not just the exact match ["ab","a"],
    // but two that overhang by one ["ab","ab"] and ["ab", "ac"]
    //
    // symbolifications_prefix("aba") === [ ["ab", "ab"], ["ab","ac"], ["ab","a"]]
    //
    // The main reason for this is to help with generation of sequences that start with a
    // prerfix.
    pub fn symbolifications_prefix(&self, v: &[T]) -> Vec<Vec<SymbolTableEntryId>>
    where
        T: Eq,
    {
        if v.is_empty() {
            return vec![vec![]];
        }

        let mut result = vec![];
        for (id, rest) in self.possible_symbols_prefix(v) {
            for mut child in self.symbolifications_prefix(rest) {
                child.insert(0, id);
                result.push(child)
            }
        }
        result
    }

    // Same as symbolification_prefix but allows first symbol to overhang front.
    pub fn symbolifications_suffix(&self, v: &[T]) -> Vec<Vec<SymbolTableEntryId>>
    where
        T: Eq,
    {
        if v.is_empty() {
            return vec![vec![]];
        }

        let mut result = vec![];
        for (id, rest) in self.possible_symbols_suffix(v) {
            for mut child in self.symbolifications_suffix(rest) {
                child.push(id);
                result.push(child)
            }
        }
        result
    }
}

impl<T> SymbolTable<T>
where
    T: Clone + Ord,
{
    pub fn append_to_vec(&self, a: SymbolTableEntryId, v: &mut Vec<T>) -> Result<()> {
        match self.get_by_id(a).unwrap() {
            SymbolTableEntry::Start => Err(Error::InvalidSymbolType(SymbolTableEntryType::Start)),
            SymbolTableEntry::End => Err(Error::InvalidSymbolType(SymbolTableEntryType::End)),
            SymbolTableEntry::Single(e) => {
                v.push(e.clone());
                Ok(())
            }
            SymbolTableEntry::Compound(es) => {
                v.extend_from_slice(&es);
                Ok(())
            }
            SymbolTableEntry::Dead(_) => Err(Error::InvalidSymbolType(SymbolTableEntryType::Dead)),
        }
    }
}

impl SymbolTable<u8> {
    pub fn symbolify_str(&self, v: &str) -> Result<Vec<SymbolTableEntryId>> {
        self.symbolify(v.as_bytes())
    }

    pub fn symbolifications_str(&self, v: &str) -> Vec<Vec<SymbolTableEntryId>> {
        self.symbolifications(v.as_bytes())
    }

    pub fn symbolifications_prefix_str(&self, v: &str) -> Vec<Vec<SymbolTableEntryId>> {
        self.symbolifications_prefix(v.as_bytes())
    }

    pub fn symbolifications_suffix_str(&self, v: &str) -> Vec<Vec<SymbolTableEntryId>> {
        self.symbolifications_suffix(v.as_bytes())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolTableWrapper {
    Bytes(SymbolTable<u8>),
    String(SymbolTable<char>),
}

impl SymbolTableWrapper {
    pub fn encoding(&self) -> TableEncoding {
        match self {
            SymbolTableWrapper::Bytes(_) => TableEncoding::Bytes,
            SymbolTableWrapper::String(_) => TableEncoding::String,
        }
    }

    pub fn max_symbol_id(&self) -> usize {
        match self {
            SymbolTableWrapper::Bytes(table) => table.max_symbol_id(),
            SymbolTableWrapper::String(table) => table.max_symbol_id(),
        }
    }

    pub fn symbolifications(&self, s: &str) -> Vec<Vec<SymbolTableEntryId>> {
        match self {
            SymbolTableWrapper::Bytes(table) => table.symbolifications_str(s),
            SymbolTableWrapper::String(table) => {
                table.symbolifications(&s.chars().collect::<Vec<_>>())
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    struct ContextDefault {
        start: SymbolTableEntryId,
        end: SymbolTableEntryId,
        a: SymbolTableEntryId,
        xyz: SymbolTableEntryId,
    }

    fn default_symbol_table() -> (SymbolTable<u8>, ContextDefault) {
        let mut result = SymbolTable::new();

        let start = result.add(SymbolTableEntry::Start);
        let end = result.add(SymbolTableEntry::End);
        let a = result.add(SymbolTableEntry::Single(b'a'));
        let xyz = result.add(SymbolTableEntry::Compound(vec![b'x', b'y', b'z']));

        (result, ContextDefault { start, end, a, xyz })
    }

    struct ContextABC {
        a: SymbolTableEntryId,
        ab: SymbolTableEntryId,
        ac: SymbolTableEntryId,
    }

    fn abc_table() -> (SymbolTable<u8>, ContextABC) {
        let mut s = SymbolTable::new();
        let a = s.add(SymbolTableEntry::Single(b'a'));
        let ab = s.add(SymbolTableEntry::Compound(vec![b'a', b'b']));
        let ac = s.add(SymbolTableEntry::Compound(vec![b'a', b'c']));
        (s, ContextABC { a, ab, ac })
    }

    struct ContextAB {
        a: SymbolTableEntryId,
        aa: SymbolTableEntryId,
        b: SymbolTableEntryId,
        ab: SymbolTableEntryId,
    }

    fn ab_table() -> (SymbolTable<u8>, ContextAB) {
        let mut s = SymbolTable::new();
        let a = s.add(SymbolTableEntry::Single(b'a'));
        let aa = s.add(SymbolTableEntry::Compound(vec![b'a', b'a']));
        let b = s.add(SymbolTableEntry::Single(b'b'));
        let ab = s.add(SymbolTableEntry::Compound(vec![b'a', b'b']));
        (s, ContextAB { a, aa, b, ab })
    }

    #[test]
    pub fn check_symbol_table_get_by_id() {
        let (s, c) = default_symbol_table();

        let e: Option<&SymbolTableEntry<u8>> = s.get_by_id(SymbolTableEntryId(123));
        assert_eq!(e, None);

        let e: Option<&SymbolTableEntry<u8>> = s.get_by_id(c.start);
        assert_eq!(e, Some(&SymbolTableEntry::Start));

        let e: Option<&SymbolTableEntry<u8>> = s.get_by_id(c.end);
        assert_eq!(e, Some(&SymbolTableEntry::End));

        let e: Option<&SymbolTableEntry<u8>> = s.get_by_id(c.a);
        assert_eq!(e, Some(&SymbolTableEntry::Single(b'a')));

        let e: Option<&SymbolTableEntry<u8>> = s.get_by_id(c.xyz);
        assert_eq!(e, Some(&SymbolTableEntry::Compound(vec![b'x', b'y', b'z'])));
    }

    #[test]
    pub fn test_symbol_table_symbolify() -> Result<()> {
        let (s, c) = default_symbol_table();
        assert_eq!(s.symbolify_str("")?, vec![]);
        assert_eq!(s.symbolify_str("a")?, vec![c.a]);
        assert_eq!(s.symbolify_str("xyz")?, vec![c.xyz]);
        assert_eq!(s.symbolify_str("Q"), Err(Error::InvalidSymbolify));
        assert_eq!(s.symbolify_str("axyz")?, vec![c.a, c.xyz]);
        Ok(())
    }

    #[test]
    pub fn test_double_add_doesnt_duplicate_symbol() {
        let mut s = SymbolTable::new();
        let a1 = s.add(SymbolTableEntry::Single('a'));
        let a2 = s.add(SymbolTableEntry::Single('a'));
        assert_eq!(a1, a2);
    }

    mod symbolification {

        use super::*;

        #[test]
        pub fn default_empty() {
            let (s, _c) = default_symbol_table();
            // The empty string has one symbolification - the empty one.
            assert_eq!(
                s.symbolifications_str(""),
                vec![Vec::<SymbolTableEntryId>::new()]
            );
        }

        #[test]
        pub fn default_invalid() {
            let (s, _c) = default_symbol_table();
            // An invalid string has no sumbolifications
            assert_eq!(
                s.symbolifications_str("Q"),
                Vec::<Vec<SymbolTableEntryId>>::new()
            );
        }

        #[test]
        pub fn default_single() {
            let (s, c) = default_symbol_table();
            assert_eq!(s.symbolifications_str("a"), vec![vec![c.a]]);
        }

        #[test]
        pub fn default_compound() {
            let (s, c) = default_symbol_table();
            assert_eq!(s.symbolifications_str("xyz"), vec![vec![c.xyz]]);
        }

        #[test]
        pub fn default_multiple() {
            let (s, c) = default_symbol_table();
            assert_eq!(s.symbolifications_str("axyz"), vec![vec![c.a, c.xyz]]);
        }

        #[test]
        pub fn default_multiple_invalid() {
            let (s, _c) = default_symbol_table();
            assert_eq!(
                s.symbolifications_str("axyzq"),
                Vec::<Vec<SymbolTableEntryId>>::new()
            );
        }

        use crate::vecutils::Sortable;

        #[test]
        pub fn ab_ambiguous() {
            let (s, c) = ab_table();
            assert_eq!(
                s.symbolifications_str("aa").sorted(),
                vec![vec![c.a, c.a], vec![c.aa]].sorted()
            );
            assert_eq!(
                s.symbolifications_str("aaa").sorted(),
                vec![vec![c.a, c.a, c.a], vec![c.a, c.aa], vec![c.aa, c.a]].sorted()
            );
            assert_eq!(
                s.symbolifications_str("aab").sorted(),
                vec![vec![c.a, c.a, c.b], vec![c.a, c.ab], vec![c.aa, c.b]].sorted()
            );
        }
    }

    mod symbolification_prefix {
        use super::*;

        #[test]
        pub fn abc_aba() {
            let (s, c) = abc_table();
            assert_eq!(
                s.symbolifications_prefix_str("aba"),
                vec![vec![c.ab, c.a], vec![c.ab, c.ab], vec![c.ab, c.ac]]
            );
        }

        #[test]
        pub fn abc_empty() {
            let (s, _c) = abc_table();
            assert_eq!(
                s.symbolifications_prefix_str(""),
                vec![Vec::<SymbolTableEntryId>::new()]
            );
        }

        #[test]
        pub fn abc_invalid() {
            let (s, _c) = abc_table();
            assert_eq!(
                s.symbolifications_prefix_str("qqq"),
                Vec::<Vec<SymbolTableEntryId>>::new()
            );
        }
    }

    mod matches_end_suffix {
        use super::*;

        #[test]
        pub fn single_match() {
            let ste = SymbolTableEntry::<char>::Single('a');
            assert!(ste.matches_end_suffix(&['a']));
            assert!(ste.matches_end_suffix(&['b', 'a']));
            assert!(!ste.matches_end_suffix(&['b']));
            assert!(!ste.matches_end_suffix(&['a', 'b']));
        }

        #[test]
        pub fn comound_match() {
            let ste = SymbolTableEntry::<char>::Compound(vec!['a', 'b']);
            assert!(ste.matches_end_suffix(&['a', 'b']));
            assert!(ste.matches_end_suffix(&['x', 'a', 'b']));
            assert!(!ste.matches_end_suffix(&['a', 'b', 'x']));
            assert!(ste.matches_end_suffix(&['b']));
        }
    }

    mod possible_symbols_suffix {
        use super::*;

        #[test]
        pub fn abc_empty() {
            let (s, _c) = abc_table();
            assert_eq!(s.possible_symbols_suffix(&[]), vec![]);
        }

        #[test]
        pub fn ab_ba() {
            let (s, c) = ab_table();
            let q = s.possible_symbols_suffix(&[b'b', b'a']);
            //println!("q = {:?}",q);
            let expected: Vec<(SymbolTableEntryId, &[u8])> = vec![(c.a, &[b'b'])];
            assert_eq!(q, expected);
        }
    }

    mod symbolification_suffix {
        use super::*;

        #[test]
        pub fn abc_empty() {
            let (s, _c) = abc_table();
            assert_eq!(s.symbolifications_suffix_str(""), vec![vec![]]);
        }

        #[test]
        pub fn ab_a() {
            let (s, c) = ab_table();
            assert_eq!(
                s.symbolifications_suffix_str("a"),
                vec![vec![c.a], vec![c.aa]]
            );
        }

        #[test]
        pub fn ab_b() {
            let (s, c) = ab_table();
            assert_eq!(
                s.symbolifications_suffix_str("b"),
                vec![vec![c.b], vec![c.ab]]
            );
        }

        #[test]
        pub fn ab_ba() {
            let (s, c) = ab_table();
            assert_eq!(
                s.symbolifications_suffix_str("ba"),
                vec![vec![c.b, c.a], vec![c.ab, c.a]]
            );
        }
    }

    #[test]
    pub fn test_serialize_table() {
        let (s, c) = default_symbol_table();
        let encoded: Vec<u8> = bincode::serialize(&s).unwrap();
        let decoded: SymbolTable<u8> = bincode::deserialize(&encoded[..]).unwrap();

        // Check the symbols are still OK
        assert_eq!(decoded.get_by_id(SymbolTableEntryId(123)), None);
        assert_eq!(decoded.get_by_id(c.start), Some(&SymbolTableEntry::Start));
        assert_eq!(decoded.get_by_id(c.end), Some(&SymbolTableEntry::End));
        assert_eq!(
            decoded.get_by_id(c.a),
            Some(&SymbolTableEntry::Single(b'a'))
        );
        assert_eq!(
            decoded.get_by_id(c.xyz),
            Some(&SymbolTableEntry::Compound(vec![b'x', b'y', b'z']))
        );
    }
}
