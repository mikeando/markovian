use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use snafu::{ensure, OptionExt, Snafu};

use crate::vecutils::select_by_lowest_value;

#[derive(Debug, Snafu, Eq, PartialEq)]
pub enum Error {
    #[snafu(display("Invalid Symbolify"))]
    InvalidSymbolify,

    #[snafu(display("Invalid SymbolType: {:?}", symbol_type))]
    InvalidSymbolType { symbol_type: SymbolTableEntryType },

    #[snafu(display("Invalid symbol id: {}", symbol_id.0))]
    InvalidId { symbol_id: SymbolTableEntryId },

    #[snafu(display("Invalid operation: {}", mesg))]
    InvalidOperation { mesg: String },

    #[snafu(display("Non existent symbol"))]
    NoSuchSymbol,

    #[snafu(display("Internal Error"))]
    InternalError,
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Serialize, Deserialize, Hash)]
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
    pub fn from_vec(mut s: Vec<T>) -> Self {
        assert!(!s.is_empty());
        if s.len() == 1 {
            SymbolTableEntry::Single(s.swap_remove(0))
        } else {
            SymbolTableEntry::Compound(s)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTableLite<T> {
    index: Vec<SymbolTableEntry<T>>,
    dead_chain_head: Option<usize>,
    n_dead: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
        self.index
            .iter()
            .enumerate()
            .map(|(i, v)| (SymbolTableEntryId(i as u64), v))
    }
}

impl<T> SymbolTable<T>
where
    T: PartialEq + Ord + Clone + Debug,
{
    pub fn add(&mut self, value: SymbolTableEntry<T>) -> Result<SymbolTableEntryId> {
        if value == SymbolTableEntry::Start {
            return Ok(self.start_symbol_id());
        }
        if value == SymbolTableEntry::End {
            return Ok(self.end_symbol_id());
        }

        ensure!(
            !matches!(value, SymbolTableEntry::Dead(_)),
            InvalidOperation {
                mesg: "can not remove add dead entry"
            }
        );

        if let SymbolTableEntry::Compound(ref vs) = value {
            ensure!(
                !vs.is_empty(),
                InvalidOperation {
                    mesg: "can not add empty compound entry"
                }
            );
        }

        if let Some(id) = self.find(&value) {
            return Ok(id);
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

        Ok(id)
    }

    pub fn add_symbols(&mut self, ss: Vec<Vec<T>>) -> Result<()> {
        for s in ss {
            self.add(SymbolTableEntry::from_vec(s))?;
        }
        Ok(())
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

    // We don't actually remove the entry, just mark it as dead, so that we don't invalidate
    // existing ids.
    pub fn remove(&mut self, id: SymbolTableEntryId) -> Result<()> {
        ensure!(
            id.0 != 0,
            InvalidOperation {
                mesg: "can not remove start symbol (id=0)".to_string()
            }
        );
        ensure!(
            id.0 != 1,
            InvalidOperation {
                mesg: "can not remove end symbol (id=1)".to_string()
            }
        );

        let idx = (id.0 - 2) as usize;
        ensure!(idx < self.index.len(), InvalidId { symbol_id: id });

        let e = &self.index[(id.0 - 2) as usize];
        ensure!(
            !matches!(e, SymbolTableEntry::Dead(_)),
            InvalidOperation {
                mesg: format!("can not remove dead symbol (id={})", id.0)
            }
        );
        let first = e.first();

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
        Ok(())
    }

    // Remove all the dead entries. Potentially invalidating any existing ids.
    pub fn compact(&mut self) -> Result<SymbolRemapper> {
        self.dead_chain_head = None;
        self.n_dead = 0;

        // Build the list of moved ids
        let mut moved_ids: BTreeMap<SymbolTableEntryId, SymbolTableEntryId> = BTreeMap::new();

        moved_ids.insert(SymbolTableEntryId(0), SymbolTableEntryId(0));
        moved_ids.insert(SymbolTableEntryId(1), SymbolTableEntryId(1));

        let mut new_id: u64 = 2;
        for (i, e) in self.index.iter().enumerate() {
            match e {
                SymbolTableEntry::Start => unreachable!("Start should never be in index"),
                SymbolTableEntry::End => unreachable!("End should never be in index"),
                SymbolTableEntry::Single(_) | SymbolTableEntry::Compound(_) => {
                    moved_ids.insert(SymbolTableEntryId(new_id), SymbolTableEntryId(i as u64));
                    new_id += 1;
                }
                SymbolTableEntry::Dead(_) => {}
            }
        }

        self.index
            .retain(|e| !matches!(e, SymbolTableEntry::Dead(_)));

        self.by_first_character = BTreeMap::new();
        for (i, e) in self.index.iter().enumerate() {
            if let Some(t) = e.first().cloned() {
                self.by_first_character.entry(t).or_default().push(i);
            }
        }

        Ok(SymbolRemapper { moved_ids })
    }

    //Removes symbols and compresses the symbol table, but provides a way to map existing symbols to the new ones
    pub fn remove_symbols_and_compress(&mut self, ss: Vec<Vec<T>>) -> Result<SymbolRemapper> {
        for s in ss {
            let entry = SymbolTableEntry::from_vec(s);
            let eid = self.find(&entry).context(NoSuchSymbol)?;
            self.remove(eid)?;
        }
        self.compact()
    }
}

pub struct SymbolRemapper {
    moved_ids: BTreeMap<SymbolTableEntryId, SymbolTableEntryId>,
}

impl SymbolRemapper {
    pub fn map(&self, v: Vec<SymbolTableEntryId>) -> Option<Vec<SymbolTableEntryId>> {
        v.iter().map(|s| self.map_one(s)).collect()
    }

    pub fn map_one(&self, id: &SymbolTableEntryId) -> Option<SymbolTableEntryId> {
        self.moved_ids.get(id).cloned()
    }
}

impl<T> SymbolTable<T>
where
    T: PartialEq + Ord + Clone + std::fmt::Debug,
{
    //Panics if something is detected as "wrong"
    pub fn check_internal_consistency(&self) {
        // Check that there are no Start or Ends in the array
        {
            for e in &self.index {
                match e {
                    SymbolTableEntry::Start => {
                        panic!("Start found in symbol_table.index");
                    }
                    SymbolTableEntry::End => {
                        panic!("End found in symbol_table.index");
                    }
                    _ => {}
                }
            }
        }

        // Check that the dead-chain is valid, doesn't loop
        // and is in range.
        // Check that number dead is correct
        {
            let mut dead_ids: BTreeSet<usize> = BTreeSet::new();
            let mut dead_link = self.dead_chain_head;
            while dead_link.is_some() {
                let next_index = dead_link.unwrap();
                if dead_ids.contains(&next_index) {
                    panic!("Loop found in symbol table dead chain");
                }

                let next = &self.index[next_index];
                if let SymbolTableEntry::Dead(d) = next {
                    dead_ids.insert(next_index);
                    dead_link = *d;
                } else {
                    panic!("Non-dead node found in dead chain")
                }
            }

            assert_eq!(dead_ids.len(), self.n_dead);
        }

        // Check that there are no duplicate contents
        {
            let mut c = BTreeSet::<Either<T, Vec<T>>>::new();
            for entry in &self.index {
                let entry: Option<Either<T, Vec<T>>> = match entry {
                    SymbolTableEntry::Single(v) => Some(Either::Left(v.clone())),
                    SymbolTableEntry::Compound(vs) => Some(Either::Right(vs.clone())),
                    _ => None,
                };
                if let Some(entry) = entry {
                    let uniq = c.insert(entry);
                    assert!(uniq, "duplicate entry found");
                }
            }
        }

        // Check that the map of first character to index is correct
        {
            let mut first_to_index = BTreeMap::<T, BTreeSet<usize>>::new();
            for (index, entry) in self.index.iter().enumerate() {
                let first = entry.first();
                if let Some(first) = first {
                    first_to_index
                        .entry(first.clone())
                        .or_insert_with(BTreeSet::new)
                        .insert(index);
                }
            }

            let mut mod_by_first_character = BTreeMap::<T, BTreeSet<usize>>::new();
            for (k, v) in &self.by_first_character {
                if !v.is_empty() {
                    let s: BTreeSet<usize> = v.iter().copied().collect();
                    mod_by_first_character.insert(k.clone(), s);
                }
            }

            assert_eq!(
                first_to_index, mod_by_first_character,
                "First character indexes do not match: {:?}",
                self
            );
        }
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<T> Default for SymbolTable<T>
where
    T: Ord + Clone,
{
    fn default() -> Self {
        SymbolTable::new()
    }
}

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
            // We don't need to consider START and END as they can never match
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
            let (_parent_id, visited, _symbol, rest) = stack[current_index as usize];
            if visited {
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
    // render(SS).starts_with(v) but !render(&SS[..SS.len()-1]).starts_with(v)
    // This means in practice that the last symbol of an allowed sequence can overhang the
    // end of v slightly. For example if our symbols are "ab" and "ac", "a"
    // then a prefix symbolification of "aba" will contain not just the exact match ["ab","a"],
    // but two that overhang by one ["ab","ab"] and ["ab", "ac"]
    //
    // symbolifications_prefix("aba") === [ ["ab", "ab"], ["ab","ac"], ["ab","a"]]
    //
    // The main reason for this is to help with generation of sequences that start with a
    // prefix.
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
            SymbolTableEntry::Start => InvalidSymbolType {
                symbol_type: SymbolTableEntryType::Start,
            }
            .fail(),
            SymbolTableEntry::End => InvalidSymbolType {
                symbol_type: SymbolTableEntryType::End,
            }
            .fail(),
            SymbolTableEntry::Single(e) => {
                v.push(e.clone());
                Ok(())
            }
            SymbolTableEntry::Compound(es) => {
                v.extend_from_slice(&es);
                Ok(())
            }
            SymbolTableEntry::Dead(_) => InvalidSymbolType {
                symbol_type: SymbolTableEntryType::Dead,
            }
            .fail(),
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

pub fn shortest_symbolifications<T>(
    symbol_table: &SymbolTable<T>,
    v: &[T],
) -> Vec<Vec<SymbolTableEntryId>>
where
    T: Eq + Clone + Ord,
{
    let v = symbol_table.symbolifications(v);
    select_by_lowest_value(&v, &|s: &Vec<SymbolTableEntryId>| s.len())
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

        let start = result.add(SymbolTableEntry::Start).unwrap();
        let end = result.add(SymbolTableEntry::End).unwrap();
        let a = result.add(SymbolTableEntry::Single(b'a')).unwrap();
        let xyz = result
            .add(SymbolTableEntry::Compound(vec![b'x', b'y', b'z']))
            .unwrap();

        (result, ContextDefault { start, end, a, xyz })
    }

    struct ContextABC {
        a: SymbolTableEntryId,
        ab: SymbolTableEntryId,
        ac: SymbolTableEntryId,
    }

    fn abc_table() -> (SymbolTable<u8>, ContextABC) {
        let mut s = SymbolTable::new();
        let a = s.add(SymbolTableEntry::Single(b'a')).unwrap();
        let ab = s.add(SymbolTableEntry::Compound(vec![b'a', b'b'])).unwrap();
        let ac = s.add(SymbolTableEntry::Compound(vec![b'a', b'c'])).unwrap();
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
        let a = s.add(SymbolTableEntry::Single(b'a')).unwrap();
        let aa = s.add(SymbolTableEntry::Compound(vec![b'a', b'a'])).unwrap();
        let b = s.add(SymbolTableEntry::Single(b'b')).unwrap();
        let ab = s.add(SymbolTableEntry::Compound(vec![b'a', b'b'])).unwrap();
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

    #[test]
    pub fn remove_invalid_symbolid_returns_error() {
        let mut s: SymbolTable<u8> = SymbolTable::new();
        let e = s.remove(SymbolTableEntryId(100));
        assert_eq!(
            e,
            Err(Error::InvalidId {
                symbol_id: SymbolTableEntryId(100)
            })
        );
    }

    #[test]
    pub fn remove_start_symbol_id_returns_error() {
        let mut s: SymbolTable<u8> = SymbolTable::new();
        let e = s.remove(SymbolTableEntryId(0));
        assert_eq!(
            e,
            Err(Error::InvalidOperation {
                mesg: "can not remove start symbol (id=0)".to_string()
            })
        );
    }

    #[test]
    pub fn remove_end_symbol_id_returns_error() {
        let mut s: SymbolTable<u8> = SymbolTable::new();
        let e = s.remove(SymbolTableEntryId(1));
        assert_eq!(
            e,
            Err(Error::InvalidOperation {
                mesg: "can not remove end symbol (id=1)".to_string()
            })
        );
    }

    #[test]
    pub fn add_dead_entry_errors() {
        let mut s: SymbolTable<u8> = SymbolTable::new();
        let entry = SymbolTableEntry::Dead(None);
        let e = s.add(entry);
        assert_eq!(
            e,
            Err(Error::InvalidOperation {
                mesg: "can not remove add dead entry".to_string()
            })
        );
    }

    #[test]
    pub fn cant_add_empty_compound() {
        let mut s: SymbolTable<char> = SymbolTable::new();
        let e = s.add(SymbolTableEntry::Compound(vec![]));
        assert_eq!(
            e,
            Err(Error::InvalidOperation {
                mesg: "can not add empty compound entry".to_string()
            })
        );
    }

    #[test]
    pub fn double_remove_is_ok_01() {
        let mut s: SymbolTable<char> = SymbolTable::new();
        let _ = s.add(SymbolTableEntry::Compound(vec!['a']));
        s.check_internal_consistency();
        let _ = s.remove(SymbolTableEntryId(2));
        s.check_internal_consistency();
        let _ = s.remove(SymbolTableEntryId(2));
        s.check_internal_consistency();
    }

    #[test]
    pub fn double_remove_is_ok_02() {
        let mut s: SymbolTable<char> = SymbolTable::new();
        let _ = s.add(SymbolTableEntry::Compound(vec!['a']));
        s.check_internal_consistency();
        let _ = s.add(SymbolTableEntry::Compound(vec!['b']));
        s.check_internal_consistency();
        let _ = s.remove(SymbolTableEntryId(3));
        s.check_internal_consistency();
        let _ = s.remove(SymbolTableEntryId(3));
        s.check_internal_consistency();
    }
}
