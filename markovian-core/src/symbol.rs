use ::serde::{Deserialize, Serialize};

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum Symbol {
    Start,
    End,
    Char(u8),
    Compound(Vec<u8>),
}

pub fn raw_symbolify_word(s: &str) -> Vec<Symbol> {
    s.as_bytes().iter().cloned().map(Symbol::Char).collect()
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Serialize, Deserialize)]
pub struct SymbolTableEntryId(pub u64);

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
pub struct SymbolTable<T> {
    index: Vec<SymbolTableEntry<T>>,
    dead_chain_head: Option<usize>,
    n_dead: usize,
}

impl<T> SymbolTable<T> {
    pub fn new() -> SymbolTable<T> {
        SymbolTable {
            index: vec![],
            dead_chain_head: None,
            n_dead: 0,
        }
    }

    // Includes dead entries.
    pub fn len(&self) -> usize {
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
    T: PartialEq,
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

        match self.dead_chain_head {
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
        }
    }

    pub fn find(&self, value: &SymbolTableEntry<T>) -> Option<SymbolTableEntryId> {
        //TODO: Keep a map so we dont need to scan for this
        for (k, v) in self.index.iter().enumerate() {
            if value == v {
                return Some(SymbolTableEntryId((k + 2) as u64));
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
    }
}

impl<T> Default for SymbolTable<T> {
    fn default() -> Self {
        SymbolTable::new()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    InvalidSymbolify,
}

type Result<T> = std::result::Result<T, Error>;

impl<T> SymbolTable<T> {
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
        for (id, s) in self.index.iter().enumerate() {
            if s.matches_start(v) {
                result.push((SymbolTableEntryId((id + 2) as u64), &v[s.length()..]));
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

        let mut result = vec![];
        for (id, rest) in self.possible_symbols(v) {
            for mut child in self.symbolifications(rest) {
                child.insert(0, id);
                result.push(child)
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
        T: Eq + std::fmt::Debug,
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

        #[test]
        pub fn ab_ambiguous() {
            let (s, c) = ab_table();
            assert_eq!(
                s.symbolifications_str("aa"),
                vec![vec![c.a, c.a], vec![c.aa]]
            );
            assert_eq!(
                s.symbolifications_str("aaa"),
                vec![vec![c.a, c.a, c.a], vec![c.a, c.aa], vec![c.aa, c.a]]
            );
            assert_eq!(
                s.symbolifications_str("aab"),
                vec![vec![c.a, c.a, c.b], vec![c.a, c.ab], vec![c.aa, c.b]]
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
