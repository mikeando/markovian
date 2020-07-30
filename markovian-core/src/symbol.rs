use ::serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

pub trait SymbolRender {
    fn render(&self) -> String;
}

impl SymbolRender for SymbolTableEntry<u8> {
    fn render(&self) -> String {
        match self {
            SymbolTableEntry::Start => "^".to_string(),
            SymbolTableEntry::End => "$".to_string(),
            SymbolTableEntry::Single(v) => String::from_utf8_lossy(&[*v]).to_string(),
            SymbolTableEntry::Compound(v) => String::from_utf8_lossy(&v).to_string(),
        }
    }
}

impl SymbolRender for SymbolTableEntry<char> {
    fn render(&self) -> String {
        match self {
            SymbolTableEntry::Start => "^".to_string(),
            SymbolTableEntry::End => "$".to_string(),
            SymbolTableEntry::Single(v) => format!("{}", v),
            SymbolTableEntry::Compound(v) => v.iter().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable<T> {
    values: BTreeMap<SymbolTableEntryId, SymbolTableEntry<T>>,
    next_key: SymbolTableEntryId,
}

impl<T> SymbolTable<T> {
    pub fn new() -> SymbolTable<T> {
        SymbolTable {
            values: BTreeMap::new(),
            next_key: SymbolTableEntryId(0),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn get_by_id(&self, id: SymbolTableEntryId) -> Option<&SymbolTableEntry<T>> {
        self.values.get(&id)
    }

    // Assumes there's only one start symbol...
    pub fn start_symbol_id(&self) -> SymbolTableEntryId {
        // TODO: should track these some other way.
        for (k, v) in &self.values {
            if matches!(v, SymbolTableEntry::Start) {
                return *k;
            }
        }
        unreachable!("No start symbol found")
    }

    pub fn end_symbol_id(&self) -> SymbolTableEntryId {
        // TODO: should track these some other way.
        for (k, v) in &self.values {
            if matches!(v, SymbolTableEntry::End) {
                return *k;
            }
        }
        unreachable!("No end symbol found")
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SymbolTableEntryId, &SymbolTableEntry<T>)> {
        self.values.iter()
    }
}

impl<T> SymbolTable<T>
where
    T: PartialEq,
{
    pub fn add(&mut self, value: SymbolTableEntry<T>) -> SymbolTableEntryId {
        //TODO: Keep a map so we dont need to scan for this
        for (k, v) in &self.values {
            if value == *v {
                return *k;
            }
        }
        let k = self.next_key;
        self.values.insert(k, value);
        self.next_key = SymbolTableEntryId(k.0 + 1);
        k
    }
}

impl<T> Default for SymbolTable<T> {
    fn default() -> Self {
        SymbolTable::new()
    }
}

impl<T> SymbolTable<T>
where
    SymbolTableEntry<T>: SymbolRender,
{
    pub fn render(&self, ids: &[SymbolTableEntryId]) -> String {
        let mut result: String = String::new();
        for id in ids {
            let s = self.get_by_id(*id);
            let ss = match s {
                Some(v) => v.render(),
                None => "?".to_string(),
            };
            result = format!("{}{}", result, ss);
        }
        result
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    InvalidSymbolify,
}

type Result<T> = std::result::Result<T, Error>;

impl<T> SymbolTable<T> {
    pub fn symbolify(&self, v: &[T]) -> Result<Vec<SymbolTableEntryId>>
    where
        T: Eq,
    {
        // TODO: Do this iteratively to build all matches.
        // recursively applying a function f: &[T] -> Iterator<(id, &[T])>
        // until array is empty
        let mut b: &[T] = v;
        let mut result: Vec<SymbolTableEntryId> = vec![];
        'outer: while !b.is_empty() {
            for (k, s) in &self.values {
                if s.matches_start(b) {
                    b = &b[s.length()..];
                    result.push(*k);
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
        for (id, s) in &self.values {
            if s.matches_start(v) {
                result.push((*id, &v[s.length()..]));
            }
        }
        result
    }

    pub fn possible_symbols_prefix<'a>(&self, v: &'a [T]) -> Vec<(SymbolTableEntryId, &'a [T])>
    where
        T: Eq,
    {
        let mut result = vec![];
        for (id, s) in &self.values {
            if s.matches_start_prefix(v) {
                let slice: &[T] = if s.length() >= v.len() {
                    &[]
                } else {
                    &v[s.length()..]
                };
                result.push((*id, slice));
            }
        }
        result
    }

    pub fn possible_symbols_suffix<'a>(&self, v: &'a [T]) -> Vec<(SymbolTableEntryId, &'a [T])>
    where
        T: Eq,
    {
        let mut result = vec![];
        for (id, s) in &self.values {
            if s.matches_end_suffix(v) {
                let slice: &[T] = if s.length() >= v.len() {
                    &[]
                } else {
                    &v[0..(v.len() - s.length())]
                };
                result.push((*id, slice));
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
    pub fn test_render_symbol_table_entry() {
        assert_eq!(SymbolTableEntry::Start.render(), "^");
        assert_eq!(SymbolTableEntry::End.render(), "$");
        assert_eq!(SymbolTableEntry::Single(b'a').render(), "a");
        assert_eq!(SymbolTableEntry::Compound(vec![b'x', b'y']).render(), "xy");
    }

    #[test]
    pub fn check_symbol_table_render() {
        let (s, c) = default_symbol_table();
        let u: String = s.render(&vec![]);
        assert_eq!(u, "");

        //TODO: Should this be an error?
        let u: String = s.render(&vec![SymbolTableEntryId(123)]);
        assert_eq!(u, "?");

        let u: String = s.render(&vec![c.start]);
        assert_eq!(u, "^");

        let u: String = s.render(&vec![c.end]);
        assert_eq!(u, "$");

        let u: String = s.render(&vec![c.a]);
        assert_eq!(u, "a");

        let u: String = s.render(&vec![c.xyz]);
        assert_eq!(u, "xyz");

        let u: String = s.render(&vec![c.start, c.a, c.xyz, c.end]);
        assert_eq!(u, "^axyz$");
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
            let (s, c) = default_symbol_table();
            // The empty string has one symbolification - the empty one.
            assert_eq!(
                s.symbolifications_str(""),
                vec![Vec::<SymbolTableEntryId>::new()]
            );
        }

        #[test]
        pub fn default_invalid() {
            let (s, c) = default_symbol_table();
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
            let (s, c) = default_symbol_table();
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
            let (s, c) = abc_table();
            assert_eq!(
                s.symbolifications_prefix_str(""),
                vec![Vec::<SymbolTableEntryId>::new()]
            );
        }

        #[test]
        pub fn abc_invalid() {
            let (s, c) = abc_table();
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
            let (s, c) = abc_table();
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
            let (s, c) = abc_table();
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
