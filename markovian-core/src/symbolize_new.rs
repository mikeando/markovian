#![allow(dead_code)]
// Until we've hooked it all up, a lot of it will show as dead
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Display};
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum SymbolizeError {
    NoPathsToZero,
    NodeIdTooLarge(usize, usize),
    TooManyPathsToNode(usize),
    InputLengthTooLong(usize, usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub struct SymbolId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathComponent {
    from_state: usize,
    symbol: SymbolId,
}

impl PathComponent {
    pub fn invalid() -> PathComponent {
        PathComponent {
            from_state: usize::MAX,
            symbol: SymbolId(usize::MAX),
        }
    }
}

pub struct ForwardPathTable {}

#[derive(Copy, Clone, Debug)]
enum TrieState {
    Present,
    Empty,
    Done,
}

#[derive(Debug)]
struct Trie<'a> {
    m: BTreeMap<&'a [u8], TrieState>,
}

impl<'a> Trie<'a> {
    pub fn contains(&self, v: &[u8]) -> TrieState {
        self.m.get(v).copied().unwrap_or(TrieState::Done)
    }

    pub fn add<'b>(&mut self, v: &'b [u8])
    where
        'b: 'a,
    {
        if v.len() > 1 {
            for i in 0..v.len() - 1 {
                let w = &v[..=i];
                self.m.entry(w).or_insert(TrieState::Empty);
            }
        }
        self.m.insert(v, TrieState::Present);
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ProgressiveIndexState<T> {
    Present(T),
    Prefix,
    NotPresent,
}

pub trait ProgressiveIndex<K, T> {
    fn contains(&self, key: &[K]) -> &ProgressiveIndexState<T>;
}

impl<'a> ProgressiveIndex<char, SymbolId> for HashMap<&'a [char], ProgressiveIndexState<SymbolId>> {
    fn contains(&self, key: &[char]) -> &ProgressiveIndexState<SymbolId> {
        self.get(key).unwrap_or(&ProgressiveIndexState::NotPresent)
    }
}

impl<'a> ProgressiveIndex<u8, SymbolId> for HashMap<&'a [u8], ProgressiveIndexState<SymbolId>> {
    fn contains(&self, key: &[u8]) -> &ProgressiveIndexState<SymbolId> {
        self.get(key).unwrap_or(&ProgressiveIndexState::NotPresent)
    }
}

pub fn symbolize_new2<T, I>(to_symbolize: &[T], symbol_index: &I) -> Vec<Vec<SymbolId>>
where
    I: ProgressiveIndex<T, SymbolId>,
{
    if to_symbolize.is_empty() {
        return vec![vec![]];
    }

    // TODO: Avoid allocations while building up these tables.
    let mut fwd_states: Vec<Option<(usize, Vec<PathComponent>)>> = vec![None; to_symbolize.len()];

    for i in 0..to_symbolize.len() {
        if (i == 0) || (fwd_states[i - 1] != None) {
            let cw = if i == 0 {
                0
            } else {
                fwd_states[i - 1].as_ref().unwrap().0
            };
            for j in i..to_symbolize.len() {
                let v = &to_symbolize[i..j + 1];
                let q = symbol_index.contains(&v);
                match q {
                    ProgressiveIndexState::Present(symbol) => {
                        let symbol = *symbol;
                        let path_len = cw + 1;
                        // Check if we've got a new or better way to get to j.
                        let old_state = std::mem::replace(&mut fwd_states[j], None);
                        let new_state = match old_state {
                            None => (
                                path_len,
                                vec![PathComponent {
                                    from_state: i,
                                    symbol,
                                }],
                            ),
                            Some((old_path_length, mut paths)) if old_path_length == path_len => {
                                paths.push(PathComponent {
                                    from_state: i,
                                    symbol,
                                });
                                (path_len, paths)
                            }
                            Some((old_path_length, _paths)) if old_path_length > path_len => (
                                path_len,
                                vec![PathComponent {
                                    from_state: i,
                                    symbol,
                                }],
                            ),
                            Some((old_path_length, paths)) if old_path_length < path_len => {
                                (old_path_length, paths)
                            }
                            Some(_) => unreachable!(),
                        };
                        fwd_states[j] = Some(new_state);
                    }
                    ProgressiveIndexState::Prefix => continue,
                    ProgressiveIndexState::NotPresent => break,
                }
            }
        }
    }

    let c = fwd_states.len();

    let mut completed_paths: Vec<Vec<SymbolId>> = vec![];
    let mut incomplete_paths: Vec<(usize, Vec<SymbolId>)> = vec![(c, vec![])];

    while !incomplete_paths.is_empty() {
        let (c, p) = incomplete_paths.pop().unwrap();

        if let Some(s) = &fwd_states[c - 1] {
            for ps in &s.1 {
                let c_next = ps.from_state;
                let mut p_next = p.clone();
                p_next.push(ps.symbol);
                if c_next == 0 {
                    p_next.reverse();
                    completed_paths.push(p_next);
                } else {
                    incomplete_paths.push((c_next, p_next));
                }
            }
        }
    }

    completed_paths
}

pub struct SymbolizeState<'a> {
    pub forward: FwdState<'a>,
    pub back: BackState<'a>,
    /*
    fwd_state_components: &'a mut [PathComponent],
    n_result_paths: usize,
    result_path_lengths: &'a mut [usize],
    result_path_offsets: &'a mut [usize],
    result_paths: &'a mut [SymbolId],
    */
}

#[derive(Copy, Clone, Debug)]
pub struct SymbolizeNode(usize);
#[derive(Copy, Clone, Debug)]
pub struct SymbolizePathId(usize);

pub struct PathInfo {
    processed: bool,
    n: SymbolizeNode,
}

impl<'a> SymbolizeState<'a> {
    //TODO: This is expensive - we should reduce the amount of work this really needs to do.
    pub fn clear(&mut self) {
        /*
        self.n_result_paths = 0;
        for i in 0..self.fwd_state_components.len() {
            self.fwd_state_components[i] = PathComponent{ from_state: usize::MAX, symbol: SymbolId(usize::MAX) };
        }
        for i in 0..self.result_path_lengths.len() {
            self.result_path_lengths[i] = usize::MAX;
        }
        for i in 0..self.result_path_offsets.len() {
            self.result_path_offsets[i] = usize::MAX;
        }
        for i in 0..self.result_paths.len() {
            self.result_paths[i] = SymbolId(usize::MAX);
        }
        */
    }

    /*
    pub fn new() -> SymbolizeState {
        SymbolizeState{
            forward: FwdState::new(),
            back: BackState::new(),
        }
    }
    */
}

pub struct FwdBacking {
    n_symbols: usize,
    to_path_count: Vec<usize>,
    best_total_length: Vec<usize>,
    paths: Vec<PathComponent>,
}

impl FwdBacking {
    pub fn allocate(n_symbols: usize) -> FwdBacking {
        FwdBacking {
            n_symbols,
            to_path_count: vec![0; n_symbols + 1],
            best_total_length: vec![0; n_symbols + 1],
            paths: vec![PathComponent::invalid(); (n_symbols * (n_symbols + 1)) / 2],
        }
    }

    pub fn state(&mut self) -> FwdState {
        FwdState {
            to_path_count: &mut self.to_path_count,
            best_total_length: &mut self.best_total_length,
            paths: &mut self.paths,
            max_input_length: self.n_symbols,
            input_length: 0,
        }
    }
}

#[derive(Debug)]
pub struct FwdState<'a> {
    to_path_count: &'a mut [usize],
    best_total_length: &'a mut [usize],
    paths: &'a mut [PathComponent],
    max_input_length: usize,
    input_length: usize,
}

impl<'a> FwdState<'a> {
    pub fn new(
        to_path_count: &'a mut [usize],
        best_total_length: &'a mut [usize],
        paths: &'a mut [PathComponent],
        max_input_length: usize,
    ) -> FwdState<'a> {
        FwdState {
            to_path_count,
            best_total_length,
            paths,
            max_input_length,
            input_length: 0,
        }
    }

    pub fn set_input_length(&mut self, input_length: usize) -> Result<(), SymbolizeError> {
        if input_length > self.max_input_length {
            return Err(SymbolizeError::InputLengthTooLong(
                input_length,
                self.max_input_length,
            ));
        }
        self.input_length = input_length;
        for i in 0..=self.input_length {
            self.to_path_count[i] = 0;
        }
        Ok(())
    }

    pub fn total_distance_to_node(&self, node: SymbolizeNode) -> Option<usize> {
        let nid = node.0;
        if nid == 0 {
            Some(0)
        } else if self.to_path_count[nid] > 0 {
            Some(self.best_total_length[nid])
        } else {
            None
        }
    }

    pub fn maybe_add_path(
        &mut self,
        from: SymbolizeNode,
        to: SymbolizeNode,
        symbol: SymbolId,
        total_length: usize,
    ) -> Result<(), SymbolizeError> {
        let nid = to.0;
        if nid == 0 {
            return Err(SymbolizeError::NoPathsToZero);
        }
        if nid > self.input_length {
            return Err(SymbolizeError::NodeIdTooLarge(nid, self.input_length));
        }
        if self.to_path_count[nid] == 0 || total_length < self.best_total_length[nid] {
            self.to_path_count[nid] = 1;
            self.best_total_length[nid] = total_length;
            let pid = nid * (nid - 1) / 2;
            self.paths[pid] = PathComponent {
                from_state: from.0,
                symbol,
            };
        } else if total_length == self.best_total_length[nid] {
            if self.to_path_count[nid] >= nid {
                return Err(SymbolizeError::TooManyPathsToNode(nid));
            }
            let pid = nid * (nid - 1) / 2 + self.to_path_count[nid];
            self.to_path_count[nid] += 1;
            self.paths[pid] = PathComponent {
                from_state: from.0,
                symbol,
            };
        }
        Ok(())
    }

    pub fn paths_to(&self, node: SymbolizeNode) -> &[PathComponent] {
        let nid = node.0;
        assert!(nid <= self.input_length);
        if nid == 0 {
            return &[];
        }
        let pid = nid * (nid - 1) / 2;
        let npaths = self.to_path_count[nid];
        &self.paths[pid..pid + npaths]
    }
}

pub struct BackBacking {
    max_length: usize,
    path_symbols: Vec<SymbolId>,
    cur_path: Vec<(SymbolizeNode, usize)>,
}

impl BackBacking {
    pub fn allocate(max_length: usize) -> BackBacking {
        BackBacking {
            max_length,
            path_symbols: vec![SymbolId(usize::MAX); max_length],
            cur_path: vec![(SymbolizeNode(usize::MAX), usize::MAX); max_length],
        }
    }

    pub fn state(&mut self) -> BackState {
        BackState {
            max_length: self.max_length,
            input_length: 0,
            path_length: 0,
            path_symbols: &mut self.path_symbols,
            cur_path: &mut self.cur_path,
        }
    }
}

pub struct BackState<'a> {
    max_length: usize,
    input_length: usize,
    path_length: usize,
    path_symbols: &'a mut [SymbolId],
    cur_path: &'a mut [(SymbolizeNode, usize)],
}

struct WU(usize);

impl Display for WU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == usize::MAX {
            write!(f, "X")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl<'a> Debug for BackState<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BackState{{")?;
        write!(f, "  ")?;
        for i in 0..self.path_length {
            write!(
                f,
                " [{},{}]",
                WU(self.cur_path[i].0 .0),
                WU(self.cur_path[i].1)
            )?;
        }
        write!(f, "|")?;
        for i in self.path_length..self.cur_path.len() {
            write!(
                f,
                " [{},{}]",
                WU(self.cur_path[i].0 .0),
                WU(self.cur_path[i].1)
            )?;
        }
        writeln!(f)?;

        for i in 0..self.path_symbols.len() {
            write!(f, " {}", WU(self.path_symbols[i].0))?;
        }
        writeln!(f)?;

        writeln!(f, "}}")
    }
}

pub trait PathSource {
    fn final_state(&self) -> SymbolizeNode;
    fn to_node(&self, node: SymbolizeNode) -> &[PathComponent];
}

impl<'a> PathSource for FwdState<'a> {
    fn final_state(&self) -> SymbolizeNode {
        SymbolizeNode(self.input_length)
    }

    fn to_node(&self, node: SymbolizeNode) -> &[PathComponent] {
        self.paths_to(node)
    }
}

impl<'a> BackState<'a> {
    pub fn reset(&mut self) {
        self.path_length = 0;
    }

    pub fn extend_initial<PS: PathSource>(&mut self, ps: &PS) {
        assert!(self.path_length > 0);
        let cur = self.cur_path[self.path_length - 1];
        let mut pc = &ps.to_node(cur.0)[cur.1];
        while pc.from_state != 0 {
            let cur = (SymbolizeNode(pc.from_state), 0);
            self.path_length += 1;
            self.cur_path[self.path_length - 1] = cur;
            pc = &ps.to_node(cur.0)[cur.1];
            self.path_symbols[self.path_symbols.len() - 1 - (self.path_length - 1)] = pc.symbol;
        }
    }

    pub fn trim_saturated<PS: PathSource>(&mut self, ps: &PS) {
        loop {
            if self.path_length == 0 {
                return;
            }
            let cur = self.cur_path[self.path_length - 1];
            let pc = ps.to_node(cur.0);
            if cur.1 != pc.len() - 1 {
                return;
            }
            self.path_length -= 1;
        }
    }

    pub fn next<PS: PathSource>(&mut self, ps: &PS) -> Option<&[SymbolId]> {
        if self.path_length == 0 {
            let fs = ps.final_state();
            let pc = ps.to_node(fs);
            if pc.is_empty() {
                return None;
            }
            self.cur_path[0] = (fs, 0);
            self.path_symbols[self.path_symbols.len() - 1] = pc[0].symbol;
            self.path_length = 1;
            self.extend_initial(ps);
        } else {
            self.trim_saturated(ps);
            if self.path_length == 0 {
                return None;
            }
            let p = &mut self.cur_path[self.path_length - 1];
            p.1 += 1;
            self.path_symbols[self.path_symbols.len() - self.path_length] =
                ps.to_node(p.0)[p.1].symbol;
            self.extend_initial(ps);
        }
        Some(&self.path_symbols[self.path_symbols.len() - self.path_length..])
    }
}

pub fn symbolize_new3<T, I, F>(
    to_symbolize: &[T],
    symbol_index: &I,
    state: &mut SymbolizeState,
    mut on_each: F,
) where
    I: ProgressiveIndex<T, SymbolId>,
    F: FnMut(&[SymbolId]),
{
    let fwd: &mut FwdState = &mut state.forward;

    if to_symbolize.is_empty() {
        on_each(&[]);
        return;
    }

    let c = to_symbolize.len();
    fwd.set_input_length(c).unwrap();

    for i in 0..c {
        let ni = SymbolizeNode(i);
        match fwd.total_distance_to_node(ni) {
            None => continue,
            Some(distance) => {
                for j in i..c {
                    let nj = SymbolizeNode(j + 1);
                    let v = &to_symbolize[i..j + 1];
                    let q = symbol_index.contains(&v);
                    match q {
                        ProgressiveIndexState::Present(s) => {
                            fwd.maybe_add_path(ni, nj, *s, distance + 1).unwrap();
                        }
                        ProgressiveIndexState::Prefix => continue,
                        ProgressiveIndexState::NotPresent => break,
                    }
                }
            }
        }
    }

    let back: &mut BackState = &mut state.back;
    back.reset();

    let fwd: &FwdState = &state.forward;

    while let Some(x) = back.next(fwd) {
        on_each(x);
    }
}

pub fn symbolize_new<T>(
    to_symbolize: &[T],
    symbol_index: &HashMap<&[T], SymbolId>,
) -> Vec<Vec<SymbolId>>
where
    T: Ord + Hash,
{
    if to_symbolize.is_empty() {
        return vec![vec![]];
    }

    // TODO: Avoid allocations while building up these tables.
    let mut fwd_states: Vec<Option<(usize, Vec<PathComponent>)>> = vec![None; to_symbolize.len()];

    for i in 0..to_symbolize.len() {
        if (i == 0) || (fwd_states[i - 1] != None) {
            let cw = if i == 0 {
                0
            } else {
                fwd_states[i - 1].as_ref().unwrap().0
            };
            for j in i..to_symbolize.len() {
                let v = &to_symbolize[i..j + 1];
                let q = symbol_index.get(&v);
                match q {
                    None => {}
                    Some(&symbol) => {
                        let path_len = cw + 1;
                        // Check if we've got a new or better way to get to j.
                        if fwd_states[j] == None {
                            fwd_states[j] = Some((
                                path_len,
                                vec![PathComponent {
                                    from_state: i,
                                    symbol,
                                }],
                            ));
                        } else if fwd_states[j].as_ref().unwrap().0 == path_len {
                            fwd_states[j].as_mut().unwrap().1.push(PathComponent {
                                from_state: i,
                                symbol,
                            });
                        } else if fwd_states[j].as_ref().unwrap().0 > path_len {
                            fwd_states[j] = Some((
                                path_len,
                                vec![PathComponent {
                                    from_state: i,
                                    symbol,
                                }],
                            ));
                        }
                    }
                }
            }
        }
    }

    let c = fwd_states.len();

    let mut completed_paths: Vec<Vec<SymbolId>> = vec![];
    let mut incomplete_paths: Vec<(usize, Vec<SymbolId>)> = vec![(c, vec![])];

    while !incomplete_paths.is_empty() {
        let (c, p) = incomplete_paths.pop().unwrap();

        if let Some(s) = &fwd_states[c - 1] {
            for ps in &s.1 {
                let c_next = ps.from_state;
                let mut p_next = p.clone();
                p_next.push(ps.symbol);
                if c_next == 0 {
                    p_next.reverse();
                    completed_paths.push(p_next);
                } else {
                    incomplete_paths.push((c_next, p_next));
                }
            }
        }
    }

    completed_paths
}

#[cfg(test)]
pub mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use super::{
        symbolize_new,
        symbolize_new3,
        BackBacking,
        BackState,
        FwdBacking,
        ProgressiveIndexState,
        SymbolId,
        SymbolizeState,
    };

    pub fn to_char_vec(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    pub fn render_symbols_with_sep(v: &[SymbolId], symbols: &[&str], sep: &str) -> String {
        let mut result = String::new();
        for (i, s) in v.iter().enumerate() {
            if i != 0 {
                result = format!("{}{}{}", result, sep, symbols[s.0]);
            } else {
                result = format!("{}", symbols[s.0]);
            }
        }
        result
    }

    fn set(vs: &[&str]) -> BTreeSet<String> {
        vs.iter().map(|v| v.to_string()).collect()
    }

    fn to_symbol_list(symbols: &[&str]) -> Vec<Vec<char>> {
        let mut result = vec![];
        for ss in symbols {
            result.push(ss.chars().collect());
        }
        result
    }

    fn symbolize_and_render(to_symbolize: &str, symbols: &[&str]) -> BTreeSet<String> {
        let ss: Vec<Vec<char>> = to_symbol_list(&symbols);
        let symbol_index: HashMap<&[char], SymbolId> = ss
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_slice(), SymbolId(i)))
            .collect();
        let to_symbolize = to_char_vec(to_symbolize);

        let completed_paths = symbolize_new(&to_symbolize, &symbol_index);

        completed_paths
            .iter()
            .map(|sids| render_symbols_with_sep(&sids, &symbols, "."))
            .collect()
    }

    #[test]
    pub fn new_symbolize() {
        assert_eq!(
            set(&["ab.a", "a.ba"]),
            symbolize_and_render("aba", &["a", "ab", "ba", "b"])
        );
        assert_eq!(set(&[""]), symbolize_and_render("", &["a"]));
        assert_eq!(
            set(&["aa.a", "a.aa"]),
            symbolize_and_render("aaa", &["a", "aa"])
        );
        assert_eq!(set(&[]), symbolize_and_render("c", &["a"]));
    }

    pub mod symbolize3 {
        use super::*;

        fn symbolize_and_render3(to_symbolize: &str, symbols: &[&str]) -> BTreeSet<String> {
            let ss: Vec<Vec<char>> = to_symbol_list(&symbols);
            let mut symbol_index: HashMap<&[char], ProgressiveIndexState<SymbolId>> =
                HashMap::new();
            for (i, s) in ss.iter().enumerate() {
                if s.len() > 1 {
                    for j in 0..s.len() - 1 {
                        symbol_index
                            .entry(&s[0..j])
                            .or_insert(ProgressiveIndexState::Prefix);
                    }
                }
                symbol_index.insert(&s[..], ProgressiveIndexState::Present(SymbolId(i)));
            }
            let to_symbolize = to_char_vec(to_symbolize);

            let mut fwd = FwdBacking::allocate(10);
            let mut back = BackBacking::allocate(10);

            let mut state = SymbolizeState {
                forward: fwd.state(),
                back: back.state(),
            };
            let mut completed_paths: Vec<Vec<SymbolId>> = vec![];

            symbolize_new3(&to_symbolize, &symbol_index, &mut state, |ss| {
                completed_paths.push(ss.to_vec());
            });

            completed_paths
                .iter()
                .map(|sids| render_symbols_with_sep(&sids, &symbols, "."))
                .collect()
        }

        #[test]
        pub fn no_path() {
            assert_eq!(set(&[]), symbolize_and_render3("c", &["a"]));
        }

        #[test]
        pub fn single_value() {
            assert_eq!(set(&["a"]), symbolize_and_render3("a", &["a"]));
        }

        #[test]
        pub fn empty_string() {
            assert_eq!(set(&[""]), symbolize_and_render3("", &["a"]));
        }

        #[test]
        pub fn simple_aba() {
            assert_eq!(
                set(&["ab.a", "a.ba"]),
                symbolize_and_render3("aba", &["a", "ab", "ba", "b"])
            );
        }

        #[test]
        pub fn simple_aaa() {
            assert_eq!(
                set(&["aa.a", "a.aa"]),
                symbolize_and_render3("aaa", &["a", "aa"])
            );
        }
    }

    pub mod test_fwd {
        use crate::symbolize_new::{FwdBacking, FwdState, SymbolId, SymbolizeError, SymbolizeNode};

        #[test]
        pub fn check_adding_path() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();

            ff.set_input_length(3).unwrap();

            assert_eq!(ff.paths_to(SymbolizeNode(1)), &[]);
            ff.maybe_add_path(SymbolizeNode(0), SymbolizeNode(1), SymbolId(0), 1)
                .unwrap();

            assert_eq!(ff.paths_to(SymbolizeNode(1)).len(), 1);
        }

        #[test]
        pub fn check_adding_path_for_node_zero() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();

            ff.set_input_length(3).unwrap();
            let fail = ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(0), SymbolId(0), 1);
            assert_eq!(fail, Err(SymbolizeError::NoPathsToZero));
        }

        #[test]
        pub fn check_adding_path_for_too_large_node() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            ff.set_input_length(3).unwrap();

            // We've set the length to 3 so we should be able to add to node 3, but not 4.
            assert_eq!(
                ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(3), SymbolId(0), 1),
                Ok(())
            );
            assert_eq!(
                ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(4), SymbolId(0), 1),
                Err(SymbolizeError::NodeIdTooLarge(4, 3))
            );
        }

        #[test]
        pub fn check_repeated_add_fails() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            ff.set_input_length(3).unwrap();

            // We've set the length to 3 so we should be able to add to nodes less than or equal to 3
            // The number of times we're able to add them should be nid exactly.
            for nid in 1..=3 {
                for j in 1..=nid {
                    assert_eq!(
                        ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(nid), SymbolId(j), 1),
                        Ok(())
                    );
                }
                assert_eq!(
                    ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(nid), SymbolId(100), 1),
                    Err(SymbolizeError::TooManyPathsToNode(nid))
                );
            }
        }

        #[test]
        pub fn check_repeated_add_fails_full() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            ff.set_input_length(4).unwrap();

            // We've set the length to 3 so we should be able to add to nodes less than or equal to 3
            // The number of times we're able to add them should be nid exactly.
            for nid in 1..=4 {
                for j in 1..=nid {
                    assert_eq!(
                        ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(nid), SymbolId(j), 1),
                        Ok(())
                    );
                }
                assert_eq!(
                    ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(nid), SymbolId(100), 1),
                    Err(SymbolizeError::TooManyPathsToNode(nid))
                );
            }
        }

        #[test]
        pub fn check_invalid_input_length() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            assert_eq!(
                ff.set_input_length(5),
                Err(SymbolizeError::InputLengthTooLong(5, 4))
            );
        }

        #[test]
        pub fn check_clearing_resets_all_lengths() {
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            ff.set_input_length(4).unwrap();

            // We've set the length to 3 so we should be able to add to nodes less than or equal to 3
            // The number of times we're able to add them should be nid exactly.
            for nid in 1..=4 {
                for j in 1..=nid {
                    ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(nid), SymbolId(j), nid)
                        .unwrap();
                }
            }

            for i in 0..=4 {
                assert_eq!(ff.total_distance_to_node(SymbolizeNode(i)), Some(i));
                assert_eq!(ff.paths_to(SymbolizeNode(i)).len(), i);
            }

            ff.set_input_length(4).unwrap();

            assert_eq!(ff.total_distance_to_node(SymbolizeNode(0)), Some(0));
            assert_eq!(ff.paths_to(SymbolizeNode(0)).len(), 0);

            for i in 1..=4 {
                assert_eq!(
                    ff.total_distance_to_node(SymbolizeNode(i)),
                    None,
                    "length for node {}",
                    i
                );
                assert_eq!(ff.paths_to(SymbolizeNode(i)).len(), 0);
            }
        }
    }

    pub mod test_back {
        use std::collections::BTreeMap;

        use crate::symbolize_new::{
            BackBacking,
            BackState,
            FwdBacking,
            FwdState,
            PathComponent,
            PathSource,
            SymbolId,
            SymbolizeError,
            SymbolizeNode,
        };

        #[test]
        pub fn simple_linear_case() -> Result<(), SymbolizeError> {
            // Here we're looking at the case where there is
            // only one symbolisation, which is just
            //
            //  0 --> 1 --> 2 --> 3
            //     A     B     C
            let mut f = FwdBacking::allocate(4);
            let mut ff: FwdState = f.state();
            ff.set_input_length(3)?;

            ff.maybe_add_path(SymbolizeNode(0), SymbolizeNode(1), SymbolId(0), 1)?;
            ff.maybe_add_path(SymbolizeNode(1), SymbolizeNode(2), SymbolId(1), 2)?;
            ff.maybe_add_path(SymbolizeNode(2), SymbolizeNode(3), SymbolId(2), 3)?;

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0), SymbolId(1), SymbolId(2)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        struct SimplePathSource {
            path_components: BTreeMap<usize, Vec<PathComponent>>,
            final_state: usize,
        }

        impl SimplePathSource {
            pub fn new() -> SimplePathSource {
                SimplePathSource {
                    path_components: BTreeMap::new(),
                    final_state: 0,
                }
            }
            pub fn add_path_component(&mut self, from: usize, to: usize, symbol: usize) {
                self.path_components
                    .entry(to)
                    .or_default()
                    .push(PathComponent {
                        from_state: from,
                        symbol: SymbolId(symbol),
                    });
            }
        }

        impl PathSource for SimplePathSource {
            fn final_state(&self) -> SymbolizeNode {
                SymbolizeNode(self.final_state)
            }

            fn to_node(&self, node: SymbolizeNode) -> &[PathComponent] {
                self.path_components
                    .get(&node.0)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            }
        }

        // Here we're looking at the case where there is
        // only one symbolisation, which is just
        //
        //  0 --> 1
        //     A

        #[test]
        pub fn simple_onestep_case() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 1;
            ff.add_path_component(0, 1, 0);

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        // This case can't be represented by the packed forward part, but it should be handled by
        // the backward part OK.

        // Here we're looking at the case where there are
        //  two symbolisation which are equivalent
        //
        //  0 --> 1
        //     A
        //     B
        #[test]
        pub fn simple_onestep_case_doubled() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 1;
            ff.add_path_component(0, 1, 0);
            ff.add_path_component(0, 1, 1);

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0)]);

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(1)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        // Here we're looking at the case where there are
        //  two symbolisation which are equivalent
        //
        //  0 --> 1 --> 2
        //     A     C
        //     B
        #[test]
        pub fn simple_twostep_case_doubled() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 2;
            ff.add_path_component(0, 1, 0);
            ff.add_path_component(0, 1, 1);
            ff.add_path_component(1, 2, 2);

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0), SymbolId(2)]);

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(1), SymbolId(2)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        // Here we're looking at the case where there are
        //  two symbolisation which are equivalent
        //
        //  0 --> 1 --> 2
        //     A     B
        //           C
        #[test]
        pub fn simple_twostep_case_doubled_2() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 2;
            ff.add_path_component(0, 1, 0);
            ff.add_path_component(1, 2, 1);
            ff.add_path_component(1, 2, 2);

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0), SymbolId(1)]);

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0), SymbolId(2)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        //        C
        //      -----
        //    /       \
        //  0 --> 1 --> 2
        //     A     B
        //
        #[test]
        pub fn two_paths() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 2;
            ff.add_path_component(0, 1, 0);
            ff.add_path_component(1, 2, 1);
            ff.add_path_component(0, 2, 2);

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(0), SymbolId(1)]);

            let ss = bb.next(&ff);
            assert_eq!(ss.unwrap(), &[SymbolId(2)]);

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }

        #[test]
        pub fn no_path() -> Result<(), SymbolizeError> {
            let mut ff = SimplePathSource::new();
            ff.final_state = 2;

            let mut b = BackBacking::allocate(4);
            let mut bb: BackState = b.state();
            bb.reset();

            let ss = bb.next(&ff);
            assert_eq!(ss, None);

            Ok(())
        }
    }
}
