use serde::{Deserialize, Serialize};

use crate::{
    num_basic::Field,
    renderer::Renderer,
    symbol::{shortest_symbolifications, SymbolTable, SymbolTableEntryId},
    vecutils::Reversible,
    weighted_sampler::WeightedSampler,
};
use rand::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

#[derive(Debug, PartialEq)]
pub enum GenerationError {
    GenericError(String),
}

impl GenerationError {
    pub fn generic_error<T: Into<String>>(v: T) -> GenerationError {
        GenerationError::GenericError(v.into())
    }
}

pub fn create_ngrams<T, W, D>(words: &[(W, D)], n: usize) -> BTreeMap<Vec<T>, D>
where
    D: Field,
    W: AsRef<[T]>,
    T: Clone + Ord,
{
    assert!(n > 0);
    let mut ngrams = BTreeMap::new();
    //TODO: Does the fact that we get n-grams of the form ^^ cause a problem?
    for nn in 1..=n {
        for (s, w) in words {
            let s = s.as_ref();
            for ww in s.windows(nn) {
                *ngrams.entry(ww.to_vec()).or_insert_with(D::zero) += *w;
            }
        }
    }
    ngrams
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransitionTable<T, D>
where
    T: Ord,
{
    n: usize,
    weights_table: BTreeMap<Vec<T>, WeightedSampler<T, D>>,
}

impl<T, D> TransitionTable<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    pub fn new(counts: BTreeMap<Vec<T>, D>, n: usize) -> TransitionTable<T, D> {
        let mut weights_table: BTreeMap<Vec<T>, WeightedSampler<T, D>> = BTreeMap::new();

        for (v, w) in counts.into_iter() {
            weights_table
                .entry(v[0..v.len() - 1].to_vec())
                .or_default()
                .add_symbol_with_weight(v[v.len() - 1].clone(), w);
        }

        TransitionTable { weights_table, n }
    }

    pub fn to_ngrams_and_weights(&self) -> (BTreeMap<Vec<T>, D>, usize) {
        let mut result = BTreeMap::new();
        for (k, ws) in &self.weights_table {
            for (s, w) in &ws.counts {
                let mut v = k.clone();
                v.push(s.clone());
                result.insert(v, *w);
            }
        }
        (result, self.n)
    }

    pub fn sample<R: Rng>(&self, key: &[T], katz_coefficient: Option<D>, rng: &mut R) -> Option<T> {
        match katz_coefficient {
            Some(katz_coefficient) => {
                let mut key = key;
                // Until we get a table with enough weight we shrink our key down.
                loop {
                    let m = self.weights_table.get(&key.to_vec());

                    if let Some(m) = m {
                        if m.total > katz_coefficient {
                            if let Some(v) = m.sample_next_symbol(rng) {
                                return Some(v);
                            }
                        }
                    }

                    if key.is_empty() {
                        return None;
                    }
                    key = &key[1..];
                }
            }
            None => {
                let m = self.weights_table.get(&key.to_vec())?;
                m.sample_next_symbol(rng)
            }
        }
    }

    pub fn context_length(&self) -> usize {
        self.n - 1
    }

    pub fn get_window_logp(&self, w: &[T], katz_coefficient: Option<D>) -> Option<f32> {
        match katz_coefficient {
            None => self
                .weights_table
                .get(&w[0..self.n - 1].to_vec())
                .and_then(|ws| ws.logp(&w[self.n - 1])),
            Some(katz_coefficient) => {
                let mut prefix = &w[0..self.n - 1];
                let last = &w[self.n - 1];
                // Until we get a table with enough weight we shrink our key down.
                loop {
                    let m = self.weights_table.get(&prefix.to_vec());

                    if let Some(m) = m {
                        if m.total > katz_coefficient {
                            return m.logp(last);
                        }
                    }

                    if prefix.is_empty() {
                        return None;
                    }
                    prefix = &prefix[1..];
                }
            }
        }
    }

    pub fn calculate_logp(&self, v: &[T], katz_coefficient: Option<D>) -> f32 {
        let mut sum_log_p = 0.0;
        for w in v.windows(self.n) {
            let log_p = self.get_window_logp(w, katz_coefficient);
            match log_p {
                Some(log_p) => {
                    sum_log_p += log_p;
                }
                None => return -f32::INFINITY,
            }
        }
        sum_log_p
    }
}

impl<T> TransitionTable<T, f32>
where
    T: Ord + Clone,
{
    fn map_probabilities<F>(&self, f: F) -> TransitionTable<T, f32>
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let mut weights_table: BTreeMap<Vec<T>, WeightedSampler<T, f32>> = BTreeMap::new();
        let n = self.n;

        for (k, v) in &self.weights_table {
            weights_table.insert(k.clone(), v.map_probabilities(f));
        }

        TransitionTable { n, weights_table }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedKeyCollection<T, W>
where
    T: PartialEq,
{
    pub key_length: usize,
    pub prefixes: Vec<T>,
    pub prefix_counts: Vec<usize>,
    pub last: Vec<T>,
    pub weights: Vec<W>,
}

impl<T, W> PackedKeyCollection<T, W>
where
    T: PartialEq + Copy,
    W: Copy,
{
    fn last_prefix(&self) -> Option<&[T]> {
        let prefix_length = self.key_length - 1;
        // Note rely on the counts rather than the prefixes
        // since it is possible to have a key-length of 1
        // which has a zero-length prefix,
        // in which case prefixes.len() / (key_length - 1)
        // gives an error.
        let n_prefixes = self.prefix_counts.len();
        if n_prefixes > 0 {
            let last_prefix_start = (n_prefixes - 1) * prefix_length;
            let last_prefix_end = n_prefixes * prefix_length;
            Some(&self.prefixes[last_prefix_start..last_prefix_end])
        } else {
            None
        }
    }

    fn add_entry(&mut self, key: &[T], weight: W) {
        assert_eq!(key.len(), self.key_length);

        let prefix_length = self.key_length - 1;

        let last_prefix = self.last_prefix();
        let key_prefix = &key[0..prefix_length];

        // If we match the last prefix we need to bump the count
        // otherwise we need to register a new prefix
        if last_prefix == Some(&key[0..prefix_length]) {
            *self.prefix_counts.last_mut().unwrap() += 1;
        } else {
            for p in key_prefix {
                self.prefixes.push(*p);
            }
            self.prefix_counts.push(1);
        }

        // Now we need to add the weight and last part of the key
        self.weights.push(weight);
        self.last.push(key[self.key_length - 1]);
    }

    fn new(key_length: usize) -> Self
    where
        T: Clone,
    {
        PackedKeyCollection {
            key_length,
            prefixes: vec![],
            prefix_counts: vec![],
            last: vec![],
            weights: vec![],
        }
    }

    fn unpack(&self) -> Vec<(Vec<T>, W)> {
        let prefix_length = self.key_length - 1;

        let mut result = vec![];
        let mut ikey = 0;
        for prefix_index in 0..self.prefix_counts.len() {
            let prefix_start = prefix_index * prefix_length;
            let prefix_end = (prefix_index + 1) * prefix_length;
            let prefix = &self.prefixes[prefix_start..prefix_end];

            for _i in 0..self.prefix_counts[prefix_index] {
                let key: Vec<T> = prefix
                    .iter()
                    .chain(std::iter::once(&self.last[ikey]))
                    .cloned()
                    .collect();
                let weight = self.weights[ikey];
                result.push((key, weight));
                ikey += 1;
            }
        }

        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorReprInternal<T, D, ST>
where
    ST: PartialEq,
    T: Ord + Clone,
    D: Clone,
{
    pub symbol_table: SymbolTable<T>,
    pub key_collections: BTreeMap<usize, PackedKeyCollection<ST, D>>,
    pub n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorRepr<T, D>
where
    T: Ord + Clone,
    D: Clone,
{
    GeneratorReprU8(GeneratorReprInternal<T, D, u8>),
    GeneratorReprU16(GeneratorReprInternal<T, D, u16>),
    GeneratorReprRaw(GeneratorReprInternal<T, D, SymbolTableEntryId>),
}

#[derive(Debug)]
pub struct WeightRange {
    pub min_weight: f32,
    pub max_weight: f32,
    pub mean_weight: f32,
    pub count: usize,
}

impl WeightRange {
    pub fn update(&mut self, w: f32) {
        if w < self.min_weight {
            self.min_weight = w;
        }
        if w > self.max_weight {
            self.max_weight = w;
        }
        let mut sum = self.mean_weight * (self.count as f32);
        sum += w;
        self.count += 1;
        self.mean_weight = sum / (self.count as f32);
    }

    pub fn new(w: f32) -> WeightRange {
        WeightRange {
            min_weight: w,
            max_weight: w,
            mean_weight: w,
            count: 1,
        }
    }
}

#[derive(Debug)]
pub struct GeneratorInfo {
    pub ngram_weights_by_length: BTreeMap<usize, WeightRange>,
    pub prefix_weights_by_length: BTreeMap<usize, WeightRange>,
}

impl GeneratorInfo {
    pub fn add_ngram_weight(&mut self, key_length: usize, w: f32) {
        self.ngram_weights_by_length
            .entry(key_length)
            .and_modify(|wr| wr.update(w))
            .or_insert_with(|| WeightRange::new(w));
    }
    pub fn add_prefix_weight(&mut self, key_length: usize, w: f32) {
        self.prefix_weights_by_length
            .entry(key_length)
            .and_modify(|wr| wr.update(w))
            .or_insert_with(|| WeightRange::new(w));
    }
    pub fn new() -> GeneratorInfo {
        GeneratorInfo {
            ngram_weights_by_length: BTreeMap::new(),
            prefix_weights_by_length: BTreeMap::new(),
        }
    }
}

impl Default for GeneratorInfo {
    fn default() -> Self {
        Self::new()
    }
}

// TODO: This serializes "badly" - there's a lot of redundency
// it should just be the symbol_table (which tends to be very small)
// then a list of the symbolid triples + weights. The TTs can be rebuilt from
// them quickly and easily.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(from = "GeneratorRepr<T,D>")]
#[serde(into = "GeneratorRepr<T,D>")]
pub struct Generator<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    pub symbol_table: SymbolTable<T>,
    transition_table: TransitionTable<SymbolTableEntryId, D>,
    rev_transition_table: TransitionTable<SymbolTableEntryId, D>,
}

impl<T, D, P> From<GeneratorReprInternal<T, D, P>> for Generator<T, D>
where
    T: Ord + Clone,
    D: Field,
    P: PackedSymbolId + Copy + PartialEq,
{
    fn from(repr: GeneratorReprInternal<T, D, P>) -> Self {
        let mut ngrams: BTreeMap<Vec<SymbolTableEntryId>, D> = BTreeMap::new();
        for (_key_size, packed_keys) in repr.key_collections {
            for (k, w) in packed_keys.unpack() {
                let kk = k.into_iter().map(|k| k.unpack()).collect();
                ngrams.insert(kk, w);
            }
        }
        Generator::from_ngrams(repr.symbol_table, ngrams, repr.n)
    }
}

impl<T, D> From<GeneratorRepr<T, D>> for Generator<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    fn from(repr: GeneratorRepr<T, D>) -> Self {
        match repr {
            GeneratorRepr::GeneratorReprU8(r) => Self::from(r),
            GeneratorRepr::GeneratorReprU16(r) => Self::from(r),
            GeneratorRepr::GeneratorReprRaw(r) => Self::from(r),
        }
    }
}

pub trait PackedSymbolId {
    fn default() -> Self;
    fn pack(v: SymbolTableEntryId) -> Self;
    fn unpack(self) -> SymbolTableEntryId;
}

impl PackedSymbolId for u8 {
    fn default() -> Self {
        0
    }

    fn pack(v: SymbolTableEntryId) -> Self {
        assert!(v.0 <= u8::MAX as u64);
        v.0 as u8
    }

    fn unpack(self) -> SymbolTableEntryId {
        SymbolTableEntryId(self as u64)
    }
}

impl PackedSymbolId for u16 {
    fn default() -> Self {
        0
    }

    fn pack(v: SymbolTableEntryId) -> Self {
        assert!(v.0 <= u16::MAX as u64);
        v.0 as u16
    }

    fn unpack(self) -> SymbolTableEntryId {
        SymbolTableEntryId(self as u64)
    }
}

impl PackedSymbolId for SymbolTableEntryId {
    fn default() -> Self {
        SymbolTableEntryId(0)
    }

    fn pack(v: SymbolTableEntryId) -> Self {
        v
    }

    fn unpack(self) -> SymbolTableEntryId {
        self
    }
}

impl<T, D, P> Into<GeneratorReprInternal<T, D, P>> for Generator<T, D>
where
    T: Ord + Clone,
    D: Field,
    P: PackedSymbolId + Copy + PartialEq,
{
    fn into(self) -> GeneratorReprInternal<T, D, P> {
        let ngram_size: usize = self.transition_table.n;
        let mut key_collections: BTreeMap<usize, PackedKeyCollection<P, D>> = BTreeMap::new();

        for (key, weight) in self.transition_table.to_ngrams_and_weights().0 {
            let key_length = key.len();
            let packed_key: Vec<_> = key.into_iter().map(P::pack).collect();
            key_collections
                .entry(key_length)
                .or_insert_with(|| PackedKeyCollection::new(key_length))
                .add_entry(&packed_key, weight);
        }
        GeneratorReprInternal {
            symbol_table: self.symbol_table,
            key_collections,
            n: ngram_size,
        }
    }
}

impl<T, D> Into<GeneratorRepr<T, D>> for Generator<T, D>
where
    T: Ord + Clone,
    D: Field,
{
    fn into(self) -> GeneratorRepr<T, D> {
        let max_symbol_id = self.symbol_table.max_symbol_id();
        if max_symbol_id <= u8::MAX as usize {
            GeneratorRepr::GeneratorReprU8(self.into())
        } else if max_symbol_id <= u16::MAX as usize {
            GeneratorRepr::GeneratorReprU16(self.into())
        } else {
            GeneratorRepr::GeneratorReprRaw(self.into())
        }
    }
}

impl<T, D> Generator<T, D>
where
    D: Field,
    T: Ord + Clone,
{
    pub fn from_ngrams(
        symbol_table: SymbolTable<T>,
        ngrams: BTreeMap<Vec<SymbolTableEntryId>, D>,
        n: usize,
    ) -> Generator<T, D> {
        let rev_ngrams: BTreeMap<Vec<SymbolTableEntryId>, D> = ngrams
            .iter()
            .map(|(ngram, w)| (ngram.reversed(), *w))
            .collect();

        let transition_table = TransitionTable::new(ngrams, n);
        let rev_transition_table = TransitionTable::new(rev_ngrams, n);
        Generator {
            symbol_table,
            transition_table,
            rev_transition_table,
        }
    }

    pub fn context_length(&self) -> usize {
        self.transition_table.context_length()
    }

    pub fn start_symbol_id(&self) -> SymbolTableEntryId {
        self.symbol_table.start_symbol_id()
    }

    pub fn end_symbol_id(&self) -> SymbolTableEntryId {
        self.symbol_table.end_symbol_id()
    }

    pub fn generate_initial_vector(&self) -> Vec<SymbolTableEntryId> {
        vec![self.start_symbol_id(); self.context_length()]
    }

    pub fn key<'a>(&self, v: &'a [SymbolTableEntryId]) -> &'a [SymbolTableEntryId] {
        &v[v.len() - self.context_length()..v.len()]
    }

    pub fn body<'a>(&self, v: &'a [SymbolTableEntryId]) -> &'a [SymbolTableEntryId] {
        &v[self.context_length()..(v.len() - self.context_length())]
    }

    pub fn augment_prefix(&self, prefix: &[SymbolTableEntryId]) -> Vec<SymbolTableEntryId> {
        iter::repeat(self.start_symbol_id())
            .take(self.context_length())
            .chain(prefix.iter().cloned())
            .collect()
    }

    pub fn log_prob(&self, word: &[T], katz_coefficient: Option<D>) -> f32 {
        let ss_logp = self
            .symbol_table
            .symbolifications(word)
            .into_iter()
            .map(|w| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(&w);
                let lp = self.transition_table.calculate_logp(&w, katz_coefficient);
                (w, lp)
            })
            .filter(|(_w, lp)| *lp > -f32::INFINITY)
            .collect::<Vec<_>>();

        if ss_logp.is_empty() {
            // They're all imposable...
            return -f32::INFINITY;
        }

        let max_log_p = ss_logp
            .iter()
            .map(|(_s, lp)| *lp)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut sump = 0.0;
        for (_ss, logp) in ss_logp {
            let w = (logp - max_log_p).exp();
            sump += w;
        }
        max_log_p + sump.ln()
    }

    pub fn augment_and_reverse_suffix(
        &self,
        suffix: &[SymbolTableEntryId],
    ) -> Vec<SymbolTableEntryId> {
        iter::repeat(self.end_symbol_id())
            .take(self.context_length())
            .chain(suffix.iter().rev().cloned())
            .collect()
    }

    pub fn build_prefix_sampler(
        &self,
        katz_coefficient: Option<D>,
        prefix: &[T],
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        let symbolified_prefixes = self.symbol_table.symbolifications_prefix(prefix);
        let prefixes_with_log_prob: Vec<_> = symbolified_prefixes
            .iter()
            .map(|prefix| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(prefix);
                let logp: f32 = self.transition_table.calculate_logp(&w, katz_coefficient);

                (w, logp)
            })
            .collect();

        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        if prefixes_with_log_prob
            .iter()
            .all(|(_k, logp)| *logp == -f32::INFINITY)
        {
            // They all have zero weight so we just assume all are equally likely
            for (ss, _logp) in prefixes_with_log_prob {
                sampler.add_symbol_with_weight(ss, 1.0);
            }
        } else {
            let mut min_log_p = 0.0;
            for (_, logp) in &prefixes_with_log_prob {
                if (*logp < min_log_p) && (*logp > -f32::INFINITY) {
                    min_log_p = *logp;
                }
            }
            for (ss, logp) in prefixes_with_log_prob {
                let w = (logp - min_log_p).exp();
                sampler.add_symbol_with_weight(ss, w);
            }
        }
        sampler
    }

    fn build_suffix_sampler(
        &self,
        katz_coefficient: Option<D>,
        suffix: &[T],
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities and select one.
        // Generate using that symbolified prefix
        let symbolified_sufixes = self.symbol_table.symbolifications_suffix(suffix);
        let suffixes_with_log_prob: Vec<_> = symbolified_sufixes
            .iter()
            .map(|suffix| {
                let w: Vec<SymbolTableEntryId> = self.augment_and_reverse_suffix(suffix);
                let logp: f32 = self
                    .rev_transition_table
                    .calculate_logp(&w, katz_coefficient);
                (w, logp)
            })
            .collect();

        let mut sampler = WeightedSampler::<Vec<SymbolTableEntryId>, f32>::new();
        if suffixes_with_log_prob
            .iter()
            .all(|(_k, logp)| *logp == -f32::INFINITY)
        {
            // They all have zero weight so we just assume all are equally likely
            for (ss, _logp) in suffixes_with_log_prob {
                sampler.add_symbol_with_weight(ss, 1.0);
            }
        } else {
            let mut min_log_p = 0.0;
            for (_, logp) in &suffixes_with_log_prob {
                if (*logp < min_log_p) && (*logp > -f32::INFINITY) {
                    min_log_p = *logp;
                }
            }
            for (ss, logp) in suffixes_with_log_prob {
                let w = (logp - min_log_p).exp();
                sampler.add_symbol_with_weight(ss, w);
            }
        }
        sampler
    }

    // TODO this probably belongs in TransitionTable
    // TODO as does self.key and self.context_length
    fn continue_prediction<R: Rng>(
        &self,
        transition_table: &TransitionTable<SymbolTableEntryId, D>,
        terminal: SymbolTableEntryId,
        mut v: Vec<SymbolTableEntryId>,
        katz_coefficient: Option<D>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        loop {
            let next: Option<SymbolTableEntryId> =
                transition_table.sample(self.key(&v), katz_coefficient, rng);
            let next = next.ok_or_else(|| {
                GenerationError::generic_error("Unable to find valid continuation")
            })?;
            if next == terminal {
                v.extend(iter::repeat(terminal).take(self.context_length()));
                return Ok(v);
            }
            v.push(next);
        }
    }

    fn continue_fwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        katz_coefficient: Option<D>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let end_id = self.end_symbol_id();
        self.continue_prediction(&self.transition_table, end_id, v, katz_coefficient, rng)
    }

    fn continue_bwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        katz_coefficient: Option<D>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let start_id = self.start_symbol_id();
        self.continue_prediction(
            &self.rev_transition_table,
            start_id,
            v,
            katz_coefficient,
            rng,
        )
    }

    pub fn generate_multi<R: Rng>(
        &self,
        prefix: Option<&[T]>,
        suffix: Option<&[T]>,
        n: usize,
        katz_coefficient: Option<D>,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError>
    where
        T: std::fmt::Debug, // TODO: Only used for error message - would be nice to remove
    {
        match (prefix, suffix) {
            (None, None) => self.generate(n, katz_coefficient, rng, renderer),
            (None, Some(suffix)) => {
                self.generate_with_suffix(suffix, n, katz_coefficient, rng, renderer)
            }
            (Some(prefix), None) => {
                self.generate_with_prefix(prefix, n, katz_coefficient, rng, renderer)
            }
            (Some(prefix), Some(suffix)) => self.generate_with_prefix_and_suffix(
                prefix,
                suffix,
                n,
                katz_coefficient,
                rng,
                renderer,
            ),
        }
    }

    pub fn generate<R: Rng>(
        &self,
        n: usize,
        katz_coefficient: Option<D>,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        // Generate an initial vector.
        let mut result = Vec::<String>::with_capacity(n);
        for _i in 0..n {
            let v = self.generate_initial_vector();
            let v = self.continue_fwd_prediction(v, katz_coefficient, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_prefix<R: Rng>(
        &self,
        prefix: &[T],
        n: usize,
        katz_coefficient: Option<D>,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the prefix
        // Calculate their probabilities
        let sampler = self.build_prefix_sampler(katz_coefficient, prefix);

        for _i in 0..n {
            // Choose one of the prefixes
            let chosen_prefix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_fwd_prediction(chosen_prefix, katz_coefficient, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_suffix<R: Rng>(
        &self,
        suffix: &[T],
        n: usize,
        katz_coefficient: Option<D>,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities
        // NOTE: This sampler generates the suffix *reversed*
        let sampler = self.build_suffix_sampler(katz_coefficient, suffix);

        for _i in 0..n {
            // Choose one of the suffixes
            let chosen_suffix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_bwd_prediction(chosen_suffix, katz_coefficient, rng)?;

            // Need to reverse v before we render it.
            let mut v = self.body(&v).to_vec();
            v.reverse();
            result.push(renderer.render(&v).unwrap())
        }

        Ok(result)
    }

    pub fn generate_with_prefix_and_suffix<R: Rng>(
        &self,
        prefix: &[T],
        suffix: &[T],
        n: usize,
        katz_coefficient: Option<D>,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError>
    where
        T: std::fmt::Debug, // TODO: Only used for error message - would be nice to remove
    {
        // TOOD: Should we add weights to any of the samplers to get a better result?
        let mut result = Vec::<String>::with_capacity(n);

        // TODO: Q. How big does N need to be? Currently it is a completely random guess.
        let n_gen = (30 * n).max(10);
        let splice_length = self.context_length() + 1;

        // We generate N forward from prefix_str
        // Then store up all the "fwd-splice-points" after prefix
        let prefix_sampler = self.build_prefix_sampler(katz_coefficient, prefix);
        let mut fwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_prefix = prefix_sampler.sample_next_symbol(rng).unwrap();
            let prefix_length = chosen_prefix.len();
            let completed_fwd = self
                .continue_fwd_prediction(chosen_prefix, katz_coefficient, rng)
                .map_err(|e| {
                    GenerationError::generic_error(format!(
                        "Unable to generate continuation of prefix '{:?}' - {:?}",
                        prefix, e
                    ))
                })?;
            fwd_completions.push((prefix_length, completed_fwd));
        }

        type StemSampler<'a> = BTreeMap<
            &'a [SymbolTableEntryId],
            WeightedSampler<(usize, &'a [SymbolTableEntryId]), f32>,
        >;

        let mut fwd_part_samplers: StemSampler = BTreeMap::new();
        for (k, v) in &fwd_completions {
            for (i, w) in v[*k..].windows(splice_length).enumerate() {
                fwd_part_samplers
                    .entry(w)
                    .or_default()
                    .add_symbol((k + i, v));
            }
        }

        //TODO: How do we handle duplication etc?

        // We generate N backward from suffix_str
        // Store up all the bwd-splice-points before suffix
        let suffix_sampler = self.build_suffix_sampler(katz_coefficient, suffix);
        let mut bwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_suffix = suffix_sampler.sample_next_symbol(rng).unwrap();
            let suffix_length = chosen_suffix.len();
            let mut completed_bwd = self
                .continue_bwd_prediction(chosen_suffix, katz_coefficient, rng)
                .map_err(|e| {
                    GenerationError::generic_error(format!(
                        "Unable to generate backward continuation of suffix '{:?}' - {:?}",
                        suffix, e
                    ))
                })?;
            completed_bwd.reverse();
            bwd_completions.push((suffix_length, completed_bwd));
        }

        let mut bwd_part_samplers: StemSampler = BTreeMap::new();
        for (k, v) in &bwd_completions {
            for (i, w) in v[..v.len() - *k].windows(splice_length).enumerate() {
                bwd_part_samplers.entry(w).or_default().add_symbol((i, v));
            }
        }

        // The we try to match up fwd and bwd splice points.
        let fwd_splice_point_keys: BTreeSet<&[SymbolTableEntryId]> =
            fwd_part_samplers.keys().cloned().collect();
        let bwd_splice_point_keys: BTreeSet<&[SymbolTableEntryId]> =
            bwd_part_samplers.keys().cloned().collect();

        // These might be one too short.
        // println!("fwd_splice_point_keys={:?}", fwd_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());
        // println!("bwd_splice_point_keys={:?}", bwd_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());

        let common_splice_point_keys: Vec<_> = fwd_splice_point_keys
            .intersection(&bwd_splice_point_keys)
            .collect();

        // println!("common_splice_point_keys={:?}", common_splice_point_keys.iter().map( |v| self.symbol_table.render(&v) ).collect::<Vec<_>>());

        let mut splice_point_sampler: WeightedSampler<&[SymbolTableEntryId], f32> =
            WeightedSampler::new();
        for sp in &common_splice_point_keys {
            splice_point_sampler.add_symbol(*sp)
        }

        for _i in 0..n {
            // Pick a splice point key
            let splice_point = splice_point_sampler.sample_next_symbol(rng).unwrap();
            // println!("picked splice_point={:?}", self.symbol_table.render(splice_point) );

            // Pick a prefix for that key
            let fwd_part_sampler = fwd_part_samplers.get(splice_point).unwrap();
            let prefix = fwd_part_sampler.sample_next_symbol(rng).unwrap();
            // println!("picked prefix u={}, v={}", prefix.0, self.symbol_table.render(prefix.1) );

            // Pick a suffix for that key
            let bwd_part_sampler = bwd_part_samplers.get(splice_point).unwrap();
            let suffix = bwd_part_sampler.sample_next_symbol(rng).unwrap();
            // println!("picked suffix u={}, v={}", suffix.0, self.symbol_table.render(suffix.1) );

            // Join it all together
            // Finally an answer is PREFIX-FWD-SPLICE-BWD-SUFFIX
            // TODO We should allow the FWD and BWD to be empty
            // TODO The SPLICE should be able to be part of the PREFIX or suffix

            let whole = [
                &prefix.1[..prefix.0],
                //splice_point,
                &suffix.1[suffix.0..],
            ]
            .concat();
            // Then we render the answer.
            // println!("whole={}", self.symbol_table.render(&whole) );

            let text = renderer.render(self.body(&whole)).unwrap();
            result.push(text);
        }

        Ok(result)
    }
}

pub fn weight_for_symbolification(v: &[SymbolTableEntryId]) -> f32 {
    1.0 / ((v.len() * v.len()) as f32)
}

impl<T> Generator<T, f32>
where
    T: Ord + Clone,
{
    pub fn map_probabilities<F>(&self, f: F) -> Generator<T, f32>
    where
        F: Fn(f32) -> f32 + Copy,
    {
        let tt: TransitionTable<SymbolTableEntryId, f32> =
            self.transition_table.map_probabilities(f);
        let (ngrams, n) = tt.to_ngrams_and_weights();
        Generator::from_ngrams(self.symbol_table.clone(), ngrams, n)
    }
}

impl<T> Generator<T, f32>
where
    T: Ord + Clone,
{
    pub fn get_info(&self) -> GeneratorInfo {
        let mut info = GeneratorInfo::new();
        for (k, w) in self.transition_table.to_ngrams_and_weights().0.iter() {
            info.add_ngram_weight(k.len(), *w);
        }
        for (prefix, sampler) in self.transition_table.weights_table.iter() {
            info.add_prefix_weight(prefix.len(), sampler.total);
        }
        info
    }
}

pub trait ToSymbolsAndWeights<T> {
    fn to_symbols_and_weights(&self, v: &[T]) -> Vec<(Vec<SymbolTableEntryId>, f32)>;
}

pub struct InverseSquareOfLengthWeigther<'a, T>
where
    T: Ord + Clone,
{
    symbol_table: &'a SymbolTable<T>,
}

impl<'a, T> InverseSquareOfLengthWeigther<'a, T>
where
    T: Ord + Clone,
{
    pub fn new(symbol_table: &'a SymbolTable<T>) -> Self {
        InverseSquareOfLengthWeigther { symbol_table }
    }
}

impl<'a, T> ToSymbolsAndWeights<T> for InverseSquareOfLengthWeigther<'a, T>
where
    T: Ord + Clone,
{
    fn to_symbols_and_weights(&self, v: &[T]) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
        let mut result = Vec::new();
        for x in self.symbol_table.symbolifications(v) {
            let w = weight_for_symbolification(&x);
            result.push((x, w));
        }
        result
    }
}

pub struct ShortestOnlyWeigther<'a, T>
where
    T: Ord + Clone,
{
    symbol_table: &'a SymbolTable<T>,
}

impl<'a, T> ShortestOnlyWeigther<'a, T>
where
    T: Ord + Clone,
{
    pub fn new(symbol_table: &'a SymbolTable<T>) -> Self {
        ShortestOnlyWeigther { symbol_table }
    }
}

impl<'a, T> ToSymbolsAndWeights<T> for ShortestOnlyWeigther<'a, T>
where
    T: Ord + Clone,
{
    fn to_symbols_and_weights(&self, v: &[T]) -> Vec<(Vec<SymbolTableEntryId>, f32)> {
        let ss = shortest_symbolifications(self.symbol_table, v);
        let l = 1.0 / (ss.len() as f32);
        ss.into_iter().map(|s| (s, l)).collect()
    }
}

pub fn add_padding(
    n: usize,
    start_id: SymbolTableEntryId,
    end_id: SymbolTableEntryId,
    v: Vec<SymbolTableEntryId>,
) -> Vec<SymbolTableEntryId> {
    iter::repeat(start_id)
        .take(n)
        .chain(v.into_iter())
        .chain(iter::repeat(end_id).take(n))
        .collect()
}

// TODO: Error if we can't get at least one symbolification
// TODO: Move this into the Symbol table?
// TODO: Provide weight_for_symbolification as argument.
// TOOD: Only use the shortest symbolifications?
pub fn augment_and_symbolify<T>(
    symbol_table: &SymbolTable<T>,
    v: &[T],
    n: usize,
) -> Vec<(Vec<SymbolTableEntryId>, f32)>
where
    T: Ord + Clone,
{
    assert!(n > 1);
    let start_id = symbol_table.start_symbol_id();
    let end_id = symbol_table.end_symbol_id();
    let result = InverseSquareOfLengthWeigther::new(symbol_table).to_symbols_and_weights(v);
    let sum_w: f32 = result.iter().map(|(_, w)| w).sum();
    result
        .into_iter()
        .map(|(x, w)| (add_padding(n - 1, start_id, end_id, x), w / sum_w))
        .collect()
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{renderer::RenderU8, symbol::SymbolTableEntry};

    fn dumb_u8_symbol_table<T: AsRef<str>>(values: &[T]) -> SymbolTable<u8> {
        let mut symbol_table = SymbolTable::new();

        symbol_table.add(SymbolTableEntry::Start);
        symbol_table.add(SymbolTableEntry::End);
        for s in values {
            for c in s.as_ref().bytes() {
                symbol_table.add(SymbolTableEntry::Single(c));
            }
        }

        symbol_table
    }

    fn simple_generator() -> Generator<u8, f32> {
        let values = vec!["hello"];
        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes(), 3))
            .collect();

        let trigrams = create_ngrams(&symbolified_values, 3);
        Generator::from_ngrams(symbol_table, trigrams, 3)
    }

    fn simple_generator_2() -> Generator<u8, f32> {
        let values = vec!["word"];
        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes(), 3))
            .collect();

        let trigrams = create_ngrams(&symbolified_values, 3);
        Generator::from_ngrams(symbol_table, trigrams, 3)
    }

    fn larger_generator() -> Generator<u8, f32> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let values: Vec<String> =
            std::fs::read_to_string(format!("{}/../resources/Moby_Names_M_lc.txt", manifest_dir))
                .unwrap()
                .lines()
                .map(|n| n.trim().to_string())
                .filter(|s| s.len() >= 3)
                .collect();

        let symbol_table = dumb_u8_symbol_table(&values);

        let symbolified_values: Vec<(Vec<SymbolTableEntryId>, f32)> = values
            .iter()
            .flat_map(|s| augment_and_symbolify(&symbol_table, s.as_bytes(), 3))
            .collect();

        let trigrams = create_ngrams(&symbolified_values, 3);
        Generator::from_ngrams(symbol_table, trigrams, 3)
    }

    #[test]
    pub fn test_augment_and_symbolify_hello() {
        let v = "hello";
        let symbol_table = dumb_u8_symbol_table(&[v]);
        let s = augment_and_symbolify(&symbol_table, v.as_bytes(), 3);
        assert_eq!(s.len(), 1);
    }

    #[test]
    pub fn test_symbolify_hello() {
        let v = "hello";
        let symbol_table = dumb_u8_symbol_table(&[v]);
        let ss = symbol_table.symbolifications(v.as_bytes());
        assert_eq!(ss.len(), 1);
    }

    #[test]
    pub fn generate_simple() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g.generate(1, None, &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_simple_2() {
        let mut rng = rand::thread_rng();
        let g = simple_generator_2();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g.generate(1, None, &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(m, "word");
    }

    #[test]
    pub fn generate_prefix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix("hel".as_bytes(), 1, None, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_prefix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix("".as_bytes(), 1, None, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_suffix("llo".as_bytes(), 1, None, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_suffix_empty() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_suffix("".as_bytes(), 1, None, &mut rng, &renderer)
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix() {
        let mut rng = rand::thread_rng();
        let g = simple_generator();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m: String = g
            .generate_with_prefix_and_suffix(
                "h".as_bytes(),
                "o".as_bytes(),
                1,
                None,
                &mut rng,
                &renderer,
            )
            .unwrap()[0]
            .clone();
        assert_eq!(m, "hello");
    }

    #[test]
    pub fn generate_with_prefix_and_suffix_big() {
        let mut rng = rand::thread_rng();
        let g = larger_generator();
        let prefix_str = "h";
        let suffix_str = "y";
        let prefix = prefix_str.as_bytes();
        let suffix = suffix_str.as_bytes();
        let renderer = RenderU8 {
            table: &g.symbol_table,
            start: b"^",
            end: b"$",
        };
        let m = g
            .generate_with_prefix_and_suffix(prefix, suffix, 10, None, &mut rng, &renderer)
            .unwrap();
        for v in m {
            assert!(
                v.starts_with(prefix_str) && v.ends_with(suffix_str),
                "Expected {}..{} but got {}",
                prefix_str,
                suffix_str,
                v,
            );
        }
    }

    #[test]
    pub fn generate_katz_fallback() {
        // A very simple generator that will only work using Katz fallback
        // ^^ -> A
        // ^A -> A
        // no AA -> ?
        // A -> B
        // AB -> $

        let mut symbol_table: SymbolTable<u8> = SymbolTable::new();
        let start = symbol_table.add(SymbolTableEntry::Start);
        let end = symbol_table.add(SymbolTableEntry::End);
        let a = symbol_table.add(SymbolTableEntry::Single(b'a'));
        let b = symbol_table.add(SymbolTableEntry::Single(b'b'));

        let mut ngrams: BTreeMap<Vec<SymbolTableEntryId>, f32> = BTreeMap::new();
        ngrams.insert(vec![start, start, a], 1.0);
        ngrams.insert(vec![start, a, a], 1.0);
        ngrams.insert(vec![a, b], 1.0);
        ngrams.insert(vec![a, b, end], 1.0);
        let gen = Generator::from_ngrams(symbol_table, ngrams, 3);

        let mut rng = rand::thread_rng();
        let renderer = RenderU8 {
            table: &gen.symbol_table,
            start: b"^",
            end: b"$",
        };

        let v = gen.generate(1, None, &mut rng, &renderer);
        assert_eq!(
            v,
            Err(GenerationError::GenericError(
                "Unable to find valid continuation".into()
            ))
        );

        let v = gen.generate(1, Some(0.0), &mut rng, &renderer);
        assert_eq!(v.unwrap()[0], "aab");
    }

    #[test]
    pub fn generate_katz_fallback_2() {
        // A very simple generator that will only work using Katz fallback
        // ^^ -> A
        // ^A -> A
        // reject AA -> C as weight is too low.
        // A -> B
        // AB -> $

        let mut symbol_table: SymbolTable<u8> = SymbolTable::new();
        let start = symbol_table.add(SymbolTableEntry::Start);
        let end = symbol_table.add(SymbolTableEntry::End);
        let a = symbol_table.add(SymbolTableEntry::Single(b'a'));
        let b = symbol_table.add(SymbolTableEntry::Single(b'b'));
        let c = symbol_table.add(SymbolTableEntry::Single(b'c'));

        let mut ngrams: BTreeMap<Vec<SymbolTableEntryId>, f32> = BTreeMap::new();
        ngrams.insert(vec![start, start, a], 1.0);
        ngrams.insert(vec![start, a, a], 1.0);
        ngrams.insert(vec![a, a, c], 0.1);
        ngrams.insert(vec![a, b], 1.0);
        ngrams.insert(vec![a, b, end], 1.0);
        ngrams.insert(vec![a, c, end], 1.0);

        let gen = Generator::from_ngrams(symbol_table, ngrams, 3);

        let mut rng = rand::thread_rng();
        let renderer = RenderU8 {
            table: &gen.symbol_table,
            start: b"^",
            end: b"$",
        };

        // Without Katz we dont reject AAC
        let v = gen.generate(1, None, &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(v, "aac");

        // With low Katz coefficient we dont reject AAC
        let v = gen.generate(1, Some(0.05), &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(v, "aac");

        // With high Katz coefficient we do reject AAC
        let v = gen.generate(1, Some(0.5), &mut rng, &renderer).unwrap()[0].clone();
        assert_eq!(v, "aab");
    }

    #[test]
    pub fn serialize_deserialize() {
        let gen = larger_generator();
        let s = bincode::serialize(&gen).unwrap();
        let gen2: Generator<u8, f32> = bincode::deserialize(&s).unwrap();

        assert_eq!(gen, gen2);
    }

    #[test]
    pub fn serialize_deserialize_with_katz() {
        let mut symbol_table: SymbolTable<u8> = SymbolTable::new();

        symbol_table.add(SymbolTableEntry::Start);
        symbol_table.add(SymbolTableEntry::End);
        let a = symbol_table.add(SymbolTableEntry::Single(b'a'));
        let b = symbol_table.add(SymbolTableEntry::Single(b'b'));

        let mut ngrams: BTreeMap<Vec<SymbolTableEntryId>, f32> = BTreeMap::new();
        ngrams.insert(vec![a, a, a], 1.0);
        ngrams.insert(vec![a, b], 1.0);
        let gen = Generator::from_ngrams(symbol_table, ngrams, 3);

        let s = bincode::serialize(&gen).unwrap();
        let gen2: Generator<u8, f32> = bincode::deserialize(&s).unwrap();

        assert_eq!(gen, gen2);
    }
}
