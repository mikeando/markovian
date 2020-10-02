use serde::{Deserialize, Serialize};

use crate::{
    num_basic::Field,
    renderer::Renderer,
    symbol::{SymbolTable, SymbolTableEntryId},
    vecutils::Reversible,
    weighted_sampler::WeightedSampler,
};
use rand::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

#[derive(Debug)]
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
    for (s, w) in words {
        let s = s.as_ref();
        for ww in s.windows(n) {
            *ngrams.entry(ww.to_vec()).or_insert_with(D::zero) += *w;
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

    pub fn sample<R: Rng>(&self, key: &[T], rng: &mut R) -> Option<T> {
        //TODO: Make this error properly, or accept something such that it can't error like &[T;3]?
        assert!(key.len() == self.n - 1);
        let m = self.weights_table.get(&key.to_vec())?;
        m.sample_next_symbol(rng)
    }

    pub fn context_length(&self) -> usize {
        self.n - 1
    }

    pub fn calculate_logp(&self, v: &[T]) -> f32 {
        let mut sum_log_p = 0.0;
        for w in v.windows(self.n) {
            let log_p = self
                .weights_table
                .get(&w[0..self.n - 1].to_vec())
                .map(|ws| ws.logp(&w[self.n - 1]));
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
pub struct KeyCollection<T> {
    pub order: usize,
    pub n_ngrams: usize,
    pub data: Vec<T>,
}

impl<T> KeyCollection<T> {
    fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.order..(i + 1) * self.order]
    }
    fn mut_row(&mut self, i: usize) -> &mut [T] {
        &mut self.data[i * self.order..(i + 1) * self.order]
    }
    fn new(order: usize, n_ngrams: usize, initial_value: T) -> Self
    where
        T: Clone,
    {
        KeyCollection {
            order,
            n_ngrams,
            data: vec![initial_value; order * n_ngrams],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorReprInternal<T, D, ST>
where
    T: Ord + Clone,
    D: Clone,
{
    pub symbol_table: SymbolTable<T>,
    pub prefixes: KeyCollection<ST>,
    pub prefix_counts: Vec<u32>,
    pub symbols: Vec<ST>,
    pub weights: Vec<D>,
    pub ngram_size: usize,
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
    P: PackedSymbolId + Copy,
{
    fn from(repr: GeneratorReprInternal<T, D, P>) -> Self {
        let mut ngrams: BTreeMap<Vec<SymbolTableEntryId>, D> = BTreeMap::new();

        let mut k: usize = 0;
        let mut key: Vec<SymbolTableEntryId> = vec![SymbolTableEntryId(0); repr.ngram_size];
        for i in 0..repr.prefixes.n_ngrams {
            let p = repr.prefixes.row(i);
            for i in 0..repr.ngram_size - 1 {
                key[i] = p[i].unpack();
            }
            let prefix_count = *repr.prefix_counts.get(i).unwrap();
            for _j in 0..prefix_count {
                let s = repr.symbols.get(k).unwrap();
                let w = repr.weights.get(k).unwrap();
                key[repr.ngram_size - 1] = s.unpack();
                ngrams.insert(key.clone(), *w);
                k += 1;
            }
        }
        Generator::from_ngrams(repr.symbol_table, ngrams, repr.ngram_size)
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
    P: PackedSymbolId + Clone,
{
    fn into(self) -> GeneratorReprInternal<T, D, P> {
        let n_prefixes: usize = self.transition_table.weights_table.len();
        let ngram_size: usize = self.transition_table.n;
        let mut prefixes: KeyCollection<P> =
            KeyCollection::new(ngram_size - 1, n_prefixes, P::default());

        let mut weights: Vec<D> = Vec::new();
        let mut symbols: Vec<P> = Vec::new();
        let mut prefix_counts: Vec<u32> = Vec::new();

        for (i, (k, v)) in self.transition_table.weights_table.iter().enumerate() {
            let row = prefixes.mut_row(i);
            for i in 0..row.len() {
                row[i] = P::pack(k[i]);
            }
            prefix_counts.push(v.counts.len() as u32);
            for (s, w) in &v.counts {
                weights.push(*w);
                symbols.push(P::pack(*s));
            }
        }
        GeneratorReprInternal {
            symbol_table: self.symbol_table,
            prefixes,
            prefix_counts,
            symbols,
            weights,
            ngram_size,
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

    pub fn log_prob(&self, word: &[T]) -> f32 {
        let ss_logp = self
            .symbol_table
            .symbolifications(word)
            .into_iter()
            .map(|w| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(&w);
                let lp = self.transition_table.calculate_logp(&w);
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
        prefix: &[T],
    ) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        let symbolified_prefixes = self.symbol_table.symbolifications_prefix(prefix);
        let prefixes_with_log_prob: Vec<_> = symbolified_prefixes
            .iter()
            .map(|prefix| {
                let w: Vec<SymbolTableEntryId> = self.augment_prefix(prefix);
                let logp: f32 = self.transition_table.calculate_logp(&w);

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

    fn build_suffix_sampler(&self, suffix: &[T]) -> WeightedSampler<Vec<SymbolTableEntryId>, f32> {
        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities and select one.
        // Generate using that symbolified prefix
        let symbolified_sufixes = self.symbol_table.symbolifications_suffix(suffix);
        let suffixes_with_log_prob: Vec<_> = symbolified_sufixes
            .iter()
            .map(|suffix| {
                let w: Vec<SymbolTableEntryId> = self.augment_and_reverse_suffix(suffix);
                let logp: f32 = self.rev_transition_table.calculate_logp(&w);
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
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        loop {
            let next: Option<SymbolTableEntryId> = transition_table.sample(self.key(&v), rng);
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
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let end_id = self.end_symbol_id();
        self.continue_prediction(&self.transition_table, end_id, v, rng)
    }

    fn continue_bwd_prediction<R: Rng>(
        &self,
        v: Vec<SymbolTableEntryId>,
        rng: &mut R,
    ) -> Result<Vec<SymbolTableEntryId>, GenerationError> {
        let start_id = self.start_symbol_id();
        self.continue_prediction(&self.rev_transition_table, start_id, v, rng)
    }

    pub fn generate_multi<R: Rng>(
        &self,
        prefix: Option<&[T]>,
        suffix: Option<&[T]>,
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError>
    where
        T: std::fmt::Debug, // TODO: Only used for error message - would be nice to remove
    {
        match (prefix, suffix) {
            (None, None) => self.generate(n, rng, renderer),
            (None, Some(suffix)) => self.generate_with_suffix(suffix, n, rng, renderer),
            (Some(prefix), None) => self.generate_with_prefix(prefix, n, rng, renderer),
            (Some(prefix), Some(suffix)) => {
                self.generate_with_prefix_and_suffix(prefix, suffix, n, rng, renderer)
            }
        }
    }

    pub fn generate<R: Rng>(
        &self,
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        // Generate an initial vector.
        let mut result = Vec::<String>::with_capacity(n);
        for _i in 0..n {
            let v = self.generate_initial_vector();
            let v = self.continue_fwd_prediction(v, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_prefix<R: Rng>(
        &self,
        prefix: &[T],
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the prefix
        // Calculate their probabilities
        let sampler = self.build_prefix_sampler(prefix);

        for _i in 0..n {
            // Choose one of the prefixes
            let chosen_prefix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_fwd_prediction(chosen_prefix, rng)?;
            result.push(renderer.render(self.body(&v)).unwrap())
        }
        Ok(result)
    }

    pub fn generate_with_suffix<R: Rng>(
        &self,
        suffix: &[T],
        n: usize,
        rng: &mut R,
        renderer: &impl Renderer,
    ) -> Result<Vec<String>, GenerationError> {
        let mut result = Vec::<String>::with_capacity(n);

        // Generate all possible symbolifications of the suffix
        // Calculate their probabilities
        // NOTE: This sampler generates the suffix *reversed*
        let sampler = self.build_suffix_sampler(suffix);

        for _i in 0..n {
            // Choose one of the suffixes
            let chosen_suffix = sampler.sample_next_symbol(rng).unwrap();

            // Generate using that symbolified prefix
            let v = self.continue_bwd_prediction(chosen_suffix, rng)?;

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
        let prefix_sampler = self.build_prefix_sampler(prefix);
        let mut fwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_prefix = prefix_sampler.sample_next_symbol(rng).unwrap();
            let prefix_length = chosen_prefix.len();
            let completed_fwd = self
                .continue_fwd_prediction(chosen_prefix, rng)
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
        let suffix_sampler = self.build_suffix_sampler(suffix);
        let mut bwd_completions = Vec::<(usize, Vec<SymbolTableEntryId>)>::with_capacity(n_gen);

        for _i in 0..n_gen {
            let chosen_suffix = suffix_sampler.sample_next_symbol(rng).unwrap();
            let suffix_length = chosen_suffix.len();
            let mut completed_bwd =
                self.continue_bwd_prediction(chosen_suffix, rng)
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
    let mut result = Vec::new();
    let order = n - 1;
    for x in symbol_table.symbolifications(v) {
        let w = weight_for_symbolification(&x);
        let ss: Vec<SymbolTableEntryId> = iter::repeat(start_id)
            .take(order)
            .chain(x.into_iter())
            .chain(iter::repeat(end_id).take(order))
            .collect();
        result.push((ss, w));
    }
    let sum_w: f32 = result.iter().map(|(_, w)| w).sum();
    result.into_iter().map(|(x, w)| (x, w / sum_w)).collect()
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
        let m: String = g.generate(1, &mut rng, &renderer).unwrap()[0].clone();
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
        let m: String = g.generate(1, &mut rng, &renderer).unwrap()[0].clone();
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
            .generate_with_prefix("hel".as_bytes(), 1, &mut rng, &renderer)
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
            .generate_with_prefix("".as_bytes(), 1, &mut rng, &renderer)
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
            .generate_with_suffix("llo".as_bytes(), 1, &mut rng, &renderer)
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
            .generate_with_suffix("".as_bytes(), 1, &mut rng, &renderer)
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
            .generate_with_prefix_and_suffix("h".as_bytes(), "o".as_bytes(), 1, &mut rng, &renderer)
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
            .generate_with_prefix_and_suffix(prefix, suffix, 10, &mut rng, &renderer)
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
    pub fn serialize_deserialize() {
        let gen = larger_generator();
        let s = bincode::serialize(&gen).unwrap();
        let gen2: Generator<u8, f32> = bincode::deserialize(&s).unwrap();

        assert_eq!(gen, gen2);
    }
}
