use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::language;

mod language_manipulation {
    use super::*;
    use crate::language::raw::*;
    use crate::tombstone_map::TombstoneMap;
    use crate::tombstone_map::Token;

    #[derive(Copy,Clone,PartialEq,Eq, Debug, PartialOrd, Ord)]
    pub struct RuleId(pub Token);

    #[derive(Clone, Debug)]
    pub struct ModifiableLanguageProduction<T> {
        pub id:RuleId,
        pub p: Production<T>,
    }

    #[derive(Clone, Debug)]
    pub struct ModifiableLanguage<T> {
        //TODO: Longer term we need to split out the weight and symbol from these.
        pub entries:TombstoneMap<Production<T>>,
        //pub entries:Vec<ModifiableLanguageProduction<T>>,
        //pub last_id:usize,

        pub substring_map: HashMap<Vec<SymbolOrLiteral<T>>, usize>,
        pub all_symbols: HashSet<Symbol>,
    }

    pub fn add_symbols_from_production<T>(all_symbols:&mut HashSet<Symbol>, p:&Production<T>) {
        all_symbols.insert(p.from.clone());
        for s in &p.to {
            if let Some(s) = s.as_symbol() {
                all_symbols.insert(s.clone());
            }
        }
    }

    impl <T> ModifiableLanguage<T> {

        pub fn rebuild_all_symbols(&mut self) {
            self.all_symbols.clear();
            for p in self.entries.iter() {
                add_symbols_from_production(&mut self.all_symbols, p);
            }
        }

        pub fn new_symbol(&self) -> language::raw::Symbol {
            let mut c: usize = 0;
            loop {
                let s = language::raw::Symbol(format!("s_{}", c));
                if !self.all_symbols.contains(&s) {
                    return s;
                }
                c += 1;
            }
        }
    }

    impl <T> From<&ModifiableLanguage<T>> for Language<T> 
        where T: Clone
    {
        fn from(ml:&ModifiableLanguage<T>) -> Language<T> {
            Language {
                entries: ml.entries.iter().map(|p| p.clone()).collect()
            }
        }
    }

    impl <T> From<&Language<T>> for ModifiableLanguage<T> 
        where T: Clone + Hash + Eq
    {
        fn from(l:&Language<T>) -> ModifiableLanguage<T> {
            //Do we need a bulk constructor?
            let mut entries: TombstoneMap<Production<T>> = TombstoneMap::new();
            let mut substring_map: HashMap<Vec<SymbolOrLiteral<T>>, usize> = HashMap::new();

            for e in &l.entries {
                entries.insert_or_get_token(e.clone());
                substring_count_into(&e.to, &mut substring_map);
            }

            let mut v = ModifiableLanguage {
                entries,
                substring_map,
                all_symbols: HashSet::new(),
            };

            v.rebuild_all_symbols();
            v
        }
    }

    #[derive(Debug)]
    pub enum LanguageChangeEntry<T> {
        Add(Production<T>),
        Remove(RuleId, usize),
        ChangeProduction(RuleId, usize, Production<T>),
        Reweight(RuleId, f32),
    }

    impl <T> LanguageChangeEntry<T> 
    {
        pub fn cost(&self) -> f32 {
            match self {
                LanguageChangeEntry::Add(p) => p.to.len() as f32,
                LanguageChangeEntry::Remove(_id, len) => -(*len as f32),
                LanguageChangeEntry::ChangeProduction(_id, len, new) => new.to.len() as f32 - (*len as f32),
                LanguageChangeEntry::Reweight(_,_) => 0.0,
            }
        }
    }

    impl <T> LanguageChangeEntry<T> 
        where T: Hash + Eq + Clone
    {
        // TODO: A lot of time is spent in here - because it 
        //       ends up doing a linear scan. Thats slow and dumb!
        pub fn apply(self, l:&mut ModifiableLanguage<T>) {
            match self {
                LanguageChangeEntry::Add(production) => {
                    substring_count_into(&production.to, &mut l.substring_map);
                    add_symbols_from_production(&mut l.all_symbols, &production);
                    l.entries.insert_or_get_token(production);
                }
                LanguageChangeEntry::Remove(id, _len) => {
                    let production = l.entries.get_by_token(&id.0).unwrap();
                    substring_uncount_into(&production.to, &mut l.substring_map);
                    l.entries.remove_by_token(&id.0).unwrap();
                }
                LanguageChangeEntry::ChangeProduction(id, _len, p) => {
                    // There's probably a better way to do this, but it works for now
                    // Something to watch out for is that this changes the token, 
                    // which we dont want to do
                    let old_production = l.entries.get_by_token(&id.0).unwrap();
                    substring_uncount_into(&old_production.to, &mut l.substring_map);
                    substring_count_into(&p.to, &mut l.substring_map);
                    add_symbols_from_production(&mut l.all_symbols, &p);
                    l.entries.remove_by_token(&id.0).unwrap();
                    l.entries.insert_or_get_token(p);
                }
                LanguageChangeEntry::Reweight(id,w) => {
                    // There's probably a better way to do this, but it works for now
                    // Something to watch out for is that this changes the token, 
                    // which we dont want to do
                    let mut pc: Production<T> = l.entries.remove_by_token(&id.0).unwrap();
                    pc.weight = nf32(w);
                    l.entries.insert_or_get_token(pc);
                }
            }
        }
    }

    #[derive(Debug)]
    pub struct LanguageDelta<T> {
        pub changes:Vec<LanguageChangeEntry<T>>
    }

    impl <T> LanguageDelta<T>  {
        pub fn cost(&self) -> f32 {
            self.changes.iter().map(|s| s.cost()).sum()
        }
    }


    impl <T> LanguageDelta<T>
        where T:Clone + Hash + Eq
    {
        pub fn apply(mut self, l:&mut ModifiableLanguage<T>) {
            // Apply removals first
            for x in self.changes.drain_filter(|x| if let LanguageChangeEntry::Remove(_,_) = x { true } else {false}) {
                x.apply(l)
            }
            for x in self.changes {
                if let LanguageChangeEntry::Remove(_,_) = x {
                } else {
                    x.apply(l);
                }
            }
        }
    }

    #[derive(Debug)]
    pub struct ExtractSequence<T> {
        sequence: Vec<SymbolOrLiteral<T>>,
    }

    #[derive(Debug)]
    pub struct FactorPrefix<T> {
        pub symbol: Symbol,
        pub prefix: Vec<SymbolOrLiteral<T>>,
    }

    #[derive(Debug)]
    pub struct FactorSuffix<T> {
        pub symbol: Symbol,
        pub suffix: Vec<SymbolOrLiteral<T>>,
    }

    pub trait Proposer<T> {
        fn get_proposal(&self, l: &ModifiableLanguage<T>) -> Option<LanguageDelta<T>>;
    }

    pub struct DeltaBuilder<T> {
        pub changes: Vec<LanguageChangeEntry<T>>,
    }

    impl <T> DeltaBuilder<T>
        where T: Clone
    {
        pub fn new() -> DeltaBuilder<T>  {
            DeltaBuilder {
                changes: vec![],
            }
        }

        pub fn remove(&mut self, rule_id: RuleId, production_len:usize) {
            self.changes.push(
                LanguageChangeEntry::Remove(rule_id, production_len)
            );
        }

        pub fn add(&mut self, p:Production<T>) {
            self.changes.push(
                LanguageChangeEntry::Add(p)
            );
        }

        pub fn change_production(&mut self, rule_id:RuleId, original_len:usize, new_production:Production<T>) {
            self.changes.push(
                LanguageChangeEntry::ChangeProduction(
                        rule_id,
                        original_len,
                        new_production,
                )
            );
        }

        pub fn reweight(&mut self, id:RuleId, weight:f32) {
            self.changes.push(
                LanguageChangeEntry::Reweight(id,weight)
            );
        }

        pub fn build(self) -> LanguageDelta<T> {
            LanguageDelta {
                changes:self.changes,
            }
        }
    }


    impl<T> ExtractSequence<T>
    where
        T: Clone + PartialEq,
    {
        pub fn apply(&self, l: &ModifiableLanguage<T>) -> LanguageDelta<T> {
            let s = l.new_symbol();
            let mut builder: DeltaBuilder<T> = DeltaBuilder::new();
            
            //First add the new rule
            builder.add(
                Production {
                    from: s.clone(),
                    to: self.sequence.clone(),
                    weight: nf32(1.0),
                }
            );

            // Now we need to find and replace occurrences of this in
            // all the productions
            for (r_id, r) in l.entries.iter_with_token() {
                if has_subsequence(&r.to, &self.sequence) {
                    builder.change_production(
                        RuleId(r_id),
                        r.to.len(),
                        Production {
                            from: r.from.clone(),
                            weight: r.weight,
                            to: replace_subsequence(
                                &r.to,
                                &self.sequence,
                                SymbolOrLiteral::Symbol(s.clone()),
                            ),
                        }
                    );
                }
            }

            builder.build()
        }
    }


    impl<T> FactorPrefix<T>
    where
        T: Clone + PartialEq,
    {
        pub fn apply(&self, l: &ModifiableLanguage<T>) -> LanguageDelta<T> {
            let s = l.new_symbol();
            let mut builder: DeltaBuilder<T> = DeltaBuilder::new();
            let mut weight: nf32 = nf32(0.0);

            for (r_id, r) in l.entries.iter_with_token() {
                if (r.from != self.symbol) || (!r.to.starts_with(&self.prefix)) {
                    continue;
                }

                //TODO: Would be better to use a replace here...
                builder.remove(RuleId(r_id), r.to.len());
                builder.add(
                    Production {
                        from: s.clone(),
                        weight: r.weight,
                        to: r.to[self.prefix.len()..].to_vec(),
                    }
                );
                weight += r.weight;
            }
            
            builder.add(
                Production {
                    from: self.symbol.clone(),
                    to: [self.prefix.clone(), vec![SymbolOrLiteral::Symbol(s)]].concat(),
                    weight,
                },
            );

            builder.build()
        }

    }

    impl<T> FactorSuffix<T>
    where
        T: Clone + PartialEq,
    {
        pub fn apply(&self, l: &ModifiableLanguage<T>) -> LanguageDelta<T> {
            let s = l.new_symbol();
            let mut builder: DeltaBuilder<T> = DeltaBuilder::new();
            let mut weight: nf32 = nf32(0.0);

            for (r_id, r) in l.entries.iter_with_token() {
                if (r.from != self.symbol) || (!r.to.ends_with(&self.suffix)) {
                    continue;
                }

                builder.remove(RuleId(r_id), r.to.len());
                builder.add(
                    Production {
                        from: s.clone(),
                        weight: r.weight,
                        to: r.to[..(r.to.len() - self.suffix.len())].to_vec(),
                    }
                );
                weight += r.weight;
            }
           
            builder.add(
                Production {
                    from: self.symbol.clone(),
                    to: [vec![SymbolOrLiteral::Symbol(s)], self.suffix.clone()].concat(),
                    weight,
                }
            );

            builder.build()
        }

    }

    pub struct ExtractSequenceProposer;

    impl<T> Proposer<T> for ExtractSequenceProposer
    where
        T: Hash + Eq + Clone,
    {
        fn get_proposal(&self, l: &ModifiableLanguage<T>) -> Option<LanguageDelta<T>> {
            l.substring_map.iter()
                .filter(|(_k,v)| **v > 0)
                .map(|(k, v)| (k, (v - 1) * (k.len() - 1)))
                .max_by_key(|e| e.1)
                .filter(|v| v.1 > 0)
                .map(|(a, _b)| ExtractSequence {
                        sequence: a.to_vec(),
                    })
                .map(|s| s.apply(l))
        }
    }

    pub struct FactorPrefixProposer;


    impl FactorPrefixProposer {

        pub fn production_iter_to_prefix_map<'a, T,Z>(z:Z) -> BTreeMap<(&'a Symbol, &'a [SymbolOrLiteral<T>]), usize> 
        where Z: 'a + Iterator<Item=&'a Production<T>>,
              T: Clone + Ord
        {
            let mut m: BTreeMap<(&'a Symbol, &'a [SymbolOrLiteral<T>]), usize> = BTreeMap::new();
            for e in z {
                for i in 0..e.to.len() {
                    let key =(&e.from, &e.to[..=i]); 
                    *m.entry(key).or_insert(0) += 1;
                }
            }
            m
        }

        pub fn get_proposal_from_prefix_map<T>(m: &BTreeMap<(&Symbol, &[SymbolOrLiteral<T>]), usize>) -> Option<FactorPrefix<T>> 
            where T: Ord + Clone
        {
            // Replacing rules i=1..N
            //   A -> w_i : [P] [S_i]
            //  total cost = N * len(P) + sum( len(S_i) )
            //with
            //  A -> sum(w_i) : [P] [S]
            //  S -> w_i : [S_i]
            // total cost = len(P) + 1 + sum( len(S_i) )
            //
            // so total saving is (N-1) * len(P) - 1
            fn score<T>(suffix: &[SymbolOrLiteral<T>], count: usize) -> i64 {
                (suffix.len() as i64) * (count as i64 - 1) - 1
            }

            m.iter()
                .map(|(k, v)| (k, score(k.1, *v)))
                .filter(|(_k, v)| *v > 0)
                .max_by_key(|e| e.1)
                .map(|((symbol, prefix), _s)| FactorPrefix {
                        symbol: (*symbol).clone(),
                        prefix: prefix.to_vec(),
                    })
        }
    }

    impl<T> Proposer<T> for FactorPrefixProposer
    where
        T: Ord + Clone + std::fmt::Debug,
    {
        fn get_proposal(&self, l: &ModifiableLanguage<T>) -> Option<LanguageDelta<T>> {
            let m = Self::production_iter_to_prefix_map(l.entries.iter());
            Self::get_proposal_from_prefix_map(&m).map(|p| p.apply(&l))
        }

    }

    pub struct FactorSuffixProposer;

    impl FactorSuffixProposer {

        pub fn production_iter_to_suffix_map<'a, T,Z>(z:Z) -> BTreeMap<(&'a Symbol, &'a [SymbolOrLiteral<T>]), usize> 
        where Z: 'a + Iterator<Item=&'a Production<T>>,
              T: Clone + Ord
        {
            let mut m: BTreeMap<(&'a Symbol, &'a [SymbolOrLiteral<T>]), usize> = BTreeMap::new();
            for e in z {
                for i in 0..e.to.len() {
                    let key = (&e.from, &e.to[i..]);
                    *m.entry(key).or_insert(0) += 1;
                }
            }
            m
        }

        pub fn get_proposal_from_suffix_map<T>(m:&BTreeMap<(&Symbol, &[SymbolOrLiteral<T>]),usize>) -> Option<FactorSuffix<T>>
            where T: Ord + Clone
        {
            // Replacing rules i=1..N
            //   A -> w_i : [P] [S_i]
            //  total cost = N * len(P) + sum( len(S_i) )
            //with
            //  A -> sum(w_i) : [P] [S]
            //  S -> w_i : [S_i]
            // total cost = len(P) + 1 + sum( len(S_i) )
            //
            // so total saving is (N-1) * len(P) - 1
            fn score<T>(suffix: &[SymbolOrLiteral<T>], count: usize) -> i64 {
                (suffix.len() as i64) * (count as i64 - 1) - 1
            }

            m.iter()
                .map(|(k, v)| (k, score(k.1, *v)))
                .filter(|(_k, v)| *v > 0)
                .max_by_key(|e| e.1)
                .map(|((symbol, suffix), _s)| FactorSuffix {
                        symbol: (*symbol).clone(),
                        suffix: suffix.to_vec(),
                    })
        }
    }

    impl<T> Proposer<T> for FactorSuffixProposer
    where
        T: Ord + Clone,
    {
        fn get_proposal(&self, l: &ModifiableLanguage<T>) -> Option<LanguageDelta<T>> {
            let m = Self::production_iter_to_suffix_map(l.entries.iter());
            Self::get_proposal_from_suffix_map(&m).map(|p| p.apply(l))
        }

    }

    pub struct PairExtractionProposer;

    #[derive(Copy,Clone,PartialOrd,Ord,PartialEq,Eq, Debug)]
    pub struct ProductionIndex(usize);


    use language::raw::SymbolOrLiteral;

    mod bimap {
        use super::*;

        pub struct ProductionIndexBiMap<'a, T> {
            production_to_idx: BTreeMap<&'a [SymbolOrLiteral<T>], ProductionIndex>,
            idx_to_production: Vec<&'a [SymbolOrLiteral<T>]>,
        }

        impl <'a, T> ProductionIndexBiMap<'a, T> 
            where T: Ord
        {
            pub fn new() -> ProductionIndexBiMap<'a, T> {
                ProductionIndexBiMap {
                    production_to_idx: BTreeMap::new(),
                    idx_to_production: vec![],
                }
            }

            //TODO: Handle duplicates in the input data?
            pub fn from_vec(v:Vec<&'a [SymbolOrLiteral<T>]>) -> ProductionIndexBiMap<'a, T>
            {
                let mut m = ProductionIndexBiMap::new();
                m.idx_to_production = v;
                m.production_to_idx = m.idx_to_production.iter().enumerate().map(|(a,b)| (*b,ProductionIndex(a))).collect();
                m
            }

            pub fn to_idx(&self, k:&'a [SymbolOrLiteral<T>]) -> Option<ProductionIndex> {
                self.production_to_idx.get(k).cloned()
            }
            pub fn to_production(&self, k:ProductionIndex) -> Option<&'a [SymbolOrLiteral<T>]> {
                self.idx_to_production.get(k.0).cloned()
            }
        }
    }

    use bimap::ProductionIndexBiMap;

    fn count<I,T>(m:I) -> BTreeMap<T,usize> 
        where T:Ord,
              I:Iterator<Item=T>,
    {
        let mut result: BTreeMap<T,usize> = BTreeMap::new();
        for v in m {
            *result.entry(v).or_insert(0) += 1;
        }
        result
    }

    mod two_rule_helper {
        use super::*;
        use language::raw::Symbol;
        use language_manipulation::RuleId;

        pub struct TwoRuleRemovalHelperIndex<'a, T> {
            pub symbol_to_production_ids:BTreeMap<Symbol, BTreeMap<ProductionIndex, (RuleId, nf32)>>,
            pub m:ProductionIndexBiMap<'a, T>,
        }

        pub struct Donkey<'a, 'b, T> {
            ss: &'a BTreeMap<ProductionIndex, (RuleId, nf32)>,
            helper: &'b TwoRuleRemovalHelperIndex<'a, T>
        }

        impl <'a, T> TwoRuleRemovalHelperIndex<'a, T> 
            where T: Ord
        {
            pub fn get_donkey(&self, s:&Symbol) -> Donkey<T> {
                let ss: &BTreeMap<ProductionIndex, (RuleId, nf32)> = self.symbol_to_production_ids.get(s).unwrap();
                Donkey {
                    ss,
                    helper: self
                }
            }
        }

        impl <'a, 'b, T> Donkey<'a, 'b, T>
            where T: Ord
        {
            pub fn get_rule(&self, prod_id:ProductionIndex) -> RuleWithMeta<'a, T> {
                let id_w = self.ss.get(&prod_id).unwrap();
                let p =  self.helper.m.to_production(prod_id).unwrap();
                let id = id_w.0;
                let w = id_w.1;
                RuleWithMeta { id, w, p, }
            }
        }


        pub struct RuleWithMeta<'a, T> {
            pub id:RuleId,
            pub w:nf32,
            pub p:&'a [SymbolOrLiteral<T>],
        }

        pub struct RulePair<'a, T> {
            pub a:RuleWithMeta<'a, T>,
            pub b:RuleWithMeta<'a, T>,
        }

        pub fn get_weights<T>(x:&RulePair<T>, y:&RulePair<T>) -> (f32, f32, f32) {
            let w_ab = f32::min(x.a.w.0/y.a.w.0, x.b.w.0/y.b.w.0) * (y.a.w.0 + y.b.w.0);
            let w_a = x.a.w.0 - w_ab * y.a.w.0 / (y.a.w.0 + y.b.w.0);
            let w_b = x.b.w.0 - w_ab * y.b.w.0 / (y.a.w.0 + y.b.w.0);
            (w_ab, w_a, w_b)
        }
    }

    impl<T> Proposer<T> for PairExtractionProposer
    where
        T: Ord + Clone,
    {
        fn get_proposal(&self, l: &ModifiableLanguage<T>) -> Option<LanguageDelta<T>> {

            // Get all production RHS, with their counts.

            let production_counts:BTreeMap<&[SymbolOrLiteral<T>], usize> = 
                count( l.entries.iter().map(|r| &r.to[..]));

            let m = ProductionIndexBiMap::from_vec(
                l.entries.iter().map(|r| &r.to[..]).collect()
            );


            // Now for each symbol generate the set of indices that occur.
            let mut symbol_to_production_ids: BTreeMap<Symbol, BTreeMap<ProductionIndex, (RuleId, nf32)>> = BTreeMap::new();

            // We're only interested in those with counts >= 2
            // since they're the only ones we can factor out.
            // For those cases we build a two level map.
            // Symbol -> ProductionId -> (RuleId, weight)
            for (r_id, r) in l.entries.iter_with_token() {
                if *production_counts.get(&r.to[..]).unwrap() > 1 {
                    symbol_to_production_ids
                        .entry(r.from.clone())
                        .or_default()
                        .insert(
                            m.to_idx(&r.to[..]).unwrap(),
                            (RuleId(r_id),r.weight)
                        );
                }
            }

            // Get the pairs that occur within a single source symbols
            // We get a map 
            // ProductionId x ProductionId -> {Symbol}
            let mut prod_pairs_to_symbols: BTreeMap<(ProductionIndex,ProductionIndex), BTreeSet<Symbol>> = BTreeMap::new();

            for (k,v) in &symbol_to_production_ids {
                for a in v {
                    for b in v {
                        if a < b {
                            prod_pairs_to_symbols.entry((*a.0,*b.0)).or_default().insert(k.clone());
                        }
                    }
                }
            }

            // Now we're looking for the best pair.
            // 
            // The conversion we do is
            //
            // Add new symbol, X, with rules
            //
            // + X (w1) -> P1
            // + X (w2) -> P2
            //
            //      change in cost: + |P1| + |P2|
            //
            // (but we've not decided on what w1 and w2 should be yet...)
            //
            // Then for each symbol S_i that has this pair we're going to go from
            //
            // - S (a1) -> P1
            // - S (a2) -> P2
            //
            //      change in cost: - (|P1| + |P2|)
            // 
            // to one of:
            //
            // CASE I: eliminate both (a1/a2 = w1/w2)
            //
            // S (a1+a2) -> X
            //
            //      change in cost: +1
            //      net change in cost: +1 - |P1| - |P2|
            //
            // or
            //
            // CASE II: eliminate P2 (a1/a2  > w1/w2)
            //
            // S (a1') -> P1
            // S (a1 + a2 - a1') -> X
            //
            //    a1' = a1  - a2 * (w1 / w2)  (>0) 
            //
            //      change in cost: + |P1| + 1
            //      net change in cost: +1 - |P2|
            //
            // or
            //
            // CASE III: eliminate P1 (a1/a3 < w1/w2)
            //
            // S(a2') -> P2
            // S(a1 + a2 - a2') -> X
            //
            //    a2' = a2 - a1 (w2 / w1) .(>0)
            //
            //      change in cost: + |P2| + 1
            //      net change in cost: +1 - |P1|
            //
            // So for each of the symbols we pick w1 and w2 to wipe out the 
            // that rule entirely via substitution CASE I. 
            // Then work out what is left for each of the symbols.
            // Then calculate how it effects the total score. 
            // Then we pick the ratio that had the best effect on the total score!

            let ww = two_rule_helper::TwoRuleRemovalHelperIndex {
                symbol_to_production_ids,
                m,
            };
            
            let mut best: Option<( ProductionIndex, ProductionIndex, BTreeSet<language::raw::Symbol>, language::raw::Symbol, i32 )> = None;
            for (prod_ids,symbols) in prod_pairs_to_symbols {


                let symbols_with_extras : Vec<(&language::raw::Symbol, language_manipulation::two_rule_helper::RulePair<'_, T>)> = symbols.iter()
                    .map( |s| {
                        let ddd = ww.get_donkey(s);
                        let a = ddd.get_rule(prod_ids.0);
                        let b = ddd.get_rule(prod_ids.1);
                        let pair_ab = two_rule_helper::RulePair{a, b};
                        (s, pair_ab)
                    })
                    .collect();


                for (s, pair_ab_1) in &symbols_with_extras {
                    let mut score = 0i32;
                    let la =  pair_ab_1.a.p.len() as i32;
                    let lb =  pair_ab_1.b.p.len() as i32;

                    for (_ss, pair_ab_2) in &symbols_with_extras {
                        let (_w_ab, w_a, w_b) = two_rule_helper::get_weights(&pair_ab_2, &pair_ab_1);

                        if w_a == 0.0 {
                            score -= la;
                        }
                        if w_b == 0.0 {
                            score -= lb;
                        }
                        score += 1;

                    }
                    // Generate the hypothetical delta
                    // Evaluate it.
                    if best.as_ref().map(|(_,_,_,_,best_score)| *best_score > score).unwrap_or(true) {
                        best = Some((prod_ids.0, prod_ids.1, symbols.clone(), (*s).clone(), score));
                    }
                }
            }
            let best = best?;

            // TODO: Only do this if the score is < 0!
            //
            // Now we need to go back and reconstruct these changes:
            let ddd = ww.get_donkey(&best.3);
            let a = ddd.get_rule(best.0);
            let b = ddd.get_rule(best.1);

            let symbols = best.2;

            let mut builder: DeltaBuilder<T> = DeltaBuilder::new(); 
            // 2 base rules
            let s:Symbol = l.new_symbol();
            builder.add(
                Production{
                    from:s.clone(),
                    weight:a.w,
                    to: a.p.to_vec(),
                }
            );
            builder.add(
                Production{
                    from:s.clone(),
                    weight:b.w,
                    to: b.p.to_vec(),
                }
            );

            let pair_ab_1 = two_rule_helper::RulePair{a,b};
            for ss in &symbols {
                let ddd = ww.get_donkey(ss);
                let a = ddd.get_rule(best.0);
                let b = ddd.get_rule(best.1);
                let pair_ab_2 = two_rule_helper::RulePair{a,b};
                let (w_ab, w_a, w_b) = two_rule_helper::get_weights(&pair_ab_2, &pair_ab_1);

                builder.add(
                    Production{
                        from:ss.clone(),
                        weight: nf32(w_ab),
                        to: vec![ SymbolOrLiteral::Symbol(s.clone()) ],
                    }
                );
                if w_a != 0.0 {
                    builder.reweight(pair_ab_2.a.id, w_a);
                } else {
                    builder.remove(pair_ab_2.a.id, pair_ab_2.a.p.len());
                }
                if  w_b != 0.0 {
                    builder.reweight(pair_ab_2.b.id, w_a);
                } else {
                    builder.remove(pair_ab_2.b.id, pair_ab_2.b.p.len());
                }
            }

            let delta = builder.build();
            Some(delta)
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

pub fn substring_count_into<'a, T, H>(v: &'a [T], m:&mut HashMap<Vec<T>, usize, H>) 
where
    T: Hash + Eq + Clone,
    H: std::hash::BuildHasher,
{
    for i in 0..v.len() {
        for j in (i + 1)..=v.len() {
            if let Some(vv) = m.get_mut(&v[i..j]) {
                *vv +=1;
            } else {
                m.insert(v[i..j].to_vec(), 1);
            }
        }
    }
}

pub fn substring_uncount_into<'a, T, H>(v: &'a [T], m:&mut HashMap<Vec<T>, usize, H>) 
where
    T: Hash + Eq + Clone,
    H: std::hash::BuildHasher,
{
    for i in 0..v.len() {
        for j in (i + 1)..=v.len() {
            if let Some(vv) = m.get_mut(&v[i..j]) {
                assert!( *vv > 0);
                *vv -=1;
            } else {
                unreachable!("Should not get a negative value");
            }
        }
    }
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

pub fn has_subsequence<T>(input: &[T], needle: &[T]) -> bool 
where
    T: PartialEq,
{
    input.windows(needle.len()).any(|c| c == needle)
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


pub fn remove_best_subseq_from_language<T>(
    l: &language::raw::Language<T>,
) -> Option<language::raw::Language<T>>
where
    T: Ord + Clone + Hash + std::fmt::Debug,
{
    let mut l: language_manipulation::ModifiableLanguage<T> = l.into();
    use language_manipulation::Proposer;

    let max_extracted=32;
    let mut i = 0;
    loop {
        let proposer = language_manipulation::PairExtractionProposer;
        let p0 = proposer.get_proposal(&l);
        println!("Suggested pair removal {:?} {} / {}", p0.as_ref().map(|p| p.cost()), i, max_extracted);
        if let Some(p0) = p0 {
            if p0.cost() <= 0.0 {
                p0.apply(&mut l);
            } else {
                break
            }
        } else {
            break
        }
        if i >= max_extracted {
            break
        }
        i += 1;
    }

    let proposer = language_manipulation::FactorPrefixProposer;
    let p1 = proposer.get_proposal(&l);
    println!("Suggested prefix removal {:?}", p1.as_ref().map(|p| p.cost()));

    let proposer = language_manipulation::FactorSuffixProposer;
    let p2 = proposer.get_proposal(&l);
    println!("Suggested suffix removal {:?}", p2.as_ref().map(|p| p.cost()));

    let proposer = language_manipulation::ExtractSequenceProposer;
    let p3 = proposer.get_proposal(&l);
    println!("Suggested sequence removal {:?}", p3.as_ref().map(|p| p.cost()));

    vec![p1, p2, p3]
        .into_iter()
        .filter_map(|f| f)
        .max_by_key(|p| language::raw::nf32(-p.cost()) )
        .map(|p| {
            println!("Applying {:?}", p.cost());
            p.apply(&mut l);
            (&l).into()
        })
}


#[cfg(test)]
mod tests {
    use super::*;
    use language::raw::nf32;

    #[test]
    pub fn simple_substring_count() {
        let v = "abc";
        let mut n = HashMap::new();
        substring_count_into(v.as_bytes(), &mut n);
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
        let mut n = HashMap::new();
        substring_count_into(v.as_bytes(), &mut n);
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
        let mut n = HashMap::new();
        substring_count_into(v.as_bytes(), &mut n);
        let n: HashMap<Vec<u8>, usize> = n.into_iter()
            .map(|(k, v)| { let l = k.len(); (k, (v - 1) * (l - 1))} )
            .collect();
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
            weight: nf32(weight as f32),
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
                    prod("A", 1, vec![s("P1"), s("s_0")]),
                    prod("B", 1, vec![s("s_0"), s("S1")]),
                    prod("C", 1, vec![s("P2"), s("s_0"), s("S2")]),
                    prod("s_0", 1, vec![s("X"), l('a'), l('b')]),
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

    pub fn language_size<T>(l: &language::raw::Language<T>) -> usize {
        l.entries.iter().map(|e| e.to.len()).sum()
    }

    pub fn language_size_m<T>(l: &language_manipulation::ModifiableLanguage<T>) -> usize {
        l.entries.iter().map(|r| r.to.len()).sum()
    }

    #[test]
    pub fn test_derive_names() {
        println!("CARGO_MANIFEST_DIR={}", env!("CARGO_MANIFEST_DIR"));
        let names = format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            //"../resources/all_names_lc.text"
            "../resources/boys_names.txt"
            //"../resources/web2.txt"
        );
        println!("names={}", names);
        let names = std::fs::read_to_string(names).unwrap();
        for name in names.lines() {
            println!("{}", name);
        }
        let s = language::raw::Symbol::new("N");
        let entries: Vec<language::raw::Production<String>> = names
            .lines()
            .map(|name| language::raw::Production {
                from: s.clone(),
                weight: nf32(1.0),
                to: vec![language::raw::SymbolOrLiteral::literal(name)],
            })
            .collect();
        let l = language::raw::Language { entries };
        let mut ll = shatter_language(l);
        let nmax:i32=1500;
        for i in 0..nmax {
            println!("==== language size = {} @ {} / {} ====", language_size(&ll), i+1, nmax);
            ll = remove_best_subseq_from_language(&ll).unwrap();
        }
        println!("==== language size = {} ====", language_size(&ll));

        let l = unshatter_language(ll);
        for e in l.entries {
            println!("{:?}", e);
        }
    }

    
    #[test]
    pub fn test_substring_extract_speed() {
        use rand::{seq::IteratorRandom, thread_rng};
        println!("CARGO_MANIFEST_DIR={}", env!("CARGO_MANIFEST_DIR"));
        let names = format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            //"../resources/all_names_lc.text"
            //"../resources/boys_names.txt"
            "../resources/web2.txt"
        );
        println!("names={}", names);
        let names = std::fs::read_to_string(names).unwrap();
        for name in names.lines() {
            println!("{}", name);
        }
        let s = language::raw::Symbol::new("N");

        let mut rng = thread_rng();
        let names_subset: Vec<&str> = names
            .lines()
            .collect()
            //.choose_multiple(&mut rng, 20000)
            ;


        let entries: Vec<language::raw::Production<String>> = 
            names_subset.into_iter()
            .map(|name| language::raw::Production {
                from: s.clone(),
                weight: nf32(1.0),
                to: vec![language::raw::SymbolOrLiteral::literal(name)],
            })
            .collect();
        let l = language::raw::Language { entries };
        let ll = shatter_language(l);
        let nmax:i32=1500;
        //let nmax:i32=15;
        let mut lm: language_manipulation::ModifiableLanguage<char> = (&ll).into();
        use language_manipulation::Proposer;
        let proposer = language_manipulation::ExtractSequenceProposer;
        let start = std::time::Instant::now();
        for i in 0..nmax {
            let now=std::time::Instant::now();
            println!("==== language size = {} @ {} / {} : {:?} ====", language_size_m(&lm), i+1, nmax, now.duration_since(start));
            let p = proposer.get_proposal(&lm);
            //println!("Suggested sequence removal {:?}", p.as_ref().map(|p| p.cost()));
            if let Some(p) = p {
                p.apply(&mut lm);
            } else {
                break
            }
        }
        println!("==== language size = {} ====", language_size_m(&lm));
        let ll: language::raw::Language<char> = (&lm).into();
        let l = unshatter_language(ll);
        for e in l.entries {
            println!("{:?}", e);
        }
    }

    #[test]
    pub fn test_find_prefix() {
        use language::raw::Language;

        let lang: Language<String> = Language {
            entries: vec![
                prod("Q", 1, vec![sl("AAABA")]), // 5
                prod("Q", 1, vec![sl("AABA")]),  // 4
                prod("Q", 1, vec![sl("AAACA")]), // 5
                prod("Q", 1, vec![sl("AARX")]),  // 4
                prod("Q", 1, vec![sl("N")]),  // 1

            ],
        };
        let lang = shatter_language(lang);

        assert_eq!(language_size(&lang), 19);


        // The repeated prefixes as A which occurs 4 times, 
        // and AA which occurs 4 times, and AAA which occurs 3 times
        //
        // replacing 
        //   4 x Q -> 'A' [..]'  // 4 x (1 + X)
        //   (total 4X + 4) 
        // with 
        //       Q -> 'A' P  // 2
        //   4 x P -> [..]   // 4 x X
        //   (total 4X + 2)
        // saving of 2
        //
        // replacing 
        //   4 x Q -> 'AA' [..]'  // 4 x (2 + X)
        //   (total 4X + 8) 
        // with 
        //       Q -> 'AA' P  // 3
        //   4 x P -> [..]   // 4 x X
        //   (total 4X + 3)
        // saving of 5
        //
        // replacing 
        //   2 x Q -> 'AAA' [..]'  // 2 x (3 + X)
        //   (total 2X + 6) 
        // with 
        //       Q -> 'AAA' P  // 4
        //   2 x P -> [..]   // 2 x X
        //   (total 2X + 4)
        // saving of 2
        //
        // so the best is 'AA' with a saving of 5

        let expected_lang: Language<char> = shatter_language(Language {
            entries: vec![
                prod("Q", 4, vec![sl("AA"), s("P")]), // 3
                prod("Q", 1, vec![sl("N")]),          // 1
                prod("P", 1, vec![sl("ABA")]),        // 3
                prod("P", 1, vec![sl("ACA")]),        // 3
                prod("P", 1, vec![sl("BA")]),         // 2 
                prod("P", 1, vec![sl("RX")]),         // 2
            ],
        });

        assert_eq!(language_size(&expected_lang), 14);

        let m = language_manipulation::FactorPrefixProposer::production_iter_to_prefix_map(lang.entries.iter());
        let fp = language_manipulation::FactorPrefixProposer::get_proposal_from_prefix_map(&m).unwrap();
        assert_eq!(fp.symbol, language::raw::Symbol::new("Q"));
        assert_eq!(fp.prefix, vec![l('A'), l('A')]);
        let lm: language_manipulation::ModifiableLanguage<char> = (&lang).into();

        let delta = fp.apply(&lm);
        assert!((delta.cost() - -5.0).abs() < 1e-6);
    }

    #[test]
    pub fn test_find_suffix() {
        use language::raw::Language;

        let lang: Language<String> = Language {
            entries: vec![
                prod("Q", 1, vec![sl("AAABA")]), // 5
                prod("Q", 1, vec![sl("AABA")]),  // 4
                prod("Q", 1, vec![sl("AAACA")]), // 5
                prod("Q", 1, vec![sl("AARX")]),  // 4
                prod("Q", 1, vec![sl("N")]),  // 1

            ],
        };
        let lang = shatter_language(lang);

        assert_eq!(language_size(&lang), 19);


        // The repeated suffixes as A which occurs 4 times, 
        // and AA which occurs 4 times, and AAA which occurs 3 times
        //
        // replacing 
        //   3 x Q -> [...] 'A'  // 3 x (1 + X)
        //   (total 3X + 3) 
        // with 
        //       Q ->  P 'A'  // 2
        //   3 x P -> [..]   // 3 x X
        //   (total 3X + 2)
        // saving of 1
        //
        // replacing 
        //   2 x Q -> [...] 'BA''  // 2 x (2 + X)
        //   (total 2X + 4) 
        // with 
        //       Q -> P 'AA'  // 3
        //   2 x P -> [..]   // 2 x X
        //   (total 2X + 3)
        // saving of 1
        //
        // replacing 
        //   2 x Q -> [...] 'ABA''  // 2 x (3 + X)
        //   (total 2X + 6) 
        // with 
        //       Q -> 'ABA' P  // 4
        //   2 x P -> [..]   // 2 x X
        //   (total 2X + 4)
        // saving of 2
        //
        // replacing 
        //   2 x Q -> [...] 'AABA''  // 2 x (4 + X)
        //   (total 2X + 8) 
        // with 
        //       Q -> 'AABA' P  // 5
        //   2 x P -> [..]   // 2 x X
        //   (total 2X + 4)
        // saving of 3

        // so the best is 'AABA' with a saving of 3

        let expected_lang: Language<char> = shatter_language(Language {
            entries: vec![
                prod("Q", 2, vec![s("P"), sl("AABA")]), // 5
                prod("Q", 1, vec![sl("AAACA")]),        // 5
                prod("Q", 1, vec![sl("AARX")]),         // 4
                prod("Q", 1, vec![sl("N")]),            // 1
                prod("P", 1, vec![sl("A")]),            // 1
                prod("P", 1, vec![sl("")]),             // 0
            ],
        });

        assert_eq!(language_size(&expected_lang), 16);

        let m = language_manipulation::FactorSuffixProposer::production_iter_to_suffix_map(lang.entries.iter());
        let fp = language_manipulation::FactorSuffixProposer::get_proposal_from_suffix_map(&m).unwrap();
        assert_eq!(fp.symbol, language::raw::Symbol::new("Q"));
        assert_eq!(fp.suffix, vec![l('A'), l('A'), l('B'), l('A')]);
        let lm: language_manipulation::ModifiableLanguage<char> = (&lang).into();
        let delta = fp.apply(&lm);
        println!("fp = {:?}", fp);
        println!("delta = {:?}", delta);
        println!("delta.cost() = {}", delta.cost());

        assert!((delta.cost() - -3.0).abs() < 1e-6);
    }



    #[test]
    pub fn test_indexify_language() {
        use language::raw::Language;
        use super::language_manipulation::Proposer;

        let lang: Language<char> = Language {
            entries: vec![
                prod("Q", 1, vec![l('A'), s("B")]),
                prod("Q", 1, vec![l('R')]),
                prod("Q", 1, vec![l('X')]),
                prod("R", 2, vec![l('A'), s("B")]),
                prod("R", 1, vec![l('X')]),
                prod("S", 1, vec![s("C")])
            ],
        };

        let mut lang = language_manipulation::ModifiableLanguage::from(&lang);
        for r in lang.entries.iter() {
            println!("{:?}", r);
        }

        let delta = language_manipulation::PairExtractionProposer.get_proposal(&lang).unwrap();

        for p in &delta.changes {
            println!("  {:?}", p);
        }
        println!("...modifying language...");
        delta.apply(&mut lang);
        for r in lang.entries.iter() {
            println!("{:?}", r);
        }
        unimplemented!("Need to convert productions to indices");
    }

    #[test]
    pub fn test_pair_extract_when_no_pair() {
        use language::raw::Language;
        use super::language_manipulation::Proposer;

        let lang: Language<char> = Language {
            entries: vec![
                prod("Q", 1, vec![l('A'), s("B")]),
                prod("Q", 1, vec![l('R')]),
                prod("Q", 1, vec![l('X')]),
                prod("R", 1, vec![l('X')]),
                prod("S", 1, vec![s("C")])
            ],
        };

        let lang = language_manipulation::ModifiableLanguage::from(&lang);
        let delta = language_manipulation::PairExtractionProposer.get_proposal(&lang);
        assert!(delta.is_none());
    }

    #[test]
    pub fn test_add_delta_updates_substrings_map() {
        use language::raw::Language;
        use language::raw::SymbolOrLiteral;
        use language_manipulation::DeltaBuilder;

        let lang: Language<char> = Language {
            entries: vec![
                prod("A", 1, vec![l('A'),l('B'), l('C')]),
                prod("B", 1, vec![l('B'), l('C')])
            ]
        };
        let mut lang = language_manipulation::ModifiableLanguage::from(&lang);

        let expected: HashMap<Vec<SymbolOrLiteral<char>>, usize> = 
            vec![
                (vec![l('A')], 1usize), 
                (vec![l('B')], 2usize),
                (vec![l('C')], 2usize),
                (vec![l('A'), l('B')], 1usize),
                (vec![l('B'), l('C')], 2usize),
                (vec![l('A'), l('B'), l('C')], 1usize)
            ].into_iter().collect();

        assert_eq!(
            lang.substring_map,
            expected
        );

        let mut builder: DeltaBuilder<char> = DeltaBuilder::new();
        builder.add(prod("C", 1, vec![l('B'), l('C')]));
        let delta = builder.build();
        delta.apply(&mut lang);

        let expected: HashMap<Vec<SymbolOrLiteral<char>>, usize> = 
        vec![
            (vec![l('A')], 1usize), 
            (vec![l('B')], 3usize),
            (vec![l('C')], 3usize),
            (vec![l('A'), l('B')], 1usize),
            (vec![l('B'), l('C')], 3usize),
            (vec![l('A'), l('B'), l('C')], 1usize)
        ].into_iter().collect();

        assert_eq!(
            lang.substring_map,
            expected
        );
    }

    #[test]
    pub fn test_replace_delta_updates_substrings_map() {
        use language::raw::Language;
        use language::raw::SymbolOrLiteral;
        use language_manipulation::DeltaBuilder;
        use language_manipulation::RuleId;
        use crate::tombstone_map::Token;


        let lang: Language<char> = Language {
            entries: vec![
                prod("A", 1, vec![l('A'),l('B'), l('C')]),
                prod("B", 1, vec![l('B'), l('C')])
            ]
        };
        let mut lang = language_manipulation::ModifiableLanguage::from(&lang);

        let expected: HashMap<Vec<SymbolOrLiteral<char>>, usize> = 
            vec![
                (vec![l('A')], 1usize), 
                (vec![l('B')], 2usize),
                (vec![l('C')], 2usize),
                (vec![l('A'), l('B')], 1usize),
                (vec![l('B'), l('C')], 2usize),
                (vec![l('A'), l('B'), l('C')], 1usize)
            ].into_iter().collect();

        assert_eq!(
            lang.substring_map,
            expected
        );

        let mut builder: DeltaBuilder<char> = DeltaBuilder::new();
        builder.change_production(RuleId( Token{slot:0, counter:0}), 3, prod("C", 1, vec![l('A'), l('C')]));
        let delta = builder.build();
        delta.apply(&mut lang);

        let expected: HashMap<Vec<SymbolOrLiteral<char>>, usize> = 
        vec![
            (vec![l('A')], 1usize), 
            (vec![l('B')], 1usize),
            (vec![l('C')], 2usize),
            (vec![l('A'), l('B')], 0usize),
            (vec![l('A'), l('C')], 1usize),
            (vec![l('B'), l('C')], 1usize),
            (vec![l('A'), l('B'), l('C')], 0usize)
        ].into_iter().collect();

        assert_eq!(
            lang.substring_map,
            expected
        );
    }
}
