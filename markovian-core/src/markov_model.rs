use crate::symbol::Symbol;
use crate::weighted_sampler::WeightedSampler;
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct MarkovModel {
    pub contexts: BTreeMap<Vec<Symbol>, WeightedSampler<Symbol>>,
    pub order: usize,
}

impl MarkovModel {
    pub fn new(order: usize) -> MarkovModel {
        MarkovModel {
            contexts: BTreeMap::new(),
            order,
        }
    }

    pub fn add(&mut self, s: &[Symbol]) {
        //println!("{:?} => {:?}", s , ss);
        use std::iter;
        let ss: Vec<Symbol> = iter::repeat(Symbol::Start)
            .take(self.order)
            .chain(s.iter().cloned())
            .chain(iter::repeat(Symbol::End).take(1))
            .collect();

        for w in ss.windows(self.order + 1) {
            for cl in 0..self.order {
                self.contexts
                    .entry(w[cl..self.order].to_vec())
                    .or_insert_with(WeightedSampler::new)
                    .add_symbol(&w[self.order]);
            }
        }
    }

    pub fn initial_context(&self) -> Vec<Symbol> {
        std::iter::repeat(Symbol::Start).take(self.order).collect()
    }

    pub fn sample_next_symbol<R: Rng>(&self, context: &[Symbol], rng: &mut R) -> Symbol {
        for i in 0..context.len() {
            let weights = self.contexts.get(&context[i..]);
            if let Some(ws) = weights {
                return ws.sample_next_symbol(rng);
            }
        }
        unreachable!();
    }

    pub fn convert_string_to_symbols(&self, s: &str) -> Vec<Symbol> {
        use crate::sequence_map::SequenceMapNode;
        // We use a greedy algorithm that is not guaranteed to work.
        // Find all the compound symbols we support.
        let mut mapper: SequenceMapNode<u8, Symbol> = SequenceMapNode::new();
        for context in self.contexts.keys() {
            for s in context {
                if let Symbol::Compound(v) = s {
                    mapper.insert(v, Some(s.clone()));
                }
            }
        }
        let mapper = mapper;
        let mut result = vec![];
        let mut st = s.as_bytes();
        while !st.is_empty() {
            match mapper.transform_prefix(&st) {
                Some((sy, stn)) => {
                    st = stn;
                    result.push(sy.clone());
                }
                None => {
                    result.push(Symbol::Char(st[0]));
                    st = &st[1..];
                }
            }
        }
        result
    }

    pub fn sample_starting_with<R: Rng>(
        &self,
        prefix_symbols: &[Symbol],
        rng: &mut R,
    ) -> Vec<Symbol> {
        let mut context = self.initial_context();
        let L = context.len();
        let mut symbols = vec![];

        // Put in the prefix.
        for s in prefix_symbols {
            symbols.push(s.clone());
            context.rotate_left(1);
            context[L - 1] = s.clone();
        }

        //Handle the rest
        loop {
            let s = self.sample_next_symbol(&context, rng);
            match &s {
                Symbol::End => break,
                Symbol::Start => unimplemented!(),
                _ => {}
            }
            symbols.push(s.clone());
            context.rotate_left(1);
            context[L - 1] = s;
        }
        symbols
    }
}
