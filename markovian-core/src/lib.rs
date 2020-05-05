#![feature(drain_filter)]

pub mod language;
pub mod markov_model;
pub mod pattern;
pub mod sequence_map;
pub mod substrings;
pub mod symbol;
pub mod weighted_sampler;
pub mod tombstone_map;
pub mod syncedheap;


extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
