use std::rc::Rc;
use std::hash::Hash;
use std::collections::HashMap;

use snafu::{ensure, Backtrace, ErrorCompat, ResultExt, OptionExt, Snafu};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Slot out of range"))]
    SlotOutOfRange,
    #[snafu(display("Slot was dead, but expected live"))]
    UnexpectedDead,
    #[snafu(display("Token counter ({}) and slot counter ({}) did not match", token_counter, slot_counter))]
    CounterMismatch{
        token_counter:u32,
        slot_counter:u32,
    },
    #[snafu(display("Internal Error"))]
    InternalError,
}

#[derive(Clone, Debug)]
struct LiveEntry<T> {
    value:T,
}

#[derive(Clone, Debug)]
struct DeadEntry {
    next_free:Option<usize>,
}

#[derive(Clone, Debug)]
enum TombstoneEntry<T> {
    Live(LiveEntry<T>),
    Tombstone(DeadEntry),
}

impl <T> TombstoneEntry<T> {
    fn as_live(&self) -> Option<&LiveEntry<T>> {
        match self {
            TombstoneEntry::Live(v) => Some(v),
            TombstoneEntry::Tombstone(_) => None,
        }
    }

    fn into_live(self) -> Option<LiveEntry<T>> {
        match self {
            TombstoneEntry::Live(v) => Some(v),
            TombstoneEntry::Tombstone(_) => None,
        }
    }

    fn as_dead(&self) -> Option<&DeadEntry> {
        match self {
            TombstoneEntry::Live(_) => None,
            TombstoneEntry::Tombstone(v) => Some(v),
        }
    }

    fn live(value:T) -> TombstoneEntry<T> {
        TombstoneEntry::Live(LiveEntry{ value })
    }

    fn dead(next_free:Option<usize>) -> TombstoneEntry<T> {
        TombstoneEntry::Tombstone(DeadEntry{ next_free })
    }
}

#[derive(Clone, Debug)]
pub struct TombstoneVec<T> {
    entries: Vec<(u32, TombstoneEntry<T>)>,
    free_list_head: Option<usize>,
}


// Vec like with O(1) insert and remove. 
impl <T> TombstoneVec<T> {
    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.entries
            .iter()
            .filter_map( |e| e.1.as_live() )
            .map( |v| &v.value )
    }

    pub fn new() -> TombstoneVec<T> {
        TombstoneVec { entries:vec![], free_list_head:None }
    }

    pub fn get_by_token(&self, token:&Token) -> Result<&T, Error> {
        let e = self.entries.get(token.slot).context( SlotOutOfRange )?;
        let live = e.1.as_live().context( UnexpectedDead )?;
        ensure!(e.0 == token.counter, CounterMismatch{ token_counter:token.counter, slot_counter:e.0 } );
        Ok(&live.value)
    }

    pub fn insert(&mut self, v:T) -> Token {
        // We need a new entry. 
        // do we have a free slot?
        if let Some(idx) = self.free_list_head {
            let e = &self.entries[idx];

            let dead = e.1.as_dead().unwrap();
            self.free_list_head = dead.next_free;

            //Now we replace e and add to the index.
            let new_token = Token{ slot:idx, counter:e.0+1 };
            self.entries[idx] = (new_token.counter, TombstoneEntry::live(v));
            return new_token
        }

        // No free slot, we need to add a new entry to the internal vec.
        let new_token = Token{ slot:self.entries.len(), counter:0 };
        self.entries.push(
            (new_token.counter, TombstoneEntry::live(v))
        );
        new_token
    }

    pub fn remove_by_token(&mut self, token:&Token) -> Result<T, Error> {
        // Find the current value
        let e = self.entries.get(token.slot).context(SlotOutOfRange)?;

        // It should be live - we check it before swapping so we can 
        // leave the input unchanged in case of error.
        e.1.as_live().context(UnexpectedDead)?;
        ensure!(e.0 == token.counter, CounterMismatch{ token_counter:token.counter, slot_counter:e.0 } );

        // Hook it into the free list and swap out the old.
        let mut v = TombstoneEntry::dead(self.free_list_head);
        std::mem::swap(&mut v, &mut self.entries[token.slot].1);
        self.free_list_head = Some(token.slot);
        Ok(v.into_live().context(InternalError)?.value)
    }
}


// Designed to provide O(1) fast access to objects of type T
// by a generated small token.
//
// And also provide O(1)
// find for the token of a given T. Also O(1) 
// insert and remove, and insert an remove do not
// invalidate the existing tokens.
//
// Finally we want safety, in that reusing a removed token
// should be an error.
#[derive(Clone, Debug)]
pub struct TombstoneMap<T> {
    entries: Vec<(u32,TombstoneEntry<Rc<T>>)>,
    lookup: HashMap<Rc<T>, Token>,
    free_list_head: Option<usize>,
}

impl <T> TombstoneMap<T> {
    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.entries
            .iter()
            .filter_map( |e| e.1.as_live() )
            .map( |v| v.value.as_ref() )
    }

    pub fn iter_with_token(&self) -> impl Iterator<Item=(Token,&T)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map( 
                |(i,e)| {
                    e.1.as_live().map(
                        |v| (Token{ slot:i, counter:e.0 }, v.value.as_ref() )
                    )
                }
            )
    }
}

impl <T> TombstoneMap<T> 
    where T: Hash + Eq
{
    pub fn new() -> TombstoneMap<T> {
        TombstoneMap { entries:vec![], lookup:HashMap::new(), free_list_head:None }
    }

    pub fn get_by_value(&self, v:&T) -> Option<Token> {
        self.lookup.get(v).cloned()
    }

    pub fn insert_or_get_token(&mut self, v:T) -> Token {
        let rc = Rc::new(v);
        let existing = self.lookup.get(&rc);
        if let Some(existing) = existing {
            return *existing;
        }

        // We need a new entry. 
        // do we have a free slot?
        if let Some(idx) = self.free_list_head {
            let e = &self.entries[idx];

            let dead = e.1.as_dead().unwrap();
            self.free_list_head = dead.next_free;

            //Now we replace e and add to the index.
            let new_token = Token{ slot:idx, counter:e.0+1 };
            self.entries[idx] = (new_token.counter, TombstoneEntry::live(rc.clone()));
            self.lookup.insert(rc, new_token);
            return new_token
        }

        // No free slot, we need to add a new entry to the internal vec.
        let new_token = Token{ slot:self.entries.len(), counter:0 };
        self.entries.push(
            (new_token.counter, TombstoneEntry::live(rc.clone()))
        );
        self.lookup.insert(rc, new_token);
        new_token
    }

    pub fn get_by_token(&self, token:&Token) -> Result<&T, Error> {
        let e = self.entries.get(token.slot).context( SlotOutOfRange )?;
        let live = e.1.as_live().context( UnexpectedDead )?;
        ensure!(e.0 == token.counter, CounterMismatch{ token_counter:token.counter, slot_counter:e.0 } );
        Ok(&live.value)
    }

    pub fn remove_by_token(&mut self, token:&Token) -> Result<T, Error> {
        // Find the current value
        let e = self.entries.get(token.slot).context(SlotOutOfRange)?;

        // It should be live
        let live = e.1.as_live().context(UnexpectedDead)?;
        ensure!(e.0 == token.counter, CounterMismatch{ token_counter:token.counter, slot_counter:e.0 } );

        // Remove it from the lookup
        self.lookup.remove(&live.value).context(InternalError)?;

        // Hook it into the free list
        let rc = live.value.clone();
        self.entries[token.slot].1 = TombstoneEntry::dead(self.free_list_head);
        self.free_list_head = Some(token.slot);
        Ok(Rc::try_unwrap(rc).ok().context(InternalError)?)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub struct Token {
    slot:usize,
    counter:u32,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_map_insert_when_empty() {
        let mut m: TombstoneMap<String> = TombstoneMap::new();
        assert_eq!(m.get_by_value(&"Hello".to_string()), None);
        let id = m.insert_or_get_token("Hello".to_string());
        assert_eq!(id, Token{ slot:0, counter:0 } );
        assert_eq!(m.get_by_token(&id).unwrap(), &"Hello".to_string());
        assert_eq!(m.get_by_value(&"Hello".to_string()), Some(id));

        m.remove_by_token(&id).unwrap();
        assert_eq!(m.get_by_value(&"Hello".to_string()), None);
    }

    #[test]
    fn test_reuse_old_slots() {
        let mut m: TombstoneMap<String> = TombstoneMap::new();
        
        let _id_a = m.insert_or_get_token("A".to_string());
        let id_b = m.insert_or_get_token("B".to_string());
        let id_c = m.insert_or_get_token("C".to_string());
        let _id_d = m.insert_or_get_token("D".to_string());

        assert_eq!(m.entries.len(), 4);
        assert_eq!(m.lookup.len(), 4);
        assert_eq!(m.free_list_head, None);

        assert_eq!(m.remove_by_token(&id_b).unwrap(), "B");
        m.remove_by_token(&id_c).unwrap();
        
        assert_eq!(m.entries.len(), 4);
        assert_eq!(m.lookup.len(), 2);
        assert_eq!(m.free_list_head, Some(2));

        let id_e = m.insert_or_get_token("E".to_string());
        assert_eq!(m.entries.len(), 4);
        assert_eq!(m.lookup.len(), 3);
        assert_eq!(id_e, Token{slot:2, counter:1});
        assert_eq!(m.free_list_head, Some(1));

        let id_f = m.insert_or_get_token("F".to_string());
        assert_eq!(m.entries.len(), 4);
        assert_eq!(m.lookup.len(), 4);
        assert_eq!(id_f, Token{slot:1, counter:1});
        assert_eq!(m.free_list_head, None);

        let id_g = m.insert_or_get_token("G".to_string());
        assert_eq!(m.entries.len(), 5);
        assert_eq!(m.lookup.len(), 5);
        assert_eq!(id_g, Token{slot:4, counter:0});
        assert_eq!(m.free_list_head, None);
    }

    #[test]
    fn test_iter() {
        let mut m: TombstoneMap<String> = TombstoneMap::new();
        let _id_a = m.insert_or_get_token("A".to_string());
        let _id_b = m.insert_or_get_token("B".to_string());
        let _id_c = m.insert_or_get_token("C".to_string());
        let _id_d = m.insert_or_get_token("D".to_string());
        assert_eq!(m.iter().cloned().collect::<Vec<_>>(), vec!["A","B","C","D"]);
    }

    #[test]
    fn test_iter_with_token() {
        let mut m: TombstoneMap<String> = TombstoneMap::new();
        let _id_a = m.insert_or_get_token("A".to_string());
        let _id_b = m.insert_or_get_token("B".to_string());
        let _id_c = m.insert_or_get_token("C".to_string());
        let _id_d = m.insert_or_get_token("D".to_string());
        assert_eq!(m.iter_with_token().collect::<Vec<_>>(), vec![
            (Token{slot:0, counter:0}, &"A".to_string()),
            (Token{slot:1, counter:0}, &"B".to_string()),
            (Token{slot:2, counter:0}, &"C".to_string()),
            (Token{slot:3, counter:0}, &"D".to_string()),
        ]);
    }

    // #[test]
    // fn test_into_iter() {
    //     let mut m: TombstoneMap<String> = TombstoneMap::new();
    //     let _id_a = m.insert_or_get_token("A".to_string());
    //     let _id_b = m.insert_or_get_token("B".to_string());
    //     let _id_c = m.insert_or_get_token("C".to_string());
    //     let _id_d = m.insert_or_get_token("D".to_string());
    //     let mut combined = "".to_string();
    //     for x in &m {
    //         let xx: &String = x;
    //         combined = format!("{}{}", combined, x);
    //     }
    //     assert_eq!(combined, "ABCDEF");
    // }

    #[test]
    fn test_vec_insert_when_empty() {
        let mut m: TombstoneVec<String> = TombstoneVec::new();
        let id = m.insert("Hello".to_string());
        assert_eq!(id, Token{ slot:0, counter:0 } );
        assert_eq!(m.get_by_token(&id).unwrap(), &"Hello".to_string());
        m.remove_by_token(&id).unwrap();
        assert!(m.get_by_token(&id).is_err());
    }
}
