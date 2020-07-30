use snafu::{ensure, OptionExt, Snafu};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Slot out of range"))]
    SlotOutOfRange,
    #[snafu(display("Slot was dead, but expected live"))]
    UnexpectedDead,
    #[snafu(display(
        "Token counter ({}) and slot counter ({}) did not match",
        token_counter,
        slot_counter
    ))]
    CounterMismatch {
        token_counter: u32,
        slot_counter: u32,
    },
    #[snafu(display("Internal Error"))]
    InternalError,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub struct Token {
    pub slot: usize,
    pub counter: u32,
}

#[derive(Clone, Debug)]
struct LiveEntry<T> {
    value: T,
}

#[derive(Clone, Debug)]
struct DeadEntry {
    next_free: Option<usize>,
}

#[derive(Clone, Debug)]
enum TombstoneEntry<T> {
    Live(LiveEntry<T>),
    Tombstone(DeadEntry),
}

impl<T> TombstoneEntry<T> {
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

    fn live(value: T) -> TombstoneEntry<T> {
        TombstoneEntry::Live(LiveEntry { value })
    }

    fn dead(next_free: Option<usize>) -> TombstoneEntry<T> {
        TombstoneEntry::Tombstone(DeadEntry { next_free })
    }
}

#[derive(Clone, Debug)]
pub struct TombstoneList<T> {
    entries: Vec<(u32, TombstoneEntry<T>)>,
    free_list_head: Option<usize>,
}

// Vec like with O(1) insert and remove.
impl<T> TombstoneList<T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.entries
            .iter()
            .filter_map(|e| e.1.as_live())
            .map(|v| &v.value)
    }

    pub fn new() -> TombstoneList<T> {
        TombstoneList {
            entries: vec![],
            free_list_head: None,
        }
    }

    pub fn get(&self, token: &Token) -> Result<&T, Error> {
        let e = self.entries.get(token.slot).context(SlotOutOfRange)?;
        let live = e.1.as_live().context(UnexpectedDead)?;
        ensure!(
            e.0 == token.counter,
            CounterMismatch {
                token_counter: token.counter,
                slot_counter: e.0
            }
        );
        Ok(&live.value)
    }

    pub fn add(&mut self, v: T) -> Token {
        // We need a new entry.
        // do we have a free slot?
        if let Some(idx) = self.free_list_head {
            let e = &self.entries[idx];

            let dead = e.1.as_dead().unwrap();
            self.free_list_head = dead.next_free;

            //Now we replace e and add to the index.
            let new_token = Token {
                slot: idx,
                counter: e.0 + 1,
            };
            self.entries[idx] = (new_token.counter, TombstoneEntry::live(v));
            return new_token;
        }

        // No free slot, we need to add a new entry to the internal vec.
        let new_token = Token {
            slot: self.entries.len(),
            counter: 0,
        };
        self.entries
            .push((new_token.counter, TombstoneEntry::live(v)));
        new_token
    }

    pub fn remove(&mut self, token: &Token) -> Result<T, Error> {
        // Find the current value
        let e = self.entries.get(token.slot).context(SlotOutOfRange)?;

        // It should be live - we check it before swapping so we can
        // leave the input unchanged in case of error.
        e.1.as_live().context(UnexpectedDead)?;
        ensure!(
            e.0 == token.counter,
            CounterMismatch {
                token_counter: token.counter,
                slot_counter: e.0
            }
        );

        // Hook it into the free list and swap out the old.
        let mut v = TombstoneEntry::dead(self.free_list_head);
        std::mem::swap(&mut v, &mut self.entries[token.slot].1);
        self.free_list_head = Some(token.slot);
        Ok(v.into_live().context(InternalError)?.value)
    }
}

impl<T> Default for TombstoneList<T> {
    fn default() -> Self {
        Self::new()
    }
}
