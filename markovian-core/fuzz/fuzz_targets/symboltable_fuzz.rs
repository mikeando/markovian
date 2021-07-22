#![no_main]
use arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use markovian_core::symbol::{SymbolTable, SymbolTableEntry, SymbolTableEntryId};

#[derive(Debug)]
enum SymbolTableAction {
    // There are only 3 actions that might mutate the internals of a SymbolTable
    AddAction(SymbolTableEntry<char>),
    RemoveAction(SymbolTableEntryId),
    CompactAction,
}

impl SymbolTableAction {
    fn apply(&self, symbol_table: &mut SymbolTable<char>) {
        match self {
            SymbolTableAction::AddAction(entry) => {
                symbol_table.add(entry.clone());
            }
            SymbolTableAction::RemoveAction(entry_id) => {
                symbol_table.remove(*entry_id);
            }
            SymbolTableAction::CompactAction => {
                symbol_table.compact();
            }
        }
    }
}

fn generate_ste(u: &mut Unstructured) -> Result<SymbolTableEntry<char>> {
    let mut options: Vec<fn(&mut Unstructured) -> Result<SymbolTableEntry<char>>> = vec![];

    options.push(|_u| Ok(SymbolTableEntry::Start));
    options.push(|_u| Ok(SymbolTableEntry::End));
    options.push(|u| Ok(SymbolTableEntry::Single(u.arbitrary()?)));
    options.push(|u| Ok(SymbolTableEntry::Compound(u.arbitrary()?)));
    options.push(|u| Ok(SymbolTableEntry::Dead(u.arbitrary()?)));

    let f = u.choose(&options)?;
    f(u)
}

impl Arbitrary for SymbolTableAction {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        let mut options: Vec<fn(&mut Unstructured) -> Result<SymbolTableAction>> = vec![];

        options.push(|u| {
            let ste: SymbolTableEntry<char> = generate_ste(u)?;
            Ok(SymbolTableAction::AddAction(ste))
        });

        options.push(|u| {
            Ok(SymbolTableAction::RemoveAction(SymbolTableEntryId(
                u.arbitrary()?,
            )))
        });

        options.push(|_u| Ok(SymbolTableAction::CompactAction));

        let f = u.choose(&options)?;
        f(u)
    }
}

#[derive(Debug)]
struct SymbolTableActions {
    actions: Vec<SymbolTableAction>,
}

impl Arbitrary for SymbolTableActions {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        let n: usize = u.int_in_range(0..=50)?;
        let mut actions: Vec<SymbolTableAction> = vec![];
        for i in 0..n {
            actions.push(u.arbitrary()?)
        }

        Ok(SymbolTableActions { actions })
    }
}

fuzz_target!(|symbol_table_actions: SymbolTableActions| {
    let mut symbol_table: SymbolTable<char> = SymbolTable::new();

    for action in symbol_table_actions.actions {
        action.apply(&mut symbol_table);
        symbol_table.check_internal_consistency();
    }
});
