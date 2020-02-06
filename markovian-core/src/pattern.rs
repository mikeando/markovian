#[derive(Debug, Copy, Clone, PartialEq)]
struct Terminal(u8);

#[derive(Debug, Copy, Clone, PartialEq)]
struct NonTerminal(u8);

#[derive(Debug, Copy, Clone, PartialEq)]
enum Symbol {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
    Empty,
}

#[derive(Debug, Copy, Clone)]
struct ProductionRule {
    s: Option<NonTerminal>,
    weight: u8,
    results: [Symbol; 4],
}

impl ProductionRule {
    pub fn new() -> ProductionRule {
        ProductionRule {
            s: None,
            weight: 0,
            results: [Symbol::Empty; 4],
        }
    }
}

//#[derive(Debug)]
struct Language {
    t: [ProductionRule; 256],
}

impl Language {
    pub fn new() -> Language {
        let t: [ProductionRule; 256] = [ProductionRule::new(); 256];
        Language { t }
    }

    pub fn produce<'a>(&self, result_buffer: &mut Vec<Symbol>, expansion_buffer: &mut Vec<Symbol>) {
        // We always start from NonTerminal[0], so stick that in to start with
        result_buffer.clear();
        expansion_buffer.clear();
        expansion_buffer.push(Symbol::NonTerminal(NonTerminal(0)));

        while !expansion_buffer.is_empty() {
            self.expand_one(result_buffer, expansion_buffer);
        }
    }

    pub fn find_rule(&self, t: NonTerminal) -> Option<&ProductionRule> {
        for x in (&self.t).iter() {
            if let Some(s) = x.s {
                if s == t {
                    return Some(x);
                }
            }
        }
        None
    }

    pub fn expand_one(&self, result_buffer: &mut Vec<Symbol>, expansion_buffer: &mut Vec<Symbol>) {
        if expansion_buffer.is_empty() {
            return;
        }
        let s = expansion_buffer.pop().unwrap();
        match s {
            Symbol::Empty => {}
            Symbol::Terminal(t) => result_buffer.push(Symbol::Terminal(t)),
            Symbol::NonTerminal(t) => {
                let r: Option<&ProductionRule> = self.find_rule(t);
                if let Some(r) = r {
                    for x in r.results.iter().rev() {
                        expansion_buffer.push(*x)
                    }
                } else {
                    unimplemented!()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_language() -> Language {
        let mut language = Language::new();
        language.t[0].s = Some(NonTerminal(0));
        language.t[0].weight = 1;
        language.t[0].results = [
            Symbol::Terminal(Terminal(1)),
            Symbol::Empty,
            Symbol::Empty,
            Symbol::Empty,
        ];
        return language;
    }

    #[test]
    fn test_language_can_produce_single() {
        let mut language = dummy_language();
        let mut result = vec![];
        let mut expansion_buffer = vec![];
        language.produce(&mut result, &mut expansion_buffer);
        assert_eq!(result, vec![Symbol::Terminal(Terminal(1))]);

        language.t[0].results = [
            Symbol::Terminal(Terminal(2)),
            Symbol::Empty,
            Symbol::Empty,
            Symbol::Empty,
        ];
        language.produce(&mut result, &mut expansion_buffer);
        assert_eq!(result, vec![Symbol::Terminal(Terminal(2))]);
    }

    /*
    #[test]
    fn test_expand_dummy() {
        let mut language = dummy_language();
        let mut result:[Symbol; 4] = [Symbol::Terminal(Terminal(1)), Symbol::Empty, Symbol::Empty, Symbol::Empty];
        language.expand_one(&result, 0, 1);
        let mut result:[Symbol ; 32] = [Symbol::Empty; 32];
        let r = language.produce(&mut result);
        assert_eq!(r, &[Symbol::Terminal(Terminal(1))]);

        language.t[0].results = [Symbol::Terminal(Terminal(2)), Symbol::Empty, Symbol::Empty, Symbol::Empty];
        let r = language.produce(&mut result);
        assert_eq!(r, &[Symbol::Terminal(Terminal(2))]);
    }
    */
}
