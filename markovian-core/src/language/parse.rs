use super::raw;
use raw::Production;
use raw::Symbol;
use raw::SymbolOrLiteral;

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError<'a> {
    MissingSymbol,
    InvalidSymbol(&'a str),
    GeneralError,
}

impl<'a> ParseError<'a> {
    pub fn description(&self) -> String {
        match self {
            ParseError::MissingSymbol => format!("missing symbol"),
            ParseError::InvalidSymbol(s) => format!("'{}' is not a valid symbol", s),
            ParseError::GeneralError => format!("general error"),
        }
    }
}

pub fn eat_spaces(v: &str) -> Result<((), &str), ParseError> {
    let mut rest = v;
    let mut it = v.chars();
    while let Some(r) = it.next() {
        if r.is_whitespace() {
            rest = it.as_str()
        } else {
            break;
        }
    }
    Ok(((), rest))
}

pub fn eat_nonspaces(v: &str) -> Result<(&str, &str), ParseError> {
    let mut rest = v;
    let mut it = v.char_indices();
    let mut end = 0;
    while let Some((i, r)) = it.next() {
        if !r.is_whitespace() {
            rest = it.as_str();
            end = i + r.len_utf8();
        } else {
            break;
        }
    }
    Ok((&v[0..end], rest))
}

pub fn is_symbol_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn eat_symbol_chars(v: &str) -> Result<(&str, &str), ParseError> {
    for (idx, cc) in v.char_indices() {
        if !is_symbol_character(cc) {
            return Ok((&v[..idx], &v[idx..]));
        }
    }
    Ok((v, ""))
}

pub fn take<'a>(p: &str, v: &'a str) -> Result<(&'a str, &'a str), ParseError<'a>> {
    let len = p.len();
    if v.len() < len {
        return Err(ParseError::GeneralError);
    }
    let pp = &v[0..len];
    if pp != p {
        return Err(ParseError::GeneralError);
    }
    Ok((pp, &v[len..]))
}

pub fn take_char(c: char, v: &str) -> Result<((), &str), ParseError> {
    let mut cit = v.chars();
    if cit.next() != Some(c) {
        Err(ParseError::GeneralError)
    } else {
        Ok(((), cit.as_str()))
    }
}

pub fn take_while_not_char(c: char, v: &str) -> Result<(&str, &str), ParseError> {
    for (idx, cc) in v.char_indices() {
        if cc == c {
            return Ok((&v[..idx], &v[idx..]));
        }
    }
    Err(ParseError::GeneralError)
}

pub fn parse_weight(v: &str) -> Result<(u32, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    let (x, rest): (&str, &str) = eat_nonspaces(rest)?;
    let (_, rest) = eat_spaces(rest)?;
    let w: u32 = x.parse::<u32>().map_err(|_| ParseError::GeneralError)?;
    Ok((w, rest))
}

pub fn parse_symbol(v: &str) -> Result<(Symbol, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    let (x, rest): (&str, &str) = eat_symbol_chars(rest).map_err(|_e| ParseError::MissingSymbol)?;
    let (_, rest) = eat_spaces(rest)?;
    if x.is_empty() {
        return Err(ParseError::MissingSymbol);
    }
    Ok((Symbol::new(x), rest))
}

pub fn parse_tag<'a>(tag: &str, v: &'a str) -> Result<(&'a str, &'a str), ParseError<'a>> {
    let (_, rest) = eat_spaces(v)?;
    let (x, rest): (&str, &str) = eat_nonspaces(rest)?;
    let (_, rest) = eat_spaces(rest)?;
    if x == tag {
        Ok((x, rest))
    } else {
        Err(ParseError::GeneralError)
    }
}

pub fn parse_string(v: &str) -> Result<(&str, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    let (_, rest) = take_char('"', rest)?;
    let (x, rest) = take_while_not_char('"', rest)?;
    let (_, rest) = take_char('"', rest)?;
    let (_, rest) = eat_spaces(rest)?;
    Ok((x, rest))
}

pub fn parse_alt<'a, F1, F2, I, R>(v: I, f1: F1, f2: F2) -> Result<R, ParseError<'a>>
where
    F1: Fn(I) -> Result<R, ParseError<'a>>,
    F2: Fn(I) -> Result<R, ParseError<'a>>,
    R: 'a,
    I: 'a + Copy,
{
    let r1 = f1(v);
    if r1.is_ok() {
        return r1;
    }
    let r2 = f2(v);
    if r2.is_ok() {
        return r2;
    }
    Err(ParseError::GeneralError)
}

//NOTE: I couldn't get this working cleanly with parse_alt
//      so I've just inlined it
pub fn parse_symbol_or_literal(v: &str) -> Result<(SymbolOrLiteral<String>, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    match parse_string(rest) {
        Ok((w, rest)) => {
            return Ok((SymbolOrLiteral::literal(w), rest));
        }
        Err(_e) => {}
    }
    match parse_symbol(rest) {
        Ok((w, rest)) => {
            return Ok((SymbolOrLiteral::Symbol(w), rest));
        }
        Err(_e) => {}
    }
    Err(ParseError::GeneralError)
}

pub fn parse_symbol_or_literal_list(
    v: &str,
) -> Result<(Vec<SymbolOrLiteral<String>>, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    let mut result = vec![];
    let mut rest = rest;
    loop {
        let r = parse_symbol_or_literal(rest);
        match r {
            Ok((s, rest_)) => {
                result.push(s);
                rest = rest_;
            }
            Err(_e) => {
                break;
            }
        }
    }
    Ok((result, rest))
}

pub fn parse_production(v: &str) -> Result<(Vec<Production<String>>, &str), ParseError> {
    let (weight, rest): (u32, &str) = parse_weight(v)?;
    let (from, rest): (Symbol, &str) = parse_symbol(rest)?;
    let (_, rest) = parse_tag("=>", rest)?;

    let mut options: Vec<Vec<SymbolOrLiteral<String>>> = vec![];
    let mut rest = rest;
    loop {
        let (symbols, r): (Vec<SymbolOrLiteral<String>>, &str) =
            parse_symbol_or_literal_list(rest)?;
        rest = r;
        options.push(symbols);
        let (_, r) = eat_spaces(r)?;
        let e = take_char('|', r);
        match e {
            Ok((_, r)) => {
                let (_, r) = eat_spaces(r)?;
                rest = r;
            }
            Err(_) => break,
        }
    }
    if options.is_empty() {
        return Err(ParseError::GeneralError);
    }

    let (_, rest) = eat_spaces(rest)?;
    Ok((
        options
            .into_iter()
            .map(|v| Production {
                weight,
                from: from.clone(),
                to: v,
            })
            .collect(),
        rest,
    ))
}

#[derive(Debug, PartialEq, Eq)]
pub struct Directive {
    pub name: String,
    pub arguments: Vec<SymbolOrLiteral<String>>,
}

pub fn parse_directive(v: &str) -> Result<(Directive, &str), ParseError> {
    let (_, rest) = eat_spaces(v)?;
    let (_, rest) = take_char('@', rest)?;
    let (name, rest) = parse_symbol(rest)?;
    let (_, rest) = eat_spaces(rest)?;
    let (_, rest) = take_char('(', rest)?;
    let (_, rest) = eat_spaces(rest)?;
    let (arguments, rest) = parse_symbol_or_literal_list(rest)?;
    let (_, rest) = take_char(')', rest)?;
    Ok((
        Directive {
            name: name.0,
            arguments,
        },
        rest,
    ))
}

pub enum Line {
    MultiProduction(Vec<Production<String>>),
    Directive(Directive),
}

pub fn parse_language_line(line: &str) -> Result<Line, ParseError> {
    let r = parse_directive(line);
    if let Ok(v) = r {
        let (_, rest) = eat_spaces(v.1)?;
        if rest.is_empty() {
            return Ok(Line::Directive(v.0));
        }
    }
    let r = parse_production(line);
    if let Ok(v) = r {
        let (_, rest) = eat_spaces(v.1)?;
        if rest.is_empty() {
            return Ok(Line::MultiProduction(v.0));
        }
    }
    Err(ParseError::GeneralError)
}

#[cfg(test)]
mod tests {

    use super::super::parse;
    use super::super::raw;

    fn prod(
        from: &str,
        weight: u32,
        to: Vec<raw::SymbolOrLiteral<String>>,
    ) -> raw::Production<String> {
        raw::Production {
            from: raw::Symbol::new(from),
            weight,
            to,
        }
    }

    fn s(k: &str) -> raw::SymbolOrLiteral<String> {
        raw::SymbolOrLiteral::symbol(k)
    }

    fn l(k: &str) -> raw::SymbolOrLiteral<String> {
        raw::SymbolOrLiteral::literal(k)
    }

    #[test]
    fn parse_single_rule() {
        let rule = r#"3 T => T "bar" Q"#;
        let (productions, _) = parse::parse_production(rule).unwrap();
        assert_eq!(
            productions,
            vec![prod("T", 3, vec![s("T"), l("bar"), s("Q")])]
        );
    }

    #[test]
    fn tokenize_alt_rule() {
        let rule = r#"3 T => T A | Q B"#;
        let (productions, _) = parse::parse_production(rule).unwrap();
        assert_eq!(
            productions,
            vec![
                prod("T", 3, vec![s("T"), s("A")]),
                prod("T", 3, vec![s("Q"), s("B")])
            ]
        );
    }

    #[test]
    fn tokenize_weight() {
        let rule = "3 XXX";
        let (weight, rest) = parse::parse_weight(rule).unwrap();
        assert_eq!(weight, 3);
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn tokenize_symbol() {
        assert_eq!(
            parse::parse_symbol("AX XXX"),
            Ok((raw::Symbol::new("AX"), "XXX"))
        );
        assert_eq!(
            parse::parse_symbol(""),
            Err(parse::ParseError::MissingSymbol)
        );
    }

    #[test]
    fn test_parse_take() {
        let rule = "XYZW";
        let (taken, rest) = parse::take("XY", rule).unwrap();
        assert_eq!(taken, "XY");
        assert_eq!(rest, "ZW");
    }

    #[test]
    fn test_take_char() {
        let rule = "XYZW";
        let (_, rest) = parse::take_char('X', rule).unwrap();
        assert_eq!(rest, "YZW");

        let e = parse::take_char('Y', rule);
        assert_eq!(e, Err(parse::ParseError::GeneralError));

        let rule = "⇄es";
        let (_, rest) = parse::take_char('⇄', rule).unwrap();
        assert_eq!(rest, "es");
    }

    #[test]
    fn test_eat_spaces_works_on_multibyte_string() {
        assert_eq!(parse::eat_spaces("⇄"), Ok(((), "⇄")));
        assert_eq!(parse::eat_spaces(" ⇄"), Ok(((), "⇄")));
    }

    #[test]
    fn test_eat_spaces_works_on_symbol_then_space_then_symbol() {
        assert_eq!(parse::eat_spaces("A B"), Ok(((), "A B")));
        assert_eq!(parse::eat_spaces(" A B"), Ok(((), "A B")));
    }

    #[test]
    fn test_eat_nonspaces_works_on_multibyte_string() {
        assert_eq!(parse::eat_nonspaces("⇄ hello"), Ok(("⇄", " hello")));
    }

    #[test]
    fn test_eat_nonspaces_works_on_empty_string() {
        assert_eq!(parse::eat_nonspaces(""), Ok(("", "")));
    }

    #[test]
    fn test_eat_nonspaces_works_on_whitespace_string() {
        assert_eq!(parse::eat_nonspaces("  "), Ok(("", "  ")));
    }

    #[test]
    fn test_eat_nonspaces_works_on_leading_whitespace_string() {
        assert_eq!(parse::eat_nonspaces(" X "), Ok(("", " X ")));
    }

    #[test]
    fn test_take_while_not_char() {
        let rule = "XYZW";
        let (pre, rest) = parse::take_while_not_char('Z', rule).unwrap();
        assert_eq!(pre, "XY");
        assert_eq!(rest, "ZW");

        let (pre, rest) = parse::take_while_not_char('X', rule).unwrap();
        assert_eq!(pre, "");
        assert_eq!(rest, "XYZW");

        let (pre, rest) = parse::take_while_not_char('W', rule).unwrap();
        assert_eq!(pre, "XYZ");
        assert_eq!(rest, "W");

        let e = parse::take_while_not_char('Q', rule);
        assert_eq!(e, Err(parse::ParseError::GeneralError));
    }

    #[test]
    fn tokenize_arrow() {
        let rule = " => XXX";
        let (tag, rest) = parse::parse_tag("=>", rule).unwrap();
        assert_eq!(tag, "=>");
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn tokenize_string() {
        let rule = r#" "a string" XXX"#;
        let (s, rest) = parse::parse_string(rule).unwrap();
        assert_eq!(s, "a string");
        assert_eq!(rest, "XXX");
    }

    #[test]
    fn parse_alt() {
        let parser = |w| {
            parse::parse_alt(
                w,
                |v| parse::parse_tag("tagA", v),
                |v| parse::parse_tag("tagB", v),
            )
        };

        assert_eq!(parser("tagA XXX"), Ok(("tagA", "XXX")));
        assert_eq!(parser("tagB XXX"), Ok(("tagB", "XXX")));
        assert_eq!(parser("tagQ XXX"), Err(parse::ParseError::GeneralError));
    }

    #[test]
    fn parse_symbol_or_literal() {
        assert_eq!(
            parse::parse_symbol_or_literal("tagA XXX"),
            Ok((raw::SymbolOrLiteral::symbol("tagA"), "XXX"))
        );
        assert_eq!(
            parse::parse_symbol_or_literal(r#" "a string"  XXX"#),
            Ok((raw::SymbolOrLiteral::literal("a string"), "XXX"))
        );
    }

    #[test]
    fn parse_symbol_or_literal_list() {
        assert_eq!(
            parse::parse_symbol_or_literal_list("tagA XXX"),
            Ok((
                vec![
                    raw::SymbolOrLiteral::symbol("tagA"),
                    raw::SymbolOrLiteral::symbol("XXX"),
                ],
                ""
            ))
        );

        assert_eq!(
            parse::parse_symbol_or_literal_list(r#" "a string"  XXX "hello" "#),
            Ok((
                vec![
                    raw::SymbolOrLiteral::literal("a string"),
                    raw::SymbolOrLiteral::symbol("XXX"),
                    raw::SymbolOrLiteral::literal("hello"),
                ],
                ""
            ))
        );
    }

    #[test]
    fn parse_symbol_fails_on_alternation() {
        let e = parse::parse_symbol("|");
        assert_eq!(
            e.map_err(|ee| ee.description()),
            Err("missing symbol".to_string())
        );
    }

    #[test]
    fn parse_symbol_with_underscore() {
        let e = parse::parse_symbol("a_b def").unwrap();
        assert_eq!(e, (raw::Symbol("a_b".to_string()), "def"));
    }

    #[test]
    fn parse_symbol_simple() {
        assert_eq!(
            parse::parse_symbol("XXX"),
            Ok((raw::Symbol::new("XXX"), ""))
        );
    }

    #[test]
    fn parse_symbol_or_literal_simple() {
        assert_eq!(
            parse::parse_symbol_or_literal("XXX"),
            Ok((raw::SymbolOrLiteral::symbol("XXX"), ""))
        );
    }

    #[test]
    fn parse_symbol_ends_at_bracket() {
        let e = parse::parse_symbol("abc(def").unwrap();
        assert_eq!(e, (raw::Symbol("abc".to_string()), "(def"));
    }

    #[test]
    fn test_parse_directive() {
        let directive_raw = r#"@import_list("city_types.txt" settlement_type)"#;
        let (directive, rest) = parse::parse_directive(directive_raw).unwrap();
        assert_eq!(
            parse::Directive {
                name: "import_list".to_string(),
                arguments: vec![
                    raw::SymbolOrLiteral::literal("city_types.txt"),
                    raw::SymbolOrLiteral::symbol("settlement_type")
                ]
            },
            directive
        );
    }

    #[test]
    fn test_parse_empty_directive() {
        let directive_raw = r#"@import_list()"#;
        let (directive, rest) = parse::parse_directive(directive_raw).unwrap();
        assert_eq!("", rest);
        assert_eq!(
            parse::Directive {
                name: "import_list".to_string(),
                arguments: vec![]
            },
            directive
        );
    }
}
