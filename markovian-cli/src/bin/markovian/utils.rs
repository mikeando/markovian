use std::io::BufRead;
use std::path::PathBuf;

pub fn read_input_lines<F>(input_files: &[PathBuf], f: F) -> Vec<String>
where
    F: Fn(String) -> String,
{
    let stdin_flag = PathBuf::from("-");
    let mut extra_lines: Vec<String> = if input_files.contains(&stdin_flag) {
        let stdin = std::io::stdin();
        let handle = stdin.lock();
        let lines: Vec<String> = handle
            .lines()
            .map(|n| n.unwrap().trim().to_string())
            .map(|s| f(s)) // TODO: Extract and make configurable
            .filter(|s| s.len() >= 3)
            .collect();
        lines
    } else {
        vec![]
    };

    // Load the text
    let mut input_tokens: Vec<String> = input_files
        .iter()
        .filter(|&p| *p != stdin_flag)
        .map(|n| {
            let v: Vec<_> = std::fs::read_to_string(n)
                .unwrap()
                .lines()
                .map(|n| n.trim().to_string())
                .map(|s| f(s)) // TODO: Extract and make configurable
                .filter(|s| s.len() >= 3)
                .collect();
            v
        })
        .flatten()
        .collect();
    input_tokens.append(&mut extra_lines);
    drop(extra_lines);
    input_tokens
}
