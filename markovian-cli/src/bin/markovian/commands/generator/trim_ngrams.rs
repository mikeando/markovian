use std::path::PathBuf;

use structopt::StructOpt;

use super::wrap_load_transform_ngrams_save;

#[derive(Debug, StructOpt)]
pub struct TrimNGramsCommand {
    /// input generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// output generator file to use
    #[structopt(parse(from_os_str))]
    output: PathBuf,

    /// ngram length to effect
    #[structopt(long, short)]
    n: usize,

    /// min weight to keep
    #[structopt(long, short)]
    w: f32,
}

pub fn command_trim_ngrams(cmd: &TrimNGramsCommand) {
    wrap_load_transform_ngrams_save(&cmd.generator, &cmd.output, |ngrams| {
        ngrams
            .into_iter()
            .filter(|(k, v)| k.len() != cmd.n || *v > cmd.w)
            .collect()
    });
}
