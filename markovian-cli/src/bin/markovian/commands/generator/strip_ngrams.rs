use std::collections::BTreeMap;
use std::path::PathBuf;

use structopt::StructOpt;

use super::wrap_load_transform_ngrams_save;

#[derive(Debug, StructOpt)]
pub struct StripNGramsCommand {
    /// input generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    /// output generator file to use
    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

pub fn command_strip_ngrams(cmd: &StripNGramsCommand) {
    wrap_load_transform_ngrams_save(&cmd.generator, &cmd.output, |_ngrams| BTreeMap::new());
}
