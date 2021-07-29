use std::path::PathBuf;

use structopt::StructOpt;

use crate::utils::{load_generator, save_generator};

#[derive(Debug, StructOpt)]
pub struct BiasCommand {
    /// Generator file to use
    #[structopt(parse(from_os_str))]
    generator: PathBuf,

    // Power to use
    #[structopt(long)]
    power: f32,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,
}

impl BiasCommand {
    pub fn run(&self) {
        let generator = load_generator(&self.generator);
        let generator = crate::modify::bias::bias(generator, self.power);
        save_generator(&generator, &self.output);
    }
}
