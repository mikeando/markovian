#![feature(total_cmp)]

use commands::generator;
use log::{debug, info};
use markovian_core::language::compiled::{ConversionError, ExpansionError};
use markovian_core::language::context::{ContextError, GrammarLoaderContext};
use snafu::{OptionExt, ResultExt, Snafu};
use structopt::StructOpt;

pub mod utils;

pub mod commands;
pub mod modify;

use std::path::PathBuf;

#[derive(Debug, StructOpt)]
#[structopt(name = "markovian", about = "Markov based name generator.")]
struct Opt {
    /// Verbosity level
    #[structopt(short, long, default_value = "0")]
    verbose: i32,

    #[structopt(subcommand)] // Note that we mark a field as a subcommand
    cmd: Command,
}

#[derive(Debug, StructOpt)]
enum Command {
    Generator(generator::Command),

    /// Show license information about markovian and the libraries it uses
    License,

    /// Work with grammar files
    Grammar(GrammarOpt),
}

fn setup_logging(verbose: i32) {
    let level = match verbose {
        v if v <= 0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        _ => log::LevelFilter::Trace,
    };
    // With fern, we can:

    // Configure logger at runtime
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                //"{}[{}][{}] {}",
                "[{}][{}] {}",
                //chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        // Add blanket level filter -
        .level(level)
        // Output to stdout, files, and other Dispatch configurations
        .chain(std::io::stdout())
        //.chain(fern::log_file("output.log").unwrap())
        // Apply globally
        .apply()
        .unwrap();
}

#[derive(Debug, StructOpt)]
struct GrammarOpt {
    /// directory to load additional rules and word lists from.
    #[structopt(long, default_value = ".")]
    library_directory: PathBuf,

    /// root list of rules to load.
    #[structopt(long)]
    rules_file: String,

    /// root symbol for productions.
    #[structopt(long)]
    start_token: String,

    /// number of results to output
    #[structopt(long, default_value = "1")]
    count: i32,
}

#[derive(Debug, Snafu)]
enum GrammarError {
    #[snafu(display("IO error: {}", source))]
    IO { source: std::io::Error },

    #[snafu(display("conversion error: {:?}", source))]
    Conversion { source: ConversionError },

    #[snafu(display("expansion error: {:?}", source))]
    Expansion { source: ExpansionError },

    #[snafu(display("invalid token '{}'", token))]
    InvalidToken { token: String },

    #[snafu(display("language error: {}", mesg))]
    Language { mesg: String },

    #[snafu(display("context error: {}", source))]
    Context { source: ContextError },
}

fn grammar(opt: &GrammarOpt) -> Result<(), GrammarError> {
    use markovian_core::language::raw::Context;
    let mut rng = rand::thread_rng();
    let mut ctx = GrammarLoaderContext::new(opt.library_directory.clone());

    let language = ctx.get_language(&opt.rules_file).context(Context)?;
    let language =
        markovian_core::language::compiled::Language::from_raw(&language).context(Conversion)?;
    let token = language
        .token_by_name(opt.start_token.clone())
        .with_context(|| InvalidToken {
            token: opt.start_token.clone(),
        })?;
    for _ in 0..opt.count {
        let s = language.expand(&[token], &mut rng).context(Expansion)?;
        println!("{}", s);
    }
    Ok(())
}

#[derive(Debug)]
enum ApplicationError {
    GrammarError(GrammarError),
}

impl From<GrammarError> for ApplicationError {
    fn from(e: GrammarError) -> Self {
        ApplicationError::GrammarError(e)
    }
}

fn grammar_wrapper(opt: &GrammarOpt) {
    let e = grammar(opt);
    match e {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error running grammar");
            let mut ee: Option<&dyn std::error::Error> = Some(&e);
            while let Some(e) = ee {
                let mesg = e.to_string();
                if mesg != "context error" && mesg != "invalid directive" {
                    eprintln!("    {}", mesg);
                }
                ee = e.source();
            }
        }
    }
}

fn main() {
    let opt = Opt::from_args();
    debug!("{:?}", opt);

    let verbose: i32 = opt.verbose;
    setup_logging(verbose);

    info!("markovian! {:?}", opt);

    match &opt.cmd {
        Command::Generator(gen_cmd) => generator::run(gen_cmd),
        Command::License => print_license_info(),
        Command::Grammar(cmd) => grammar_wrapper(cmd),
    };
}

fn print_license_info() {
    let self_license = include_str!("../../../../LICENSE-MIT");
    let third = include_str!("../../../../thirdparty.md");
    println!();
    println!("----------------------------");
    println!();
    println!("markovian is licensed under the MIT licese");
    println!();
    println!("{}", self_license);
    println!();
    println!("----------------------------");
    println!();
    println!("markovian also contains code from many third party sources");
    println!();
    println!("{}", third);
    println!();
    println!("----------------------------");
    println!();
}
