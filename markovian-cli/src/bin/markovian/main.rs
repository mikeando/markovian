#![feature(total_cmp)]

use log::{debug, info};
use structopt::StructOpt;

pub mod generator;
pub mod simple;
pub mod symboltable;

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
    SymbolTable(symboltable::Command),
    Generator(generator::Command),
    Simple(simple::Command),
    /// Show license information about markovian and the libraries it uses
    License,
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

fn main() {
    let opt = Opt::from_args();
    debug!("{:?}", opt);

    let verbose: i32 = opt.verbose;
    setup_logging(verbose);

    info!("markovian! {:?}", opt);

    match &opt.cmd {
        Command::SymbolTable(st_cmd) => symboltable::run(st_cmd),
        Command::Generator(gen_cmd) => generator::run(gen_cmd),
        Command::Simple(cmd) => simple::run(cmd),
        Command::License => print_license_info(),
    }
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
