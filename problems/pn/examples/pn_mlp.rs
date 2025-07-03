mod utils;

use std::io;

use utils::run_network;

fn main() -> io::Result<()> {
    run_network("mlp")
}
