use std::process::Command;
use std::io;

fn main() -> io::Result<()> {
    let output = Command::new("python3")
        .arg("./examples/py_scripts/sln.py")
        .arg(".data/train.csv")
        .arg(".data/test.csv")
        .output()?;

    println!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}
