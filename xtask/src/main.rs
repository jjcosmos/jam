//cargo run -p xtask -- build-all
use clap::Parser;
use std::{fs, path::Path, process::Command};

#[derive(Parser)]
#[command(name = "xtask", about = "Workspace build helper")]
struct Cli {
    /// Build in release mode
    #[arg(long)]
    release: bool,
}

fn main() {
    let cli = Cli::parse();

    let build_mode = if cli.release { "release" } else { "debug" };

    let mut cmd = Command::new("cargo");
    cmd.args(&["build", "--workspace", "--exclude", "xtask"]);
    if cli.release {
        cmd.arg("--release");
    }

    let status = cmd.status().expect("Failed to run cargo build");
    if !status.success() {
        panic!("Build failed");
    }

    let dest_dir = Path::new("jam_install/bin");
    fs::create_dir_all(dest_dir).unwrap();

    let bin_pattern = format!("target/{build_mode}/*.exe");
    let entries = glob::glob(&bin_pattern).unwrap();

    for entry in entries.flatten() {
        if entry.is_file() && is_executable(&entry) {
            let name = entry.file_name().unwrap();
            let dest = dest_dir.join(name);
            fs::copy(&entry, &dest).expect("Failed to copy binary");
            println!("Copied {} -> {}", entry.display(), dest.display());
        } else {
            println!("skipping entry {:?}", entry);
        }
    }

    println!("finished xtask")
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_executable(path: &Path) -> bool {
    path.extension().map(|ext| ext == "exe").unwrap_or(false)
}
