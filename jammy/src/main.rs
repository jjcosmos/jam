use anyhow::Ok;
use anyhow::anyhow;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    env::consts::EXE_EXTENSION,
    fs,
    path::{Path, PathBuf},
    vec,
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct JamArgs {
    #[command(subcommand)]
    cmd: Command,

    #[clap(long, short)]
    verbose: bool,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Build(Build),
    Run(Run),
    Clean(Clean),
    Import { name: String },
    New { name: String },
}

#[derive(Parser, Debug, Clone)]
struct Build {
    #[clap(long)]
    pub release: bool,

    #[clap(long, short, num_args = 0.., value_delimiter = ' ')]
    args: Vec<String>,
}

#[derive(Parser, Debug, Clone)]
struct Run {
    #[clap(long)]
    pub release: bool,

    #[clap(num_args = 0.., value_delimiter = ' ', last=true)]
    args: Vec<String>,
}

#[derive(Parser, Debug, Clone)]
struct Clean {
    #[clap(long, short)]
    configuration: Option<String>,
}

fn main() -> Result<(), anyhow::Error> {
    let args = JamArgs::parse();

    match args.cmd {
        Command::Build(build) => {
            let toml_config = get_toml()?;
            build_from_toml(build.release, &toml_config)?;
        }
        Command::Run(run) => {
            let toml_config = get_toml()?;
            build_from_toml(run.release, &toml_config)?;
            build_runner(&toml_config)?;

            let mut exe_path = std::env::current_dir()?;
            exe_path.push("out");
            exe_path.push(Path::new(&format!(
                "{}.{}",
                toml_config.package_name, EXE_EXTENSION
            )));

            assert!(exe_path.is_file(), "exe path {:?} is invalid", exe_path);

            let status = std::process::Command::new(exe_path)
                .args(run.args)
                .status()?;
            if !status.success() {
                panic!("process failed with exit code: {:?}", status.code());
            }
        }
        Command::Clean(_clean) => {
            // Just doing this to ensure we are in a jam dir
            let _toml_config = get_toml()?;
            let current_dir = std::env::current_dir()?;
            let mut assert_jamfile = current_dir.clone();
            assert_jamfile.push("jammy.toml");
            assert!(assert_jamfile.exists());

            let mut out_dir = current_dir.clone();
            out_dir.push("out");

            let mut temp_dir = current_dir.clone();
            temp_dir.push("temp");

            clean_dir(&out_dir)?;
            clean_dir(&temp_dir)?;
        }
        Command::New { name } => {
            let current_dir = std::env::current_dir()?;
            let base_dir = current_dir.join(&name);
            fs::create_dir(&base_dir)?;
            fs::create_dir(base_dir.join(Path::new("include")))?;
            fs::create_dir(base_dir.join(Path::new("lib")))?;
            fs::create_dir(base_dir.join(Path::new("src")))?;

            // TODO: ??
            #[cfg(target_os = "windows")]
            let linker_args = vec![
                "-lvcruntime",
                "-lmsvcrt",
                "-llegacy_stdio_definitions",
                "-Wl,/subsystem:console",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();

            #[cfg(target_os = "linux")]
            let linker_flags = vec![
                // maybe "-lc"
            ];

            #[cfg(target_os = "macos")]
            let linker_flags = vec![
                // maybe "-lSystem"
            ];

            let toml = JammyToml {
                package_name: name.clone(),
                lib_dir: "lib".to_string(),
                include_dir: "include".to_string(),
                linker_args: Some(linker_args),
                target_triple: "x86_64-pc-windows-msvc".to_string(),
            };

            fs::write(base_dir.join("jammy.toml"), toml::to_string(&toml)?)?;
            fs::write(
                base_dir.join("src/main.jam"),
                include_str!("../default.jam"),
            )?;
            fs::write(
                base_dir.join(".gitignore"),
                include_str!("../default_gitignore.txt"),
            )?;

            println!("created project \"{}\"", &name);
        }
        Command::Import { name } => {
            let toml_config = get_toml()?;

            let status = std::process::Command::new("jamc")
                .arg("import")
                .arg("--inc-dir")
                .arg(toml_config.include_dir)
                .arg("--name")
                .arg(name)
                .status()?;
            if !status.success() {
                panic!("failed to run jamc with args");
            }
        }
    }

    Ok(())
}

fn get_toml() -> anyhow::Result<JammyToml> {
    let mut jammy_toml = std::env::current_dir()?;
    jammy_toml.push(Path::new("jammy.toml"));
    let toml_str = std::fs::read_to_string(jammy_toml)?;
    let toml_config: JammyToml = toml::from_str(&toml_str)?;
    Ok(toml_config)
}

#[derive(Debug, Deserialize, Serialize)]
struct JammyToml {
    package_name: String,
    #[serde(default)]
    lib_dir: String,
    #[serde(default)]
    include_dir: String,
    #[serde(default)]
    linker_args: Option<Vec<String>>,
    // Should be wasm jamrt, wasm raw, c2native
    #[serde(default)]
    target_triple: String,
}

fn clean_dir(dir: &PathBuf) -> anyhow::Result<()> {
    if dir.exists() {
        fs::remove_dir_all(&dir)?
    }

    Ok(())
}

fn build_from_toml(release: bool, toml_config: &JammyToml) -> anyhow::Result<()> {
    let mut src_paths = vec![];

    let src_dir = std::env::current_dir()?.join("src");
    for info in WalkDir::new(src_dir) {
        let info = info?;
        if let Some(ext) = info.path().extension() {
            if ext == "jam" {
                src_paths.push(info.path().to_owned());
            }
        }
    }

    let mut command = std::process::Command::new("jamc");
    command
        .arg("compile")
        .arg("--files")
        .args(&src_paths)
        .arg("--lib-dir")
        .arg(&toml_config.lib_dir) //todo: multiple lib dirs
        .arg("--inc-dir")
        .arg(&toml_config.include_dir)
        .arg("--output-name")
        .arg(&toml_config.package_name)
        .arg("--target-triple")
        .arg(&toml_config.target_triple);

    if !release {
        command.arg("--debug");
    }

    command
        .arg("--")
        .args(&toml_config.linker_args.clone().unwrap_or_default());

    let status = command.status()?;

    if !status.success() {
        println!();
        return Err(anyhow!(format!(
            "failed to compile module {}",
            &toml_config.package_name
        )));
    }

    Ok(())
}

fn build_runner(toml_config: &JammyToml) -> anyhow::Result<()> {
    const JAM_INSTALL_DIR_KEY: &'static str = "JAM_INSTALL_DIR";
    let jam_install_dir = std::env::var(JAM_INSTALL_DIR_KEY)
        .map(PathBuf::from)
        .map_err(|_| "JAM_INSTALL_DIR not set")
        .expect("failed to get jam install dir");

    let source_runner_path = {
        let mut path = jam_install_dir.join("bin");
        path.push(if cfg!(target_os = "windows") {
            "jamrt.exe"
        } else {
            "jamrt"
        });
        path
    };

    let current_dir = std::env::current_dir()?;
    let dest_runner_path = {
        let mut path = current_dir.join("out");

        path.push(if cfg!(target_os = "windows") {
            format!("{}.exe", toml_config.package_name)
        } else {
            toml_config.package_name.clone()
        });
        path
    };

    //println!("copying runner {:?} to {:?}", source_runner_path, dest_runner_path);
    std::fs::copy(source_runner_path, dest_runner_path)?;

    Ok(())
}
