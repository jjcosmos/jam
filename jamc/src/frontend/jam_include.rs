use std::{
    collections::HashSet,
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use crate::frontend::ast::{IncludeStmt, Item};
use crate::frontend::parser::Parser;
use crate::{SourceFile, report_parse_error};

pub const JAM_INSTALL_DIR_KEY: &'static str = "JAM_INSTALL_DIR";

// rewrite to handle path based resolution
pub fn include_includes2(
    source_files: &mut Vec<SourceFile>,
    includes: &HashSet<IncludeStmt>,
    local_inc_dir: &PathBuf,
    seen_paths: &mut HashSet<PathBuf>,
    next_id: &mut usize,
) -> bool {
    for include in includes {
        let raw_path = include.mod_name.text.replace("\"", ""); // strip quotes if any
        let relative_path = Path::new(&raw_path);

        let import_roots = [
            env::var(JAM_INSTALL_DIR_KEY)
                .map(|dir| Path::new(&dir).join("import"))
                .unwrap_or_else(|_| local_inc_dir.clone()),
            local_inc_dir.clone(),
        ];

        let mut found = false;

        for base in &import_roots {
            let full_path = base.join(relative_path);

            if full_path.extension() != Some(OsStr::new("jam")) {
                continue;
            }

            let Ok(canonical) = fs::canonicalize(&full_path) else {
                continue;
            };

            if !seen_paths.insert(canonical.clone()) {
                // Already included
                found = true;
                break;
            }

            match fs::read_to_string(&canonical) {
                Ok(source) => match Parser::parse_program(&source, *next_id) {
                    Ok(program) => {
                        let nested_includes: HashSet<IncludeStmt> = program
                            .items
                            .iter()
                            .filter_map(|item| {
                                if let Item::IncludeStmt(stmt) = item {
                                    Some(stmt.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        source_files.push(SourceFile {
                            id: *next_id,
                            module_name: canonical
                                .file_name()
                                .expect("failed to get filename during include parsing")
                                .to_string_lossy()
                                .to_string(), // This should just be the file name
                            module_path: canonical,
                            program,
                            source,
                        });

                        *next_id += 1;

                        // Recurse
                        if !include_includes2(
                            source_files,
                            &nested_includes,
                            local_inc_dir,
                            seen_paths,
                            next_id,
                        ) {
                            return false;
                        }

                        found = true;
                        break;
                    }
                    Err(e) => {
                        report_parse_error(&e, &source, &raw_path);
                        return false;
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read include '{}': {}", full_path.display(), e);
                    return false;
                }
            }
        }

        if !found {
            eprintln!("[jamc ERROR] Could not resolve include '{}'", raw_path);
            return false;
        }
    }

    true
}
