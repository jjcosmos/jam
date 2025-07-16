# Jam lang

A C-like experimental language and toolchain built with Rust and WASM.

---

##  Features

- AOT Compilation
- Runtime with builtin open GL bindings (WIP)
- Cargo-style CLI with `jammy`

---

##  Building (Windows)

###  Requirements

- [rust](https://www.rust-lang.org/) 

---

### Building `jamc` and `jammy`

git clone https://github.com/jjcosmos/jamlang_all.git

```
cargo run -p xtask -- --release
```
For convenience, add the compiled binaries to your system `PATH`.

---

## Getting Started

With `jammy`, `jamc`, available in your `PATH`, you can create a new project:

```sh
jammy new my_project
```

Available commands:
- `jammy new <name>` — creates a project directory with default files
- `jammy build` — compiles the project  
- `jammy run` — builds and runs the project  
- `jammy clean` — removes build artifacts  
- `jammy --help` — full command reference 

Creating a project with jammy will set up a `jammy.toml` with some default configuration fields.

>  Currently tested only on Windows. Linker arguments are currently not used

---

## Example
```jam
extern proc println(msg: cstr);

proc main() -> i32 {
    println("Hello World!");
    return 0i32;
}
```

THIS IS OUT OF DATE AND NEEDS RE-DONE (post LLVM removal)
Browse additional examples in the [`examples/`](https://github.com/jjcosmos/jamlang_all/tree/main/examples) directory (WIP).

I would have a **TODO** section, but it would honestly just be too massive at the moment.
