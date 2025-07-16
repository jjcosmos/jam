mod byte_loader;
mod gl_bindings;
mod jam_types;
mod window;

use wasmer::{Function, FunctionEnv, FunctionEnvMut, Instance, Memory, Module, Store, imports};

use crate::jam_types::load_string_nt;

fn main() -> anyhow::Result<()> {
    return Ok(run_wasm()?);
}

fn run_wasm() -> anyhow::Result<()> {
    println!("Starting Wasm Runner..");

    let mut path_to_wasm = std::env::current_exe()?;
    path_to_wasm.pop();
    path_to_wasm.push("main.wasm");
    let mut store = Store::default();
    let env = FunctionEnv::new(&mut store, Env { memory: None });
    let bytes = std::fs::read(path_to_wasm).expect("could not find main.wasm");
    let module = Module::new(&store, &bytes)?;

    let import_object = imports! {"env" => {
            // main rt
            "println" => Function::new_typed_with_env(&mut store, &env, make_println),
            "println_safe" => Function::new_typed_with_env(&mut store, &env, make_print_safe),
            "print_i32" => Function::new_typed_with_env(&mut store, &env, make_print_i32),
            "print_f32" => Function::new_typed_with_env(&mut store, &env, make_print_f32),
            "print_f64" => Function::new_typed_with_env(&mut store, &env, make_print_f64),
            "byteview" => Function::new_typed_with_env(&mut store, &env, make_byte_view),
            "sin_f32" => Function::new_typed_with_env(&mut store, &env, make_sin_f32),
            "sin_f64" => Function::new_typed_with_env(&mut store, &env, make_sin_f64),
            "cos_f32" => Function::new_typed_with_env(&mut store, &env, make_cos_f32),
            "cos_f64" => Function::new_typed_with_env(&mut store, &env, make_cos_f64),

            // fs
            "start_file_load" => Function::new_typed_with_env(&mut store, &env, gl_bindings::start_file_load),
            "get_file_load_status" => Function::new_typed_with_env(&mut store, &env, gl_bindings::get_file_load_status),

            // renderer functions
            "begin_default_pass" => Function::new_typed_with_env(&mut store, &env, gl_bindings::begin_default_pass),
            "end_render_pass" => Function::new_typed_with_env(&mut store, &env, gl_bindings::end_render_pass),
            "commit_frame" => Function::new_typed_with_env(&mut store, &env, gl_bindings::commit_frame),
            "apply_pipeline" => Function::new_typed_with_env(&mut store, &env, gl_bindings::apply_pipeline),
            "apply_bindings" => Function::new_typed_with_env(&mut store, &env, gl_bindings::apply_bindings),
            "apply_uniforms_from_bytes" => Function::new_typed_with_env(&mut store, &env, gl_bindings::apply_uniforms_from_bytes),
            "draw" => Function::new_typed_with_env(&mut store, &env, gl_bindings::draw),
            "new_buffer" => Function::new_typed_with_env(&mut store, &env, gl_bindings::new_buffer),
            "new_bindings" => Function::new_typed_with_env(&mut store, &env, gl_bindings::new_bindings),
            "create_texture_from_file" => Function::new_typed_with_env(&mut store, &env, gl_bindings::create_texture_from_file),
            "create_texture_from_rgba8" => Function::new_typed_with_env(&mut store, &env, gl_bindings::create_texture_from_rgba8),
            "create_texture_from_data_and_format" => Function::new_typed_with_env(&mut store, &env, gl_bindings::create_texture_from_data_and_format),
            "new_shader" => Function::new_typed_with_env(&mut store, &env, gl_bindings::new_shader),
            "new_pipeline" => Function::new_typed_with_env(&mut store, &env, gl_bindings::new_pipeline),
        },
    };

    let instance = Instance::new(&mut store, &module, &import_object)?;

    let memory = instance.exports.get_memory("memory")?;
    env.as_mut(&mut store).memory = Some(memory.clone());

    if let Ok(_window_main) = instance.exports.get_function("window_main") {
        window::start_window_loop(instance, store)
    } else {
        let main = instance.exports.get_function("main")?;
        let result = main.call(&mut store, &[])?;

        println!("program exited with result {:?}", result);

        Ok(())
    }
}

struct Env {
    memory: Option<Memory>,
}

/// Print starting at a pointer to a cstring, up until the starting ptr + len
fn make_print_safe(mut env: FunctionEnvMut<Env>, ptr: i32, len: i32) {
    let (env_data, store) = env.data_and_store_mut();
    let mem = env_data.memory.as_ref().unwrap();

    let mut buf = vec![0; len as usize];
    mem.view(&store)
        .read(ptr as u64, &mut buf)
        .expect("memory read error during print");

    let string = String::from_utf8_lossy(&buf);
    println!("{}", string);
}

fn make_print_i32(_env: FunctionEnvMut<Env>, int: i32) {
    println!("{}", int)
}

fn make_print_f32(_env: FunctionEnvMut<Env>, float: f32) {
    println!("{}", float)
}

fn make_print_f64(_env: FunctionEnvMut<Env>, double: f64) {
    println!("{}", double)
}

/// Print until encountering a null byte (up to a max len)
fn make_println(mut env: FunctionEnvMut<Env>, ptr: i32) {
    let string = load_string_nt(&mut env, ptr);
    println!("{}", string);
}

/// Print the memory at [ptr .. ptr + bytecount]
fn make_byte_view(mut env: FunctionEnvMut<Env>, ptr: i32, byte_count: i32) {
    let (env_data, store) = env.data_and_store_mut();
    let mem = env_data.memory.as_ref().unwrap();

    let mut buf = vec![0; byte_count as usize];
    mem.view(&store)
        .read(ptr as u64, &mut buf)
        .expect("memory read error during byteview");

    let text = format!("{:#04X?}", buf);
    println!("{}", text.replace("\n", "").replace("    ", " "));
}

fn make_sin_f32(_env: FunctionEnvMut<Env>, theta: f32) -> f32 {
    theta.sin()
}

fn make_sin_f64(_env: FunctionEnvMut<Env>, theta: f64) -> f64 {
    theta.sin()
}

fn make_cos_f32(_env: FunctionEnvMut<Env>, theta: f32) -> f32 {
    theta.cos()
}

fn make_cos_f64(_env: FunctionEnvMut<Env>, theta: f64) -> f64 {
    theta.cos()
}
