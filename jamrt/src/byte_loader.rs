#![allow(unused)]

// stolen from my web game engine, Talos

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct ByteLoader {
    load_handles: HashMap<String, Arc<Mutex<LoadHandle>>>,
}

impl ByteLoader {
    pub fn new() -> Self {
        Self {
            load_handles: HashMap::new(),
        }
    }

    pub fn request_load(&mut self, path: &str) -> Arc<Mutex<LoadHandle>> {
        if let Some(existing) = self.try_get_handle(path) {
            return existing;
        }

        let new_handle = LoadHandle { result: None };

        let handle_mutex = Arc::new(Mutex::new(new_handle));
        self.load_handles
            .insert(path.to_string(), handle_mutex.clone());

        #[cfg(target_arch = "wasm32")]
        {
            wasm_kick_off_file_load(path.to_string(), rc_handle.clone());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            kick_off_file_load(path.to_string(), handle_mutex.clone());
        }

        handle_mutex.clone()
    }

    pub fn try_get_handle(&self, path: &str) -> Option<Arc<Mutex<LoadHandle>>> {
        self.load_handles.get(path).cloned()
    }

    /// Remove reference from byte_loader instance. Memory will not be freed until 0 Arc<Mutex<LoadHandle>>'s of this resource remain.
    pub fn try_obtain(&mut self, path: &str) -> Option<Arc<Mutex<LoadHandle>>> {
        if let Some(h) = self.load_handles.remove(path) {
            return Some(h);
        }

        return None;
    }

    /// Removes all refs from the byte loader instance.
    /// However, if there are any refs floating around in components, the memory won't be freed.
    /// Set shrink to true to fully free the memory used to hold the handles, with the cost being slower allocations later.
    /// The keys are just Strings and the handles are Arcs, so it is usually best not to shrink.
    pub fn release_all(&mut self, shrink: bool) {
        self.load_handles.clear();

        if shrink {
            self.load_handles.shrink_to_fit();
        }
    }
}

pub struct LoadHandle {
    pub result: Option<Vec<u8>>,
}

pub fn kick_off_file_load(path: String, load_handle: Arc<Mutex<LoadHandle>>) {
    let handle_arc = load_handle.clone();

    let _handle = std::thread::spawn(move || match smol::block_on(read_file_to_bytes(&path)) {
        Ok(content) => {
            let mut locked = handle_arc.lock().unwrap();
            locked.result = Some(content);
        }
        Err(e) => eprintln!("Failed to read {}: {}", path, e),
    });
}

// Removed res so that assets or res can be read. Needs to be specified now
async fn read_file_to_bytes(file_path: &str) -> std::io::Result<Vec<u8>> {
    let contents = async_fs::read(file_path).await?;
    println!("[JAMRT] completed load for file: {:?}", file_path);
    Ok(contents)
}

#[cfg(target_arch = "wasm32")]
pub fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    let mut pathname = location.pathname().unwrap();
    pathname = pathname.replace("/index.html", "");

    let res_path = format!("{}{}", origin, pathname);

    let base = reqwest::Url::parse(&format!("{}/", res_path,)).unwrap();
    base.join(file_name).unwrap()
}

#[allow(unused)]
pub fn resource_dir() -> std::path::PathBuf {
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe directory");
    let res_path_buf = exe_dir.join("res"); // Assuming "res" is located alongside the executable
    return res_path_buf;
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;
#[cfg(target_arch = "wasm32")]
use web_sys::Response;
#[cfg(target_arch = "wasm32")]
use web_sys::js_sys::Uint8Array;

#[cfg(target_arch = "wasm32")]
pub fn wasm_kick_off_file_load(path: String, load_handle: Arc<Mutex<LoadHandle>>) {
    // Spawning the async task to load the file

    let path = format_url(path.as_str());
    let clone = load_handle.clone();

    spawn_local(async move {
        match wasm_async_load_file(path.to_owned()).await {
            Ok(contents) => {
                let mut locked = load_handle.lock().unwrap();
                println!("[JAMRT] completed load for file: {:?}", path);
                locked.result = Some(contents);
            }
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to load file: {:?}", e).into());
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
async fn wasm_async_load_file(url: reqwest::Url) -> Result<Vec<u8>, JsValue> {
    let response = wasm_bindgen_futures::JsFuture::from(
        web_sys::window().unwrap().fetch_with_str(url.as_str()),
    )
    .await?
    .dyn_into::<Response>()?;

    if !response.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP error: {}",
            response.status()
        )));
    }

    let buf_promise = response.array_buffer()?;
    let buf = wasm_bindgen_futures::JsFuture::from(buf_promise).await?;

    let uint8_array = Uint8Array::new(&buf);
    let bytes = uint8_array.to_vec();
    Ok(bytes)
}

pub async fn load_file_async(path: &str) -> std::io::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        let path = format_url(path.as_str());
        wasm_load_file(path).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        read_file_to_bytes(path).await
    }
}
