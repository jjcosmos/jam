use miniquad::{
    Bindings, BufferId, EventHandler, Pipeline, RenderingBackend, ShaderId, TextureId, conf, date,
    window,
};
use stable_vec::StableVec;
use std::{
    cell::RefCell,
    ptr,
    sync::{Arc, Mutex},
    time::Instant,
};
use wasmer::{Function, Instance, Store, Value};

use crate::byte_loader::{ByteLoader, LoadHandle};

#[allow(dead_code)]
pub struct Stage {
    pub(crate) instance: Instance,
    pub(crate) store: Store,
    pub(crate) ctx: Box<dyn RenderingBackend>,

    pub(crate) window_main: Option<Function>,
    pub(crate) window_draw: Option<Function>,
    pub(crate) window_update: Option<Function>,

    pub(crate) udpate_count: u64,
    pub(crate) last_update: Instant,
    pub(crate) last_delta: f64,

    pub(crate) pipelines: StableVec<Pipeline>,
    pub(crate) geo_bindings: StableVec<Bindings>,
    pub(crate) buffers: StableVec<BufferId>,
    pub(crate) textures: StableVec<TextureId>,
    pub(crate) shaders: StableVec<ShaderId>,

    pub(crate) byte_loader: ByteLoader,
    pub(crate) file_load_operations: StableVec<Arc<Mutex<LoadHandle>>>,
}

// we BALL
thread_local! {
    pub static STAGE_GLOBAL: RefCell<*mut Stage > = RefCell::new(ptr::null_mut())
}

#[macro_export]
/// Borrows the global stage state from the RefCell<*mut Stage>. Should only be used after stage initialization.
macro_rules! with_stage {
    ($stage:ident => $body:block) => {{
        STAGE_GLOBAL.with(|handle| {
            let ptr = *handle.borrow();
            assert!(!ptr.is_null(), "STAGE_GLOBAL is uninitialized");
            let $stage: &mut Stage = unsafe { &mut *ptr };
            $body
        })
    }};
}

impl Stage {
    pub fn new(instance: Instance, store: Store) -> Stage {
        let ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();
        let pipelines = StableVec::new();
        let geo_bindings = StableVec::new();
        let buffers = StableVec::new();
        let textures = StableVec::new();

        let byte_loader = ByteLoader::new();
        let file_load_operations = StableVec::new();
        let shaders = StableVec::new();

        Stage {
            window_main: instance.exports.get_function("window_main").ok().cloned(),
            window_draw: instance.exports.get_function("window_draw").ok().cloned(),
            window_update: instance.exports.get_function("window_update").ok().cloned(),
            instance,
            store,
            pipelines,
            geo_bindings,
            buffers,
            textures,
            shaders,

            byte_loader,
            file_load_operations,

            ctx,
            udpate_count: 0,
            last_update: Instant::now(),
            last_delta: 0.0016f64,
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        let frame_time = self.last_update.elapsed();
        self.last_delta = frame_time.as_secs_f64();
        self.last_update = Instant::now();

        if self.udpate_count == 0 {
            if let Some(start_fn) = &self.window_main {
                start_fn
                    .call(&mut self.store, &vec![])
                    .expect("failed to call start function");
            }
        }

        if let Some(update_fn) = &self.window_update {
            update_fn
                .call(&mut self.store, &vec![])
                .expect("failed to call function");
        }

        self.udpate_count += 1;
    }

    fn draw(&mut self) {
        if let Some(draw_fn) = &self.window_draw {
            let now = date::now();
            let delta = self.last_delta;
            draw_fn
                .call(&mut self.store, &vec![Value::F64(delta), Value::F64(now)])
                .expect("failed to call function");
        }
    }
}

pub fn start_window_loop(instance: Instance, store: Store) -> anyhow::Result<()> {
    let mut conf = conf::Conf::default();
    //conf.platform.swap_interval = Some(0);
    conf.platform.swap_interval = None;

    miniquad::start(conf, move || {
        let mut stage = Box::new(Stage::new(instance, store));

        STAGE_GLOBAL.with(|handle| {
            *handle.borrow_mut() = &mut *stage;
        });

        stage
    });

    Ok(())
}
