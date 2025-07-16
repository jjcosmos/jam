use crate::jam_types::DummyBufferLayout;
use crate::jam_types::DummyVertexAttribute;
use crate::jam_types::UnitOnlyEnum;
use crate::jam_types::read_list_ptrs;
use crate::window::STAGE_GLOBAL;
use crate::window::Stage;
use image::{GenericImageView, ImageReader};
use miniquad::BufferLayout;
use miniquad::PipelineParams;
use miniquad::ShaderMeta;
use miniquad::VertexAttribute;
use miniquad::VertexFormat;
use miniquad::VertexStep;
use miniquad::{Bindings, BufferSource, BufferType, BufferUsage, TextureParams};
use wasmer::FunctionEnvMut;

use crate::{
    Env,
    jam_types::{FromWasmMem, load_string_nt, read_mem_slice},
    with_stage,
};

/// Returns the index into the vec of shader Ids
pub fn new_shader(
    mut env: FunctionEnvMut<Env>,
    vertex_str_ptr: i32,
    fragment_str_ptr: i32,
    shader_meta_ptr: i32,
) -> u32 {
    let vertex_str = load_string_nt(&mut env, vertex_str_ptr);
    let fragment_str = load_string_nt(&mut env, fragment_str_ptr);

    let shader_meta = ShaderMeta::from_wasm_mem(&mut env, shader_meta_ptr);

    with_stage!(stage => {
        let shader = stage.ctx.new_shader(miniquad::ShaderSource::Glsl { vertex: &vertex_str, fragment:& fragment_str }, shader_meta).unwrap();
        stage.shaders.push(shader) as u32
    })
}

pub fn new_pipeline(
    mut env: FunctionEnvMut<Env>,
    buffer_layout_ptr: i32,
    buffer_layout_count: i32,
    attributes_ptr: i32,
    attributes_count: i32,
    shader_id: i32,
    pipeline_params_ptr: i32,
) -> u32 {
    //println!("Read ptrs dummybufferlayout");
    let dummy_layout_list: Vec<DummyBufferLayout> =
        read_list_ptrs(&mut env, buffer_layout_ptr, buffer_layout_count as usize);

    let dummy_attr_list: Vec<DummyVertexAttribute> =
        read_list_ptrs(&mut env, attributes_ptr, attributes_count as usize);

    let layout_list: Vec<BufferLayout> = dummy_layout_list
        .iter()
        .map(|dummy| BufferLayout {
            stride: dummy.stride,
            step_func: VertexStep::from_i32(dummy.step_func),
            step_rate: dummy.step_rate,
        })
        .collect();

    let attr_list: Vec<VertexAttribute> = dummy_attr_list
        .iter()
        .map(|dummy| VertexAttribute {
            name: Box::leak(load_string_nt(&mut env, dummy.name).into_boxed_str()),
            format: VertexFormat::from_i32(dummy.format),
            buffer_index: dummy.buffer_index as usize,
            gl_pass_as_float: if dummy.gl_pass_as_float == 0 {
                false
            } else {
                true
            },
        })
        .collect();

    let params = PipelineParams::from_wasm_mem(&mut env, pipeline_params_ptr);

    with_stage!(stage => {
        let pipeline = stage.ctx.new_pipeline(&layout_list, &attr_list, *stage.shaders.get(shader_id as usize).expect("invalid shader id"), params);
        stage.pipelines.push(pipeline) as u32
    })
}

/// Monster of a signature lol
pub fn new_bindings(
    mut env: FunctionEnvMut<Env>,
    vertex_buffer_id_list_ptr: i32,
    vertex_buffer_id_count: i32,
    index_buffer_id: i32,
    image_id_list_ptr: i32,
    image_id_count: i32,
) -> u32 {
    // buffer of size id count * 4 (for the 32 bit int)
    let mut buf_vertex_buf_ids = vec![0; vertex_buffer_id_count as usize * 4];
    read_mem_slice(&mut env, vertex_buffer_id_list_ptr, &mut buf_vertex_buf_ids);
    let vertex_buf_ids: Vec<i32> = buf_vertex_buf_ids
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // basically same thing here
    let mut buf_image_ids = vec![0; image_id_count as usize * 4];
    read_mem_slice(&mut env, image_id_list_ptr, &mut buf_image_ids);
    let image_ids: Vec<i32> = buf_image_ids
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    with_stage!(stage => {
        let bindings = Bindings {
            vertex_buffers: vertex_buf_ids.iter().map(|id| *stage.buffers.get(*id as usize).expect("invalid vertex buffer id")).collect(),
            index_buffer: *stage.buffers.get(index_buffer_id as usize).expect("invalid index buffer id"),
            images: image_ids.iter().map(|id| *stage.textures.get(*id as usize).expect("invalid image id")).collect(),
        };

        stage.geo_bindings.push(bindings) as u32
    })
}

// Create a new buffer and return the index into the saved buffers
// TODO: This should take element size & count
pub fn new_buffer(
    mut env: FunctionEnvMut<Env>,
    buffer_type: i32,
    buffer_usage: i32,
    data_ptr: i32,
    size: i32,
    element_size: i32,
) -> u32 {
    with_stage!(stage => {
        let mut buf = vec![0; size as usize];
        read_mem_slice(&mut env, data_ptr, &mut buf);

        let data_ptr = buf.as_ptr() as *const u8;
        // Can't just use the slice, because that erases the element size
        let buffer_source = unsafe {BufferSource::pointer(data_ptr, size as usize, element_size as usize)};

        let buffer_type = match buffer_type {
            0 => BufferType::VertexBuffer,
            1 => BufferType::IndexBuffer,
            _ => {panic!("invalid buffer type {}", buffer_type)}
        };

        let buffer_usage = match buffer_usage {
            0 => BufferUsage::Immutable,
            1 => BufferUsage::Dynamic,
            2 => BufferUsage::Stream,
            _ => {panic!("invalid buffer usage {}", buffer_usage)}
        };

        let buffer_id = stage.ctx.new_buffer(buffer_type, buffer_usage, buffer_source);
        let map_id =
        stage.buffers.push(buffer_id);

        map_id as u32
    })
}

pub fn start_file_load(mut env: FunctionEnvMut<Env>, path_string_ptr: i32) -> u32 {
    let path = load_string_nt(&mut env, path_string_ptr);

    with_stage!(stage => {
        let handle = stage.byte_loader.request_load(&path);
        let idx = stage.file_load_operations.push(handle) as u32;
        println!("[JAMRT] loading file: {:?}", path);
        idx
    })
}

pub fn get_file_load_status(_env: FunctionEnvMut<Env>, file_id: i32) -> i32 {
    with_stage!(stage => {
        if file_id as u32 != u32::MAX && let Some(handle) = stage.file_load_operations.get(file_id as usize) {
            // This doesn't block
            if let Ok(locked) = handle.try_lock() {
                return if locked.result.is_some() {1} else {0};
            }
            else{
                return 0;
            }
        }
        else{
            println!("tried to get a file with an invalid handle {}", file_id);
            return 0;
        }
    })
}

/// Load a texture and return a texture handle
pub fn create_texture_from_file(
    mut env: FunctionEnvMut<'_, Env>,
    file_handle: i32,
    texture_params_ptr: i32,
) -> u32 {
    with_stage!(stage => {
        let op = stage.file_load_operations.get(file_handle as usize).expect("file handle id is invalid!");

        let Ok(locked) = op.lock() else {
            panic!("Failed to aquire resource. Check file load status before access if unsure.")
        };

        let Some(ref bytes) = locked.result else {
            eprintln!("texture accessed before load was complete");
            return u32::MAX;
        };

        let mut params = TextureParams::from_wasm_mem(&mut env, texture_params_ptr);

        let cursor = std::io::Cursor::new(bytes);
        let img = ImageReader::new(cursor)
        .with_guessed_format()
        .expect("Guess format failed")
        .decode()
        .expect("Decode failed");

        params.width =img.width();
        params.height = img.height();

       let img_bytes = match params.format {
            miniquad::TextureFormat::RGB8 => img.to_rgb8().into_raw(),
            miniquad::TextureFormat::RGBA8 => img.to_rgba8().into_raw(),
            miniquad::TextureFormat::RGBA16F => {
                // Convert to 16-bit RGBA, then cast to &[u8]
                let img = img.to_rgba16();
                let data_u16: &[u16] = img.as_raw();
                let data_u8: &[u8] = bytemuck::cast_slice(data_u16);
                data_u8.to_vec()
            }
            miniquad::TextureFormat::Depth => {
                // If this is used for rendering depth, just create a blank buffer
                let (w, h) = img.dimensions();
                vec![0u8; (w * h) as usize * 2] // 16-bit depth = 2 bytes
            }
            miniquad::TextureFormat::Depth32 => {
                let (w, h) = img.dimensions();
                vec![0u8; (w * h) as usize * 4] // 32-bit depth = 4 bytes
            }
            miniquad::TextureFormat::Alpha => {
                let gray = img.to_luma8();
                gray.into_raw() // 1 byte per pixel
            }
        };

        let tex_id = stage.ctx.new_texture_from_data_and_format(&img_bytes, params);

        stage.textures.push(tex_id) as u32
    })
}

pub fn create_texture_from_rgba8(
    mut env: FunctionEnvMut<Env>,
    width: i32,
    height: i32,
    pixel_ptr: i32,
    pixel_count: i32,
) -> u32 {
    let mut buf = vec![0; pixel_count as usize * 4]; // 4 bytes per pixel
    read_mem_slice(&mut env, pixel_ptr, &mut buf);

    with_stage!(stage => {
        let tex_handle = stage.ctx.new_texture_from_rgba8(width as u16, height as u16, &buf);
        stage.textures.push(tex_handle) as u32
    })
}

pub fn create_texture_from_data_and_format(
    mut env: FunctionEnvMut<Env>,
    byte_ptr: i32,
    byte_count: i32,
    texture_params_ptr: i32,
) -> u32 {
    let params = TextureParams::from_wasm_mem(&mut env, texture_params_ptr);
    let mut buf = vec![0; byte_count as usize];
    read_mem_slice(&mut env, byte_ptr, &mut buf);

    with_stage!(stage => {
        let tex_handle = stage.ctx.new_texture_from_data_and_format(&buf, params);
        stage.textures.push(tex_handle) as u32
    })
}

pub fn draw(_env: FunctionEnvMut<Env>, base_element: i32, num_elements: i32, num_instances: i32) {
    with_stage!(stage => {
        stage.ctx.draw(base_element, num_elements, num_instances);
    });
}

pub fn apply_uniforms_from_bytes(mut env: FunctionEnvMut<Env>, ptr: i32, byte_count: i32) {
    with_stage!(stage => {
        let mut buf = vec![0; byte_count as usize];
        read_mem_slice(&mut env, ptr, &mut buf);
        let mem_ptr = buf.as_ptr() as *const u8;
        stage.ctx.apply_uniforms_from_bytes(mem_ptr, byte_count as usize);
    });
}

pub fn apply_pipeline(_env: FunctionEnvMut<Env>, pipeline_id: i32) {
    with_stage!(stage => {
        stage.ctx.apply_pipeline(&stage.pipelines[pipeline_id as usize]);
    });
}

pub fn apply_bindings(_env: FunctionEnvMut<Env>, binding_id: i32) {
    with_stage!(stage => {
        stage.ctx.apply_bindings(&stage.geo_bindings[binding_id as usize]);
    });
}

/// Takes an address to a jam pass action sum type
pub fn begin_default_pass(mut env: FunctionEnvMut<Env>, pass_action_pointer: i32) {
    let action = miniquad::PassAction::from_wasm_mem(&mut env, pass_action_pointer);
    with_stage!(stage => {
        stage.ctx.begin_default_pass(action);
    });
}

pub fn end_render_pass(_env: FunctionEnvMut<Env>) {
    with_stage!(stage => {
        stage.ctx.end_render_pass();
    });
}

pub fn commit_frame(_env: FunctionEnvMut<Env>) {
    with_stage!(stage => {
        stage.ctx.commit_frame();
    })
}
