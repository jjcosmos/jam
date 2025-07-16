use bytemuck::{Pod, Zeroable};
use miniquad::{
    BlendFactor, BlendState, CompareFunc, PipelineParams, PrimitiveType, StencilFaceState,
    StencilOp, StencilState, UniformBlockLayout, UniformDesc, UniformType, VertexStep,
};
use wasmer::FunctionEnvMut;

use crate::Env;

macro_rules! impl_unit_enum_conv {
    ($enum:path { $($variant:ident = $val:expr),* $(,)? }) => {
        impl UnitOnlyEnum for $enum {
            fn from_i32(val: i32) -> Self {
                match val {
                    $($val => <$enum>::$variant),*,
                    _ => panic!("invalid enum value {}", val),
                }
            }
            fn to_i32(&self) -> i32 {
                match self {
                    $(<$enum>::$variant => $val),*
                }
            }
        }
    };
}

pub trait UnitOnlyEnum {
    fn from_i32(val: i32) -> Self;

    #[allow(unused)]
    fn to_i32(&self) -> i32;
}

// Common functionality
pub trait FromWasmMem<T> {
    fn from_wasm_mem(env: &mut FunctionEnvMut<Env>, address: i32) -> T;
}

type JamDiscriminant = i32;
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DummyOption<T> {
    discrim: JamDiscriminant,
    value: T,
}

impl<T> DummyOption<T> {
    fn is_some(&self) -> bool {
        self.discrim == 1
    }

    fn into_option(&self) -> Option<T>
    where
        T: Clone,
    {
        if self.is_some() {
            Some(self.value.clone())
        } else {
            None
        }
    }
}

pub fn load_string_nt(env: &mut FunctionEnvMut<Env>, nt_pointer: i32) -> String {
    const MAX_LEN: usize = 1024;

    let (env_data, store) = env.data_and_store_mut();
    let mem = env_data.memory.as_ref().expect("memory not initialized");

    let mut buf = Vec::new();
    for i in 0..MAX_LEN {
        let offset = nt_pointer as u64 + i as u64;
        let byte = mem
            .view(&store)
            .read_u8(offset)
            .expect("failed to read memory during print");

        if byte == 0 {
            if i == 0 {
                print!("passed in 0 len string at address {}", nt_pointer);
            }
            break;
        }
        buf.push(byte);
    }

    match String::from_utf8(buf) {
        Ok(s) => s,
        Err(e) => panic!("[invalid utf8: {:?}]", e),
    }
}

pub fn read_mem_slice(env: &mut FunctionEnvMut<Env>, address: i32, buf: &mut Vec<u8>) //-> *const u8
{
    if address == 0 {
        panic!("null pointer passed from environment");
    }

    let (env_data, store) = env.data_and_store_mut();
    let mem = env_data.memory.as_ref().unwrap();

    mem.view(&store)
        .read(address as u64, buf)
        .expect("memory read error");

    // let ptr = buf.as_ptr() as *const u8;
    // ptr
}

/// Load into a dummy structure that should be an exact match given the jam ABI.
///
/// Union types always have an i32 discriminant, but simple struct and array types with repr(c) should otherwise suffice
pub fn from_ptr<T>(env: &mut FunctionEnvMut<Env>, address: i32) -> T {
    if address == 0 {
        panic!("null pointer passed from environment");
    }

    let (env_data, store) = env.data_and_store_mut();
    let mem = env_data.memory.as_ref().unwrap();

    let size_t = size_of::<T>();

    let mut buf = vec![0; size_t];
    mem.view(&store)
        .read(address as u64, &mut buf)
        .expect("memory read error");

    // could use *bytemuck::from_bytes::<T>(&buf) if requiring pod

    let ptr = buf.as_ptr() as *const T;
    let result = unsafe { std::ptr::read(ptr) };
    result
}

// Pass action

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DummyPassAction {
    pub discrim: i32, // 0 = Nothing, 1 = Clear
    pub color: DummyOption<[f32; 4]>,
    pub depth: DummyOption<f32>,
    pub stencil: DummyOption<i32>,
}

impl DummyPassAction {
    const PASS_ACTION_NOTHING: i32 = 0;
    const PASS_ACTION_CLEAR: i32 = 1;
}

impl FromWasmMem<miniquad::PassAction> for miniquad::PassAction {
    fn from_wasm_mem(env: &mut FunctionEnvMut<Env>, address: i32) -> miniquad::PassAction {
        let result = from_ptr::<DummyPassAction>(env, address);

        match result.discrim {
            DummyPassAction::PASS_ACTION_NOTHING => miniquad::PassAction::Nothing,
            DummyPassAction::PASS_ACTION_CLEAR => {
                let color = result.color.into_option().map(|o| {
                    let [r, g, b, a] = o;
                    (r, g, b, a)
                });

                let depth = result.depth.into_option();
                let stencil = result.stencil.into_option();
                miniquad::PassAction::Clear {
                    color: color,
                    depth: depth,
                    stencil: stencil,
                }
            }
            _ => {
                panic!("unknown discriminant {}", result.discrim)
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DummyTextureParams {
    pub kind: i32,
    pub format: i32,
    pub wrap: i32,
    pub min_filter: i32,
    pub mag_filter: i32,
    pub mipmap_filter: i32,
    pub width: u32,
    pub height: u32,
    // All miniquad API could work without this flag being explicit.
    // We can decide if mipmaps are required by the data provided
    // And reallocate non-mipmapped texture(on metal) on generateMipmaps call
    // But! Reallocating cubemaps is too much struggle, so leave it for later.
    pub allocate_mipmaps: bool,
    /// Only used for render textures. `sample_count > 1` allows anti-aliased render textures.
    ///
    /// On OpenGL, for a `sample_count > 1` render texture, render buffer object will
    /// be created instead of a regulat texture.
    ///
    pub sample_count: i32,
}

impl FromWasmMem<miniquad::TextureParams> for miniquad::TextureParams {
    fn from_wasm_mem(env: &mut FunctionEnvMut<Env>, address: i32) -> miniquad::TextureParams {
        let result = from_ptr::<DummyTextureParams>(env, address);
        miniquad::TextureParams {
            kind: miniquad::TextureKind::from_i32(result.kind),
            format: miniquad::TextureFormat::from_i32(result.format),
            wrap: miniquad::TextureWrap::from_i32(result.wrap),
            min_filter: miniquad::FilterMode::from_i32(result.min_filter),
            mag_filter: miniquad::FilterMode::from_i32(result.mag_filter),
            mipmap_filter: miniquad::MipmapFilterMode::from_i32(result.mag_filter),
            width: result.width,
            height: result.height,
            allocate_mipmaps: result.allocate_mipmaps,
            sample_count: result.sample_count,
        }
    }
}

#[derive(Debug, Clone, Copy, Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct DummyBufferLayout {
    pub stride: i32,
    pub step_func: i32, // VertexStep
    pub step_rate: i32,
}

impl FromWasmMem<miniquad::graphics::BufferLayout> for miniquad::graphics::BufferLayout {
    fn from_wasm_mem(
        env: &mut FunctionEnvMut<Env>,
        address: i32,
    ) -> miniquad::graphics::BufferLayout {
        let result = from_ptr::<DummyBufferLayout>(env, address);
        miniquad::graphics::BufferLayout {
            stride: result.stride,
            step_func: VertexStep::from_i32(result.step_func),
            step_rate: result.step_rate,
        }
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct DummyShaderMeta {
    pub uniforms: DummyUniformBlockLayout,
    // These are the names of the texture samplers, not the actual textures
    //pub images: Vec<i32>, //Vec<String>, (read as pointer)
    pub images_ptr: i32,
    pub images_count: i32,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct DummyUniformBlockLayout {
    //pub uniforms: Vec<DummyUniformDesc>,
    pub uniform_ptr: i32,
    pub uniform_count: i32,
}

#[derive(Debug, Clone, Copy, Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct DummyUniformDesc {
    // Name of the uniform
    pub name: i32,         //String, (read as pointer)
    pub uniform_type: i32, //UniformType,
    pub array_count: i32,
}

impl FromWasmMem<miniquad::ShaderMeta> for miniquad::ShaderMeta {
    fn from_wasm_mem(env: &mut FunctionEnvMut<Env>, address: i32) -> miniquad::ShaderMeta {
        let result = from_ptr::<DummyShaderMeta>(env, address);

        let dummy_uniform_vec = read_list_ptrs::<DummyUniformDesc>(
            env,
            result.uniforms.uniform_ptr,
            result.uniforms.uniform_count as usize,
        );

        let uniform_vec = dummy_uniform_vec
            .iter()
            .map(|dummy| UniformDesc {
                name: load_string_nt(env, dummy.name),
                uniform_type: UniformType::from_i32(dummy.uniform_type),
                array_count: dummy.array_count as usize,
            })
            .collect();

        let image_ptr_list: Vec<i32> =
            read_list_prim(env, result.images_ptr, result.images_count as usize);
        let img_sampler_names = image_ptr_list
            .iter()
            .map(|str_ptr| load_string_nt(env, *str_ptr))
            .collect();

        miniquad::ShaderMeta {
            uniforms: UniformBlockLayout {
                uniforms: uniform_vec,
            },
            images: img_sampler_names,
        }
    }
}

#[derive(Debug, Clone, Copy, Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct DummyVertexAttribute {
    pub name: i32,   //&'static str,
    pub format: i32, //VertexFormat,
    pub buffer_index: u32,
    pub gl_pass_as_float: u8, //bool,
    pub _padding: [u8; 3],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DummyPipelineParams {
    pub cull_face: i32,        // CullFace,
    pub front_face_order: i32, // FrontFaceOrder,
    pub depth_test: i32,       //Comparison,
    pub depth_write: bool,
    pub depth_write_offset: DummyOption<[f32; 2]>, // Option<(f32, f32)>,
    pub color_blend: DummyOption<DummyBlendState>, //Option<BlendState>,
    pub alpha_blend: DummyOption<DummyBlendState>,
    pub stencil_test: DummyOption<DummyStencilState>,
    pub color_write: [bool; 4],
    pub primitive_type: i32, //PrimitiveType,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DummyBlendState {
    equation: i32, //Equation,
    sfactor: DummyBlendFactor,
    dfactor: DummyBlendFactor,
}

impl DummyBlendState {
    fn into_blend_state(&self) -> BlendState {
        BlendState::new(
            miniquad::Equation::from_i32(self.equation),
            self.sfactor.into_blend_factor(),
            self.dfactor.into_blend_factor(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DummyBlendFactor {
    discriminant: u32,
    value: [u8; 4], // Size of max variant
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct DummyStencilState {
    pub front: DummyStencilFaceState,
    pub back: DummyStencilFaceState,
}

impl DummyStencilState {
    fn into_stencil_state(&self) -> StencilState {
        StencilState {
            front: self.front.into_stencil_face_state(),
            back: self.back.into_stencil_face_state(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct DummyStencilFaceState {
    pub fail_op: i32,       // StencilOp,
    pub depth_fail_op: i32, //StencilOp,
    pub pass_op: i32,       //StencilOp,
    pub test_func: i32,     //CompareFunc,
    pub test_ref: i32,
    pub test_mask: u32,
    pub write_mask: u32,
}

impl DummyStencilFaceState {
    fn into_stencil_face_state(&self) -> StencilFaceState {
        StencilFaceState {
            fail_op: StencilOp::from_i32(self.fail_op),
            depth_fail_op: StencilOp::from_i32(self.depth_fail_op),
            pass_op: StencilOp::from_i32(self.pass_op),
            test_func: CompareFunc::from_i32(self.test_func),
            test_ref: self.test_ref,
            test_mask: self.test_mask,
            write_mask: self.write_mask,
        }
    }
}

impl DummyBlendFactor {
    fn into_blend_factor(&self) -> BlendFactor {
        match self.discriminant {
            0 => BlendFactor::Zero,
            1 => BlendFactor::One,
            2 => {
                let val_int = i32::from_le_bytes(self.value);
                BlendFactor::Value(miniquad::BlendValue::from_i32(val_int))
            }
            3 => {
                let val_int = i32::from_le_bytes(self.value);
                BlendFactor::OneMinusValue(miniquad::BlendValue::from_i32(val_int))
            }
            4 => BlendFactor::SourceAlphaSaturate,
            _ => panic!("invalid enum disrim {}", self.discriminant),
        }
    }
}

impl FromWasmMem<miniquad::PipelineParams> for miniquad::PipelineParams {
    fn from_wasm_mem(env: &mut FunctionEnvMut<Env>, address: i32) -> miniquad::PipelineParams {
        let result = from_ptr::<DummyPipelineParams>(env, address);

        PipelineParams {
            cull_face: miniquad::CullFace::from_i32(result.cull_face),
            front_face_order: miniquad::FrontFaceOrder::from_i32(result.front_face_order),
            depth_test: miniquad::Comparison::from_i32(result.depth_test),
            depth_write: result.depth_write,
            depth_write_offset: result
                .depth_write_offset
                .into_option()
                .map(|vec| tuple2v(&vec)),
            color_blend: result
                .color_blend
                .into_option()
                .map(|value| value.into_blend_state()),
            alpha_blend: result
                .alpha_blend
                .into_option()
                .map(|value| value.into_blend_state()),
            stencil_test: result
                .stencil_test
                .into_option()
                .map(|value| value.into_stencil_state()),
            color_write: tuple4v(&result.color_write),
            primitive_type: PrimitiveType::from_i32(result.primitive_type),
        }
    }
}

// This would be a really good case for proc macros
impl_unit_enum_conv!(miniquad::PrimitiveType {
    Triangles=0,
    Lines=1,
    Points=2,
});

impl_unit_enum_conv!(miniquad::CompareFunc {
    Always=0,
    Never=1,
    Less=2,
    Equal=3,
    LessOrEqual=4,
    Greater=5,
    NotEqual=6,
    GreaterOrEqual=7,
});

impl_unit_enum_conv!(miniquad::StencilOp {
    Keep=0,
    Zero=1,
    Replace=2,
    IncrementClamp=3,
    DecrementClamp=4,
    Invert=5,
    IncrementWrap=6,
    DecrementWrap=7,
});

impl_unit_enum_conv!(miniquad::BlendValue {
    SourceColor = 0,
    SourceAlpha = 1,
    DestinationColor = 2,
    DestinationAlpha = 3,
});

impl_unit_enum_conv!(miniquad::Equation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
});

impl_unit_enum_conv!(miniquad::CullFace {
    Nothing = 0,
    Front = 1,
    Back = 2,
});

impl_unit_enum_conv!(miniquad::FrontFaceOrder{
    Clockwise = 0,
    CounterClockwise=1,
});

impl_unit_enum_conv!(miniquad::Comparison{
    Never=0,
    Less=1,
    LessOrEqual=2,
    Greater=3,
    GreaterOrEqual=4,
    Equal=5,
    NotEqual=6,
    Always=7,
});

// Read a list of primitive structures.
// Would only work for arrays of primitives, otherwise would return arrays of pointers
pub fn read_list_prim<T>(
    env: &mut FunctionEnvMut<Env>,
    address: i32,
    element_count: usize,
) -> Vec<T>
where
    T: Pod, // + Zeroable
{
    let mut buf = vec![0; size_of::<T>() * element_count];
    read_mem_slice(env, address, &mut buf);

    // Doesn't work, align is off?
    // bytemuck::try_cast_vec::<u8, T>(buf).expect("failed to unwrap list vector")

    // Safer, but double copy (wmem -> rt -> owned)
    let vec: Vec<T> = buf
        .chunks_exact(size_of::<T>())
        .map(|chunk| bytemuck::from_bytes::<T>(chunk).to_owned())
        .collect();
    vec

    // This assumes correct align (which is should be)
    // let ptr = buf.as_mut_ptr();
    // let cap = buf.capacity();
    // std::mem::forget(buf);

    // let t_ptr = ptr as *mut T;

    // unsafe { Vec::from_raw_parts(t_ptr, element_count, cap / size_of::<T>()) }
}

pub fn read_list_ptrs<T>(
    env: &mut FunctionEnvMut<Env>,
    address: i32,
    element_count: usize,
) -> Vec<T> {
    let ptrs: Vec<i32> = read_list_prim(env, address, element_count);
    let vals: Vec<T> = ptrs.iter().map(|p| from_ptr(env, *p)).collect();
    vals
}

impl_unit_enum_conv!(miniquad::VertexFormat {
    Float1 = 0,
    Float2 = 1,
    Float3 = 2,
    Float4 = 3,
    Byte1 = 4,
    Byte2 = 5,
    Byte3 = 6,
    Byte4 = 7,
    Short1 = 8,
    Short2=9,
    Short3=10,
    Short4=11,
    Int1=12,
    Int2=13,
    Int3=14,
    Int4=15,
    Mat4=16,
});

impl_unit_enum_conv!(miniquad::UniformType {
    Float1 = 0,
    Float2 = 1,
    Float3 = 2,
    Float4 = 3,
    Int1 = 4,
    Int2 = 5,
    Int3 = 6,
    Int4 = 7,
    Mat4 = 8,
});

impl_unit_enum_conv!(miniquad::TextureKind {
    Texture2D = 0,
    CubeMap = 1,
});

impl_unit_enum_conv!(miniquad::TextureFormat {
    RGB8 = 0,
    RGBA8 = 1,
    RGBA16F = 2,
    Depth = 3,
    Depth32 = 4,
    Alpha = 5,
});

impl_unit_enum_conv!(miniquad::TextureWrap {
    Repeat = 0,
    Mirror = 1,
    Clamp = 2,
});

impl_unit_enum_conv!(miniquad::FilterMode {
    Linear = 0,
    Nearest = 1,
});

impl_unit_enum_conv!(miniquad::MipmapFilterMode {
    None = 0,
    Linear = 1,
    Nearest = 2,
});

impl_unit_enum_conv!(miniquad::graphics::VertexStep {
    PerVertex = 0,
    PerInstance = 1,
});

#[allow(unused)]
fn tuple1<T>(a: &[T]) -> &T {
    &a[0]
}
#[allow(unused)]
fn tuple2<T>(a: &[T]) -> (&T, &T) {
    (&a[0], &a[1])
}
#[allow(unused)]
fn tuple3<T>(a: &[T]) -> (&T, &T, &T) {
    (&a[0], &a[1], &a[2])
}
#[allow(unused)]
fn tuple4<T>(a: &[T]) -> (&T, &T, &T, &T) {
    (&a[0], &a[1], &a[2], &a[3])
}
#[allow(unused)]
fn tuple5<T>(a: &[T]) -> (&T, &T, &T, &T, &T) {
    (&a[0], &a[1], &a[2], &a[3], &a[4])
}
#[allow(unused)]
fn tuple6<T>(a: &[T]) -> (&T, &T, &T, &T, &T, &T) {
    (&a[0], &a[1], &a[2], &a[3], &a[4], &a[5])
}
#[allow(unused)]
fn tuple1v<T>(a: &[T]) -> T
where
    T: Copy,
{
    a[0]
}
#[allow(unused)]
fn tuple2v<T>(a: &[T]) -> (T, T)
where
    T: Copy,
{
    (a[0], a[1])
}
#[allow(unused)]
fn tuple3v<T>(a: &[T]) -> (T, T, T)
where
    T: Copy,
{
    (a[0], a[1], a[2])
}
#[allow(unused)]
fn tuple4v<T>(a: &[T]) -> (T, T, T, T)
where
    T: Copy,
{
    (a[0], a[1], a[2], a[3])
}
#[allow(unused)]
fn tuple5v<T>(a: &[T]) -> (T, T, T, T, T)
where
    T: Copy,
{
    (a[0], a[1], a[2], a[3], a[4])
}
#[allow(unused)]
fn tuple6v<T>(a: &[T]) -> (T, T, T, T, T, T)
where
    T: Copy,
{
    (a[0], a[1], a[2], a[3], a[4], a[5])
}
