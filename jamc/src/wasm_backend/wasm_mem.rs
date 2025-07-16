use hashbrown::{HashMap, HashSet};

use wasm_encoder::{ConstExpr, MemoryType};

use crate::wasm_backend::wasm_codegen::WasmModule;

// Stack space (virtual stack bump allocator)
pub(crate) const VIRTUAL_STACK_SIZE: usize = 32 * 1024; // 32 KB

// Heap space
pub(crate) const DEFAULT_HEAP_SIZE: usize = 1024 * 1024 * 4; // 4 MiB

// Size of a wasm page in bytes
pub(crate) const WASM_PAGE_SIZE: usize = 64 * 1024;

// Buffer can never be more than the default heap size
#[allow(unused)]
pub(crate) const DATA_BASE: usize = size_of::<i64>(); // Should maybe not start at 0 to reserve addr 0 for null ptr?

const fn round_up_to_page(x: usize) -> usize {
    (x + WASM_PAGE_SIZE - 1) & !(WASM_PAGE_SIZE - 1)
}

//const MIN_PAGES: usize = (STARTING_MEM_SIZE + WASM_PAGE_SIZE - 1) / (WASM_PAGE_SIZE);

impl<'a> WasmModule<'a> {
    pub(crate) fn init_mem(&mut self) {
        let starting_mem_size = round_up_to_page(
            DATA_BASE
                + self.data_builder.size()
                + self.bss_builder.size()
                + VIRTUAL_STACK_SIZE
                + DEFAULT_HEAP_SIZE,
        );
        let min_pages = (starting_mem_size + WASM_PAGE_SIZE - 1) / (WASM_PAGE_SIZE);

        self.memory_section.memory(MemoryType {
            minimum: min_pages as u64,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
    }

    pub(crate) fn write_data_section(&mut self) {
        for (offset, bytes) in &self.data_builder.entries {
            self.data_section.active(
                0 as u32, // Memory index if for mult-memory models
                &ConstExpr::i32_const(*offset as i32),
                bytes.clone(),
            );
        }

        self.module.section(&self.data_section);
    }

    pub fn get_stack_base(&self) -> usize {
        let stack_base = align_up(
            DATA_BASE + self.data_builder.size() + self.bss_builder.size(),
            16,
        );
        stack_base
    }
}

pub struct BssSectionBuilder {
    pub base_offset: usize,
    pub current_offset: usize,
    pub entries: Vec<usize>,
}

impl BssSectionBuilder {
    pub fn uninit() -> Self {
        Self {
            base_offset: 0,
            current_offset: 0,
            entries: vec![],
        }
    }

    pub fn init_from_data_builder(
        data_builder: &DataSectionBuilder,
        static_user_strings: &HashSet<String>,
    ) -> Self {
        // data section size should be all 'static strs + constants
        // annoying part is that I don't want to duplicate 'static strs, so they are a special case
        // bss data size can be easily determined with just the static defs
        let base_offset = DATA_BASE
            + data_builder.non_static_string_size
            + Self::static_user_string_size(static_user_strings);
        Self {
            base_offset: base_offset,
            current_offset: base_offset,
            entries: vec![],
        }
    }

    fn static_user_string_size(static_user_strings: &HashSet<String>) -> usize {
        static_user_strings.iter().map(|s| s.len() + 1).sum()
    }

    pub fn add_entry(&mut self, size_to_resevere: usize, align: usize) -> usize {
        let offset = align_up(self.current_offset, align);
        self.entries.push(size_to_resevere);
        self.current_offset += size_to_resevere;
        offset
    }

    pub fn size(&self) -> usize {
        self.current_offset - self.base_offset
    }
}

pub struct DataSectionBuilder {
    pub base_offset: usize,
    pub current_offset: usize,
    pub entries: Vec<(usize, Vec<u8>)>, // (offset, bytes)

    // Keep a map of seen static strings to their offset
    pub string_map: HashMap<String, usize>,
    pub non_static_string_size: usize,
}

// The only things in the data section will be cstrings defined in either the
// const section or function bodies, and consts.
// Therefor, if the constants are all parsed, the size of this section is
// non_static_string_size + bytelen(sema_static_strings)
impl DataSectionBuilder {
    pub fn new(base_offset: usize) -> Self {
        Self {
            base_offset: base_offset,
            current_offset: base_offset,
            entries: vec![],
            string_map: HashMap::new(),
            non_static_string_size: 0,
        }
    }

    // Add and null-terminate a rust string
    // Not used in const expressions
    pub fn add_string(&mut self, string: &str) -> usize {
        if let Some(entry) = self.string_map.get(string) {
            return *entry;
        }

        let mut bytes = string.as_bytes().to_vec();
        bytes.push(b'\0');

        // Align for cstrings is 1, since it is just a byte array
        let offset = self.add_data(&bytes, true, 1);
        self.string_map.insert(string.to_owned(), offset);
        offset
    }

    pub fn add_data(&mut self, bytes: &[u8], is_static_string: bool, mut align: usize) -> usize {
        // Hack since cstrings are semantically ptr(u8)
        if is_static_string {
            align = 1
        };

        let offset = align_up(self.current_offset, align);
        //println!("Adding bytes at offset {} {:?}", offset, bytes);
        self.entries.push((offset, bytes.to_vec()));
        self.current_offset += bytes.len();

        if !is_static_string {
            self.non_static_string_size += bytes.len();
        }

        // return offset as compile time known pointer
        offset
    }

    pub fn size(&self) -> usize {
        self.current_offset - self.base_offset
    }
}

// i.e 3 align 2 -> nearest power of 2

// value 3
// 0011 (3)
// 0100 (+ 2 - 1) = 4
// &
// align 2
// 0001 (align 2 - 1)
// 1110 (invert)
// =
// 0100 // 4
// I def wrote this somewhere else as well
pub fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

// This way I can realloc space without breaking the stack's state
// 0x00000000 ─────────────▶ +------------------+
//                          │  .data section     │
//                          │  Static data       │ (Data section)
//                          │  Globals (RW)      │
//                          │--------------------│
//                          │  .bss section:     │
//                          │  Globals 0 init    │
//                          │--------------------│
//                          │  Virtual Stack     │ <- stack pointer initialized here
//                          │...grows downward...│
//                          │--------------------│
//                          │  Heap start here   │ <- bump allocator, malloc, etc
//                          │  ...grows upward...│
//                          │                    │
//                          │--------------------│
//                          │  Free space        │
//                          +--------------------+
//                             HIGH ADDRESS SPACE

// 2's compl
// 1010 = −(1×2^3) + (0×2^2) + (1×2^1) + (0×2^0) = 1×−8 + 0 + 1×2 + 0 = −6.
// Indexed right to left :
// -(bitval[max_place] * 2^max_place) + sum for n in 0..max_place (bitval[n] * 2^n)
