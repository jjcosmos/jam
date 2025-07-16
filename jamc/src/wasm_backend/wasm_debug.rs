use hashbrown::HashMap;

use wasm_encoder::{IndirectNameMap, NameMap};

use crate::{frontend::sema::Ty, wasm_backend::wasm_codegen::WasmModule};

impl<'a> WasmModule<'a> {
    pub fn write_name_section(&mut self) {
        self.name_section.functions(&self.dbg_name_bld.funcs);
        self.name_section.globals(&self.dbg_name_bld.globals);
        self.name_section.types(&self.dbg_name_bld.types);
        self.name_section
            .locals(&self.dbg_name_bld.gen_indirect_map());
        self.module.section(&self.name_section);
    }
}

pub fn get_var_dbg_name(name: &str, ty: &Ty) -> String {
    format!("{}_{:?}", name, ty)
        .replace("(", "__")
        .replace(")", "__")
        .replace("\"", "_")
        .replace(" ", "")
        //.replace(".", "")
        .replace(",", "")
        .replace("[", "")
        .replace("]", "")
}

pub struct DebugNameBuilder {
    funcs: NameMap,
    globals: NameMap,
    types: NameMap,
    locals_hashmap: HashMap<u32, NameMap>,
}

impl DebugNameBuilder {
    pub fn new() -> Self {
        Self {
            funcs: NameMap::new(),
            globals: NameMap::new(),
            types: NameMap::new(),
            locals_hashmap: HashMap::new(),
        }
    }

    pub fn add_func(&mut self, index: u32, name: &str) {
        self.funcs.append(index, name);
    }

    pub fn gen_indirect_map(&self) -> IndirectNameMap {
        let mut map = IndirectNameMap::new();
        for (k, v) in self.locals_hashmap.iter() {
            map.append(*k, v);
        }

        map
    }

    pub fn add_global(&mut self, global_idx: u32, name: &str) {
        self.globals.append(global_idx, name);
    }

    pub fn add_type(&mut self, type_idx: u32, name: &str) {
        self.types.append(type_idx, name);
    }

    pub fn add_local(&mut self, func_index: u32, local_index: u32, name: &str) {
        self.locals_hashmap
            .entry(func_index)
            .or_insert_with(NameMap::new)
            .append(local_index, &format!("loc_{}_{}", local_index, name));
    }
}
