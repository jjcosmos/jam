use wasm_encoder::ExportKind;

use crate::wasm_backend::wasm_codegen::WasmModule;

impl<'a> WasmModule<'a> {
    pub fn write_export_section(&mut self) {
        // TODO: Map out exports like the code sections
        self.export_section.export("memory", ExportKind::Memory, 0);
        self.module.section(&self.export_section);
    }
}
