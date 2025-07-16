use wasm_encoder::Instruction;

use crate::{frontend::sema::Ty, wasm_backend::wasm_codegen::WasmModule};

impl<'a> WasmModule<'a> {
    pub fn normalize_to_i32(&mut self, from_ty: &Ty) {
        match from_ty {
            Ty::Bool => {
                self.emit_current(Instruction::I32Const(1));
                self.emit_current(Instruction::I32And);
            }
            Ty::I8 => self.emit_current(Instruction::I32Extend8S),
            Ty::I16 => self.emit_current(Instruction::I32Extend16S),
            Ty::U8 => {
                self.emit_current(Instruction::I32Const(0xFF));
                self.emit_current(Instruction::I32And);
            }
            Ty::U16 => {
                self.emit_current(Instruction::I32Const(0xFFFF));
                self.emit_current(Instruction::I32And);
            }
            _ => {} // already i32 or wider
        }
    }

    pub fn cast_i32_to(&mut self, from_ty: &Ty, to_ty: &Ty) {
        match to_ty {
            Ty::Bool => {
                // Normalize to 0 or 1: (x != 0) as bool
                self.emit_current(Instruction::I32Const(0));
                self.emit_current(Instruction::I32Ne);
            }
            Ty::I32 | Ty::U32 => {} // already normalized
            Ty::I64 => self.emit_current(Instruction::I64ExtendI32S),
            Ty::U64 => self.emit_current(Instruction::I64ExtendI32U),
            Ty::I8 => self.emit_current(Instruction::I32Extend8S),
            Ty::U8 => {
                self.emit_current(Instruction::I32Const(0xFF));
                self.emit_current(Instruction::I32And);
            }
            Ty::I16 => self.emit_current(Instruction::I32Extend16S),
            Ty::U16 => {
                self.emit_current(Instruction::I32Const(0xFFFF));
                self.emit_current(Instruction::I32And);
            }
            Ty::F32 => {
                if from_ty.is_signed() {
                    self.emit_current(Instruction::F32ConvertI32S);
                } else {
                    self.emit_current(Instruction::F32ConvertI32U);
                }
            }
            Ty::F64 => {
                if from_ty.is_signed() {
                    self.emit_current(Instruction::F64ConvertI32S);
                } else {
                    self.emit_current(Instruction::F64ConvertI32U);
                }
            }
            _ => panic!("unsupported cast from {:?} to {:?}", from_ty, to_ty),
        }
    }

    pub fn truncate_long_int_to_small(&mut self, to_ty: &Ty) {
        self.emit_current(Instruction::I32WrapI64);
        self.cast_i32_to(&Ty::I32, to_ty);
    }
}
