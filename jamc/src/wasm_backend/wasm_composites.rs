use wasm_encoder::{Instruction, MemArg};

use crate::{
    frontend::{
        ast::{Expr, Ident, StructDef},
        sema::Ty,
    },
    wasm_backend::{wasm_codegen::WasmModule, wasm_function::GenericInstTyKey},
};

impl<'a> WasmModule<'a> {
    /// Dest -> then value need to be on the stack
    pub fn emit_store_to_ptr(&mut self, ty: &Ty) {
        let memarg = self.memarg_for_ty(ty);

        // println!(
        //     "Emitting store with memarg {:?} (align: {})",
        //     memarg,
        //     2_i32.pow(memarg.align)
        // );

        match ty {
            Ty::I8 | Ty::U8 => self.emit_current(Instruction::I32Store8(memarg)),
            Ty::I16 | Ty::U16 => self.emit_current(Instruction::I32Store16(memarg)),
            Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => {
                self.emit_current(Instruction::I32Store(memarg))
            }
            Ty::I64 | Ty::U64 => self.emit_current(Instruction::I64Store(memarg)),
            Ty::F32 => self.emit_current(Instruction::F32Store(memarg)),
            Ty::F64 => self.emit_current(Instruction::F64Store(memarg)),
            Ty::Ptr(_) | Ty::NullPtr(_) => {
                // Treat pointer as i32 offset
                self.emit_current(Instruction::I32Store(memarg))
            }
            Ty::FuncPtr(_, _) => {
                self.emit_current(Instruction::I32Store(memarg));
            }
            _ => panic!("Unsupported type for store: {:?}", ty),
        }
    }

    /// This expects the ptr to already be on the stack
    pub fn emit_load_from_ptr(&mut self, ty: &Ty) {
        let memarg = self.memarg_for_ty(ty);

        match ty {
            Ty::I8 => self.emit_current(Instruction::I32Load8S(memarg)),
            Ty::U8 => self.emit_current(Instruction::I32Load8U(memarg)),
            Ty::I16 => self.emit_current(Instruction::I32Load16S(memarg)),
            Ty::U16 => self.emit_current(Instruction::I32Load16U(memarg)),
            Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => {
                self.emit_current(Instruction::I32Load(memarg))
            }
            Ty::I64 | Ty::U64 => self.emit_current(Instruction::I64Load(memarg)),
            Ty::F32 => self.emit_current(Instruction::F32Load(memarg)),
            Ty::F64 => self.emit_current(Instruction::F64Load(memarg)),
            Ty::Ptr(_) | Ty::NullPtr(_) => {
                // Treat pointer as raw i32 offset
                self.emit_current(Instruction::I32Load(memarg))
            }
            _ => panic!("Unsupported type for load: {:?}", ty),
        }
    }

    /// Pushes the value at the address stored in 'ptr_local' onto the stack.
    pub fn emit_load_from_local(&mut self, local_for_ptr: u32, ty: &Ty) {
        println!("Emitting load from local {} {:?}", local_for_ptr, ty);

        // Push the pointer onto the stack
        self.emit_current(Instruction::LocalGet(local_for_ptr));
        self.emit_load_from_ptr(ty);
    }

    fn memarg_for_ty(&mut self, ty: &Ty) -> MemArg {
        let (_size, align) = self.get_size_and_align(ty);

        let align_log2 = align.next_power_of_two().trailing_zeros(); // log2
        MemArg {
            offset: 0,
            align: align_log2,
            memory_index: 0,
        }
    }

    //
    pub fn pointer_gep(
        &mut self,
        ele_ty: &Ty,
        collection_expr: &Expr,
        index_expr: &Expr,
        dps_local_id: Option<u32>,
    ) {
        // address of pointer + index * sizeof ty
        // Actual order is (index * sizeof ty) + base pointer

        // Push index onto stack
        self.gen_expr(&index_expr, dps_local_id);
        // Push size onto the stack
        let (size, _align) = self.get_size_and_align(&ele_ty);
        //println!("calculated sizeof {:?} to be {}", ele_ty, size);
        self.emit_current(Instruction::I32Const(size as i32));
        // int mul
        self.emit_current(Instruction::I32Mul);
        // Push base pointer onto the stack
        self.gen_lvalue(&collection_expr);
        // add
        self.emit_current(Instruction::I32Add);
    }

    /// Load a pointer to a struct field onto the stack. Requires base pointer to be on the stack
    ///
    /// Set generic context before call. TODO: If actually required, this is dumb. Should set that here
    pub fn struct_gep(&mut self, struct_def: &StructDef, field_ident: &Ident, generics: &Vec<Ty>) {
        let Some(index_of_field) = struct_def
            .fields
            .iter()
            .position(|n| n.0.text == field_ident.text)
        else {
            let fields: Vec<&Ident> = struct_def.fields.iter().map(|f| &f.0).collect();
            panic!(
                "Could not find field {:?} in {:?}. Options are: {:?}",
                field_ident.text, struct_def.name.text, fields
            );
        };

        let struct_key = GenericInstTyKey {
            name: struct_def.name.text.clone(),
            tys: generics.clone(),
        };

        let offsets = if let Some((_size, offsets)) = self.struct_layouts.get(&struct_key) {
            offsets
        } else {
            // Resolve them with an ast2ty (which recurses inward)
            let field_tys: Vec<Ty> = struct_def
                .fields
                .iter()
                .map(|f| self.ast_to_ty(&f.1))
                .collect();

            let size_offsets: (usize, Vec<usize>) = self.layout_struct_ty(&field_tys);
            self.struct_layouts.insert(struct_key.clone(), size_offsets);

            &self.struct_layouts[&struct_key].1
        };

        let offset = offsets[index_of_field];

        self.emit_current(Instruction::I32Const(offset as i32));
        self.emit_current(Instruction::I32Add);
    }

    /// Destination, source, (num bytes)
    pub fn emit_memcpy(&mut self, size: usize) {
        self.emit_current(Instruction::I32Const(size as i32));
        self.emit_current(Instruction::MemoryCopy {
            src_mem: 0,
            dst_mem: 0,
        });
    }

    // TODO: Truncated / Euclidian / Floor
    // Current approach is not consistent - rem_s is truncated, this is floored
    // https://en.wikipedia.org/wiki/Modulo
    pub fn emit_rem_floor_f32(&mut self) {
        let local_x = self.add_local_current(wasm_encoder::ValType::F32, "modulo.temp.x");
        let local_y = self.add_local_current(wasm_encoder::ValType::F32, "modulo.temp.y");

        let local_div = self.add_local_current(wasm_encoder::ValType::F32, "modulo.temp.div");
        // Reverse order since we are unwinding the stack
        self.emit_current(Instruction::LocalSet(local_y));
        self.emit_current(Instruction::LocalTee(local_x));
        self.emit_current(Instruction::LocalGet(local_y));

        // rem_euclid(x, y) = y * fract(x / y)
        // fract -> x - floor(x)
        self.emit_current(Instruction::F32Div);

        // push 2 divs to stack
        self.emit_current(Instruction::LocalTee(local_div));
        self.emit_current(Instruction::LocalGet(local_div));

        // this consumes the first div
        self.emit_current(Instruction::F32Floor);

        // now with the other div on the stack, div - floor(div)
        // first onto the stack - second
        self.emit_current(Instruction::F32Sub);
        // fract result now on stack

        // load y
        self.emit_current(Instruction::LocalGet(local_y));
        // y * fract result
        self.emit_current(Instruction::F32Mul);
    }

    pub fn emit_rem_floor_f64(&mut self) {
        let local_x = self.add_local_current(wasm_encoder::ValType::F64, "modulo.temp.x");
        let local_y = self.add_local_current(wasm_encoder::ValType::F64, "modulo.temp.y");
        let local_div = self.add_local_current(wasm_encoder::ValType::F64, "modulo.temp.div");
        // Reverse order since we are unwinding the stack
        self.emit_current(Instruction::LocalSet(local_y));
        self.emit_current(Instruction::LocalTee(local_x));
        self.emit_current(Instruction::LocalGet(local_y));

        // rem_euclid(x, y) = y * fract(x / y)
        // fract -> x - floor(x)
        self.emit_current(Instruction::F64Div);

        // push 2 divs to stack
        self.emit_current(Instruction::LocalTee(local_div));
        self.emit_current(Instruction::LocalGet(local_div));

        // this consumes the first div
        self.emit_current(Instruction::F64Floor);

        // now with the other div on the stack, div - floor(div)
        // first onto the stack - second
        self.emit_current(Instruction::F64Sub);
        // fract result now on stack

        // load y
        self.emit_current(Instruction::LocalGet(local_y));
        // y * fract result
        self.emit_current(Instruction::F64Mul);
    }
}
