use wasm_encoder::{ConstExpr, Instruction, ValType};

use crate::{
    frontend::sema::Ty,
    wasm_backend::{
        wasm_codegen::WasmModule, wasm_debug::get_var_dbg_name, wasm_mem::VIRTUAL_STACK_SIZE,
    },
};

impl<'a> WasmModule<'a> {
    /// Create a fixed size stack region for the program to use
    pub(crate) fn init_stack(&mut self) {
        // Set the stack pointer to the end of the heap + stack size
        let stack_base = self.get_stack_base();
        let initial_stack = stack_base + VIRTUAL_STACK_SIZE;
        assert!(
            initial_stack <= u32::MAX as usize,
            "Address exceeds 32-bit memory range"
        );

        let stack_id = self.new_global(
            ValType::I32,
            &ConstExpr::i32_const(initial_stack as i32),
            true,
            "global.stack",
        );
        self.stack_ptr_id = stack_id;

        self.dbg_name_bld.add_global(stack_id, "global.stack");
    }

    pub(crate) fn stack_alloc(&mut self, ty: &Ty, alloc_name: &str) -> u32 {
        assert!(
            !matches!(ty, Ty::NullPtr(_)),
            "pointers should always be typed in this context"
        );

        let (size, align) = self.get_size_and_align(ty);
        let alloc_size = self.align_to(size, align);

        let stack_ptr_id = self.stack_ptr_id;

        let local_ty = self.wasm_param_type(ty);

        // Create a local to hold the pointer to the allocated stack space
        let dbg_name = get_var_dbg_name(alloc_name, ty);
        let ptr_local = self.add_local_current(local_ty, &dbg_name);

        // Load the value of this global onto the stack
        self.emit_current_prologue(Instruction::GlobalGet(stack_ptr_id));

        // Load alloc size onto the stack
        self.emit_current_prologue(Instruction::I32Const(alloc_size as i32));

        // Subtracts the last loaded from the previous last loaded (setting the top of the stack to the result)
        self.emit_current_prologue(Instruction::I32Sub);

        // Align fix attempt
        // let align_mask = !(align - 1);
        // self.emit_current(Instruction::I32Const(align_mask as i32));
        // self.emit_current(Instruction::I32And);
        //

        // Sets the value of a local (ptr_local) to the value at the top of the stack, and keeps that value at the top
        self.emit_current_prologue(Instruction::LocalTee(ptr_local));

        // Sets the value of the stack ptr global to address of the top of the (VM's) stack
        self.emit_current_prologue(Instruction::GlobalSet(stack_ptr_id));

        ptr_local
    }
}
