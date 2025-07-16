use hashbrown::HashMap;

use wasm_encoder::{Function, Instruction, ValType};

use crate::wasm_backend::{wasm_codegen::WasmModule, wasm_function::FunctionVariable};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlLabel {
    Loop,
    Block,
    NoBreak,
}

pub struct FunctionCtx<'a> {
    pub name: String,
    pub current_function_id: u32,
    pub current_function: FunctionBuilder<'a>,
    // Variables need a full context stack, as we don't want scoped definitions to preclude later definitions
    // locals passed in will always be basic types, including structs -> pointers
    pub local_variables: GenericContext<FunctionVariable>,
    // The index of the local that HOLDS the sret pointer
    pub(crate) sret_ptr_local: Option<u32>,
    // Need to manually track depth
    pub(crate) control_flow_stack: Vec<ControlLabel>,
}

impl<'a> FunctionCtx<'a> {
    pub fn new(
        name: String,
        param_types: Vec<ValType>,
        current_function_id: u32,
        module: &mut WasmModule,
    ) -> Self {
        Self {
            current_function_id,
            current_function: FunctionBuilder::new(
                name.to_owned(),
                param_types,
                module,
                current_function_id,
            ),
            local_variables: GenericContext::new(false),
            sret_ptr_local: None,
            name,
            control_flow_stack: vec![],
        }
    }

    pub fn find_label(&self, kind: ControlLabel) -> Option<u32> {
        self.control_flow_stack
            .iter()
            .rev()
            .position(|l| *l == kind)
            .map(|i| i as u32)
    }
}

pub struct FunctionBuilder<'a> {
    #[allow(unused)]
    name: String, // For debugging
    param_types: Vec<ValType>, // maybe keep this around for reference
    local_types: Vec<ValType>,
    pre_prologue: Vec<Instruction<'a>>,
    prologue: Vec<Instruction<'a>>,
    instructions: Vec<Instruction<'a>>,
    saved_stack_pointer: u32,

    raw_injection_map: HashMap<usize, Vec<u8>>,
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(
        name: String,
        param_types: Vec<ValType>,
        module: &mut WasmModule,
        current_function_id: u32,
    ) -> Self {
        let mut val = Self {
            name,
            local_types: param_types.clone(),
            param_types,
            pre_prologue: vec![],
            prologue: vec![],
            instructions: vec![],
            saved_stack_pointer: 0,

            raw_injection_map: HashMap::new(),
        };

        let stack_ptr_local = val.begin_stack_frame(module.stack_ptr_id);
        val.saved_stack_pointer = stack_ptr_local;

        // Register debug name
        module
            .dbg_name_bld
            .add_local(current_function_id, stack_ptr_local, "saved.stack.ptr");
        val
    }

    pub fn add_injection_mapping(&mut self, mut bytes: Vec<u8>) {
        let insert_before = self.instructions.len();

        // Push bytes after other inlined entries
        if let Some(entry) = self.raw_injection_map.get_mut(&insert_before) {
            entry.append(&mut bytes);
        } else {
            self.raw_injection_map.insert(insert_before, bytes);
        }
    }

    pub fn last_instruction(&self) -> Option<Instruction> {
        self.instructions.last().cloned()
    }

    pub fn emit(&mut self, instruction: Instruction<'a>) {
        self.instructions.push(instruction);
    }

    /// Reserved for starting the stack frame
    pub fn emit_pre_prologue(&mut self, instruction: Instruction<'a>) {
        self.pre_prologue.push(instruction);
    }

    pub fn emit_prologue(&mut self, instruction: Instruction<'a>) {
        self.prologue.push(instruction);
    }

    pub fn add_local(&mut self, ty: ValType) -> u32 {
        let idx = self.local_types.len();
        self.local_types.push(ty);
        idx as u32
    }

    pub fn begin_stack_frame(&mut self, stack_ptr_id: u32) -> u32 {
        let temp_local = self.add_local(ValType::I32);
        // Load the value of the global stack pointer
        self.emit_pre_prologue(Instruction::GlobalGet(stack_ptr_id));
        // Store it in the new local
        self.emit_pre_prologue(Instruction::LocalSet(temp_local));
        temp_local
    }

    pub fn end_stack_frame(&mut self, stack_ptr_id: u32) {
        // Load the local that holds the stack pointer to restore
        self.emit(Instruction::LocalGet(self.saved_stack_pointer));
        // Set the global stack pointer to it
        self.emit(Instruction::GlobalSet(stack_ptr_id));
    }

    pub fn into_encoder(mut self, stack_ptr_id: u32, static_data_boundry: usize) -> Function {
        let locals_only = &self.local_types[self.param_types.len()..];

        let locals = Self::encode_locals_grouped(locals_only);

        self.emit(Instruction::End);

        let mut func = Function::new(locals);

        // If the prologue is empty, don't need this
        for instruction in self.pre_prologue {
            func.instruction(&instruction);
        }

        for instruction in self.prologue {
            func.instruction(&instruction);
        }

        // emit overflow check
        // TODO: Only necessary in functions that need a stack frame
        func.instruction(&Instruction::GlobalGet(stack_ptr_id));
        func.instruction(&Instruction::I32Const(static_data_boundry as i32));
        func.instruction(&Instruction::I32LtU);
        func.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
        func.instruction(&Instruction::Unreachable);
        func.instruction(&Instruction::End);

        let mut body_instruction_idx = 0;
        for instruction in self.instructions {
            if let Some(bytes_entry) = self.raw_injection_map.remove(&body_instruction_idx) {
                func.raw(bytes_entry);
            }

            func.instruction(&instruction);
            body_instruction_idx += 1;
        }
        func
    }

    /// This was a huge pain to track down. Apparently it doesn't want a fully compressed map, just run length encoded
    fn encode_locals_grouped(locals_only: &[ValType]) -> Vec<(u32, ValType)> {
        let mut result = Vec::new();

        let mut iter = locals_only.iter().peekable();
        while let Some(&ty) = iter.next() {
            let mut count = 1;
            while let Some(&&next_ty) = iter.peek() {
                if next_ty == ty {
                    count += 1;
                    iter.next();
                } else {
                    break;
                }
            }
            result.push((count, ty));
        }

        result
    }
}

impl<'a> WasmModule<'a> {
    pub fn add_local_current(&mut self, ty: ValType, name: &str) -> u32 {
        let current = &mut self.function_ctx_stack.current_mut().current_function;
        let idx = current.add_local(ty);

        let current_function_idx = self.function_ctx_stack.current().current_function_id;
        self.dbg_name_bld.add_local(current_function_idx, idx, name);
        idx
    }

    pub fn emit_current(&mut self, instruction: Instruction<'a>) {
        let current = &mut self.function_ctx_stack.current_mut().current_function;
        //println!("[{}] Emit: {:?} to {:?}", count, &instruction, name);
        current.emit(instruction);
    }

    /// Reserved for stack allocations
    pub fn emit_current_prologue(&mut self, instruction: Instruction<'a>) {
        let current = &mut self.function_ctx_stack.current_mut().current_function;
        //println!("[{}] Emit: {:?} to {:?}", count, &instruction, name);
        current.emit_prologue(instruction);
    }
}

#[derive(Debug)]
pub struct GenericContext<T> {
    only_check_top: bool,
    context_stack: Vec<HashMap<String, T>>,
}

#[allow(dead_code)]
impl<T: std::fmt::Debug> GenericContext<T> {
    pub fn new(only_check_top: bool) -> Self {
        GenericContext {
            only_check_top,
            context_stack: vec![],
        }
    }

    pub fn has_key(&self, key: &str) -> bool {
        for map in self.context_stack.iter().rev() {
            if map.contains_key(key) {
                return true;
            }

            // break early
            if self.only_check_top {
                break;
            }
        }
        return false;
    }

    pub fn insert_in_last(&mut self, key: String, item: T) {
        let last = self
            .context_stack
            .last_mut()
            .expect("Context Stack is Empty!");
        last.insert(key, item);
    }

    pub fn remove_from_last(&mut self, key: &str) -> Option<T> {
        let last = self
            .context_stack
            .last_mut()
            .expect("Context Stack is Empty!");
        last.remove(key)
    }

    pub fn push(&mut self, ctx: HashMap<String, T>) {
        self.context_stack.push(ctx);
    }

    pub fn pop(&mut self) -> Option<HashMap<String, T>> {
        self.context_stack.pop()
    }

    pub fn has_context(&self) -> bool {
        !self.context_stack.is_empty()
    }

    pub fn try_get_mapping(&self, key: &str) -> Option<&T> {
        for map in self.context_stack.iter().rev() {
            if map.contains_key(key) {
                return Some(&map[key]);
            }

            if self.only_check_top {
                break;
            }
        }

        return None;
    }

    pub fn get_mapping(&self, key: &str) -> &T {
        for map in self.context_stack.iter().rev() {
            if map.contains_key(key) {
                return &map[key];
            }
        }

        panic!(
            "Mapping for generic {} not found in context stack ({:#?})",
            key, self.context_stack
        );
    }
}
