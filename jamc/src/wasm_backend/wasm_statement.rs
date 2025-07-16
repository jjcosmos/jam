use hashbrown::HashMap;
use wasm_encoder::{BlockType, Instruction};

use crate::{
    frontend::{
        ast::{Expr, Ident, Stmt, Type},
        sema::Ty,
    },
    wasm_backend::{
        function_ctx::ControlLabel,
        wasm_codegen::WasmModule,
        wasm_debug::get_var_dbg_name,
        wasm_expression::{GenValResult, ValueKind},
        wasm_function::FunctionVariable,
    },
};

impl<'a> WasmModule<'a> {
    pub fn gen_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::ExprStmt(expr) => {
                let val = self.gen_expr(expr, None);

                // For a balanced stack, we need to drop unused values
                if val.ty != Ty::Void {
                    match val.kind {
                        ValueKind::RValue => {
                            self.emit_current(Instruction::Drop);
                        }
                        ValueKind::SretRValue => {}
                        _ => {
                            panic!("got lvalue from expression statement")
                        }
                    }
                }
            }
            Stmt::Let(ident, decl_ty, opt_expr) => self.gen_let_stmt(ident, decl_ty, opt_expr),
            Stmt::Assign(expr_a, expr_b) => self.gen_assign_stmt(expr_a, expr_b),
            // Stmt::If(expr, body_block, opt_else_block) => {
            //     self.with_new_variable_ctx(|module| {
            //         module.gen_if_stmt(expr, body_block, opt_else_block.as_ref())
            //     });
            // }
            Stmt::While(expr, block) => {
                self.with_new_variable_ctx(|module| {
                    module.gen_while_stmt(expr, block);
                });
            }
            Stmt::For(opt_init_stmt, opt_condit_expr, opt_post_loop_stmt, block) => {
                self.with_new_variable_ctx(|module| {
                    module.gen_for_stmt(opt_init_stmt, opt_condit_expr, opt_post_loop_stmt, block);
                });
            }
            Stmt::Return(expr) => {
                self.gen_ret_stmt(expr);
            }
            // Stmt::Block(stmts) => {
            //     // Block should have a local scope
            //     self.with_new_variable_ctx(|module| {
            //         // Do stuff
            //         for stmt in stmts {
            //             module.gen_stmt(stmt);
            //         }
            //     });
            // }
            Stmt::Break => {
                let block_label = self
                    .function_ctx_stack
                    .current()
                    .find_label(ControlLabel::Block)
                    .expect("missing block");
                self.emit_current(Instruction::Br(block_label));
            }
            Stmt::Continue => {
                let loop_label = self
                    .function_ctx_stack
                    .current()
                    .find_label(ControlLabel::Loop)
                    .expect("missing loop");
                self.emit_current(Instruction::Br(loop_label));
            }
        }
    }

    pub(crate) fn with_new_variable_ctx(&mut self, func: impl FnOnce(&mut Self)) {
        self.function_ctx_stack
            .current_mut()
            .local_variables
            .push(HashMap::new());
        func(self);
        self.function_ctx_stack.current_mut().local_variables.pop();
    }

    pub(crate) fn gen_while_stmt(&mut self, expr: &Expr, block: &Expr) {
        // Branching to a loop moves execution to the start of the loop, while branching to a block branches to the end of the block
        // Create an outer 'label' this loop can break to
        self.emit_current(Instruction::Block(BlockType::Empty));
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::Block);

        // Create loop block
        self.emit_current(Instruction::Loop(BlockType::Empty));
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::Loop);

        // push conditional expression result to the stack
        self.gen_expr(expr, None);
        // Check if result is zero
        self.emit_current(Instruction::I32Eqz);
        // branch to OUTER if false, as a 'break'

        let block_label = self
            .function_ctx_stack
            .current()
            .find_label(ControlLabel::Block)
            .expect("missing block");
        self.emit_current(Instruction::BrIf(block_label));

        // Loop body
        // for stmt in block {
        //     self.gen_stmt(stmt);
        // }
        let res_ty = self.gen_expr(block, None);
        assert!(
            res_ty.ty == Ty::Void,
            "type of while block expression must be void"
        );

        // Unconditional branch to loop start
        let loop_label = self
            .function_ctx_stack
            .current()
            .find_label(ControlLabel::Loop)
            .expect("missing loop");
        self.emit_current(Instruction::Br(loop_label));

        // End loop block
        self.emit_current(Instruction::End);
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();

        // End outer block
        self.emit_current(Instruction::End);
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();
    }

    pub(crate) fn gen_for_stmt(
        &mut self,
        opt_init_stmt: &Box<Option<Stmt>>,
        opt_condit_expr: &Option<Expr>,
        opt_post_loop_stmt: &Box<Option<Stmt>>,
        block: &Expr,
    ) {
        // generate initial let
        if let Some(init) = &**opt_init_stmt {
            self.gen_stmt(init);
        }

        // Create outer block
        self.emit_current(Instruction::Block(BlockType::Empty));
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::Block);
        // start loop
        self.emit_current(Instruction::Loop(BlockType::Empty));
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::Loop);
        // check condition, break if false
        if let Some(condit) = opt_condit_expr {
            // push result to stack
            self.gen_expr(condit, None);
            // Check if == 0
            self.emit_current(Instruction::I32Eqz);

            let block_label = self
                .function_ctx_stack
                .current()
                .find_label(ControlLabel::Block)
                .expect("missing block");
            self.emit_current(Instruction::BrIf(block_label));
        }
        // body
        // for stmt in block {
        //     self.gen_stmt(stmt);
        // }
        let res_ty = self.gen_expr(block, None);
        assert!(
            res_ty.ty == Ty::Void,
            "type of for block expression must be void"
        );

        // post loop stmt
        if let Some(post_loop_stmt) = &**opt_post_loop_stmt {
            self.gen_stmt(post_loop_stmt);
        }

        // Unconditional branch to loop start
        let loop_label = self
            .function_ctx_stack
            .current()
            .find_label(ControlLabel::Loop)
            .expect("missing loop");
        self.emit_current(Instruction::Br(loop_label));

        // End loop block
        self.emit_current(Instruction::End);
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();

        // End outer block
        self.emit_current(Instruction::End);
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();
    }

    fn gen_ret_stmt(&mut self, expr_opt: &Option<Expr>) {
        if let Some(expr) = expr_opt {
            let _sema_expr_ty = self.sema_expr_ty(&expr.id);
            if let Some(sret_ptr_local) = self.function_ctx_stack.current().sret_ptr_local {
                // Push the destination pointer onto the stack
                self.emit_current(Instruction::LocalGet(sret_ptr_local));
                // memcpy the result of the expression into the sret ptr's mem
                //self.emit_copy_from_expr(expr, &sema_expr_ty);
                // don't need to copy if the expression "fills" sret ptr
                self.gen_expr(expr, Some(sret_ptr_local));
            } else {
                // Push the result of the expression to the stack
                self.gen_expr(expr, None);
            }
        }
        let stack_ptr_id = self.stack_ptr_id;
        self.function_ctx_stack
            .current_mut()
            .current_function
            .end_stack_frame(stack_ptr_id);
        // Result of the expression should be on the stack (if there is one)
        self.emit_current(Instruction::Return);
    }

    fn gen_assign_stmt(&mut self, lhs: &Expr, rhs: &Expr) {
        let lhs_ty = self.sema_expr_ty(&lhs.id);
        let _rhs_ty = self.sema_expr_ty(&rhs.id);

        // LHS must be a pointer lvalue
        let gen_lval_result = self.gen_lvalue(lhs);
        if let GenValResult {
            ty: Ty::Ptr(_inner_ty),
            kind: ValueKind::LValuePtr,
        } = gen_lval_result
        {
            // Instead can just store in the memory that the lhs occupies
            let dps_local_id = self.add_local_current(
                wasm_encoder::ValType::I32,
                &get_var_dbg_name("local.dps.assign", &lhs_ty),
            );
            // Set the dps pointer to the result of gen_lvalue
            self.emit_current(Instruction::LocalSet(dps_local_id));
            // Generate an expression that writes to the memory in lhs
            self.gen_expr(rhs, Some(dps_local_id));
        } else {
            // scalar assignment, but, can (and should) still pass a dps pointer
            // Or just gen expression + local store.. ? That would be a lot easier

            // Position to the stack
            let GenValResult {
                ty: _,
                kind: ValueKind::LValueLocalId(lval_local_id),
            } = gen_lval_result
            // The pointer to store in, pushed to the stack
            else {
                panic!(
                    "Cannot assign to non-lvalue {:?} {:?}",
                    gen_lval_result, lhs_ty
                );
            };

            // The value to store, pushed to the stack
            let _g_expr_result = self.gen_expr(rhs, None);

            // TODO new: Check this
            // gen_expr with none, the value should be on the stack
            self.emit_current(Instruction::LocalSet(lval_local_id));
        }
    }

    pub fn gen_let_stmt(&mut self, ident: &Ident, decl_ty: &Type, opt_expr: &Option<Expr>) {
        // Decl type should have priority. This allows typing nullptrs
        let init_ty = if !matches!(decl_ty, Type::Inferred(_)) {
            self.ast_to_ty(decl_ty)
        } else {
            let Some(init_expr) = opt_expr else {
                panic!("type cannot be inferred without init expression");
            };
            self.sema_expr_ty(&init_expr.id)
        };

        // The way this is done prevents shadowing
        let needs_stack_loc = if let Some(vec) = self
            .addr_of_uses_by_function
            .get(&self.function_ctx_stack.current().name)
        {
            vec.contains(&ident.text)
        } else {
            false
        };

        let mem_loc_and_ty = if init_ty.is_composite_ty() || needs_stack_loc {
            // uses stack memory, alloca and pass dps as stack mem
            let local_for_ptr = self.stack_alloc(&init_ty, &ident.text);
            self.function_ctx_stack
                .current_mut()
                .local_variables
                .insert_in_last(
                    ident.text.to_owned(),
                    FunctionVariable::PtrLocal(local_for_ptr, true), // TODO: "let mut"
                );
            *self
                .function_ctx_stack
                .current()
                .local_variables
                .get_mapping(&ident.text)
        } else {
            // use a local, and pass as local value kind. Should probably update emit_store to take the valuekind
            // need to make sure the value is correctly padded and such
            let local = self.add_local_current(
                self.wasm_param_type(&init_ty),
                &format!("local.let.{}", ident.text),
            );
            self.function_ctx_stack
                .current_mut()
                .local_variables
                .insert_in_last(ident.text.to_owned(), FunctionVariable::Local(local, true));
            *self
                .function_ctx_stack
                .current()
                .local_variables
                .get_mapping(&ident.text)
        };

        if let Some(init_expr) = opt_expr {
            match mem_loc_and_ty {
                FunctionVariable::PtrLocal(ptr_local_id, _mutable) => {
                    // gen expression, storing the result in the dps
                    self.gen_expr(init_expr, Some(ptr_local_id));
                }
                FunctionVariable::Local(local_to_data_id, _mutable) => {
                    // Push result of expression to stack
                    self.gen_expr(init_expr, None);
                    // Store it in the local created earlier
                    self.emit_current(Instruction::LocalSet(local_to_data_id));
                }
            }
        } else {
            match mem_loc_and_ty {
                // For stack pointers, need to zero fill memory
                FunctionVariable::PtrLocal(local_for_ptr, _mutable) => {
                    // Zero fill memory
                    let (size, _align) = self.get_size_and_align(&init_ty);
                    // load the pointer from the local
                    self.emit_current(Instruction::LocalGet(local_for_ptr));
                    // push the byte value
                    self.emit_current(Instruction::I32Const(0));
                    // push the number of bytes
                    self.emit_current(Instruction::I32Const(size as i32));
                    // emit fill instruction
                    self.emit_current(Instruction::MemoryFill(0));
                }
                // Locals are already zero initialized, I think?
                FunctionVariable::Local(_, _) => {}
            }
        }
    }
}
