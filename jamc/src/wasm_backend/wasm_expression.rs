use hashbrown::HashMap;

use wasm_encoder::{BlockType, Instruction, ValType};

use crate::{
    frontend::{
        ast::{
            Expr, ExprKind, FunctionReference, Ident, Literal, MatchArm, MatchExpression,
            ScopedIdent, Type, UnOp,
        },
        sema::Ty,
    },
    wasm_backend::{
        function_ctx::ControlLabel,
        wasm_codegen::WasmModule,
        wasm_function::{FunctionVariable, GenericInstTyKeyRef},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueKind {
    RValue,
    LValuePtr, // l-value (ptr)
    // Note: this should be the case with ident lvalues (for simple values) and function args
    // The value is directly in this local
    LValueLocalId(u32), // l-value (local)
    SretRValue,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GenValResult {
    pub(crate) ty: Ty,
    pub(crate) kind: ValueKind,
}

impl<'a> WasmModule<'a> {
    /// Generate an r-value
    pub(crate) fn gen_expr(&mut self, base_expr: &Expr, dps_local_id: Option<u32>) -> GenValResult {
        match &base_expr.kind {
            crate::frontend::ast::ExprKind::Literal(literal) => {
                self.gen_literal(base_expr, literal, dps_local_id)
            }
            crate::frontend::ast::ExprKind::ScopedIdent(scoped_ident) => {
                self.gen_scoped_ident(base_expr, scoped_ident, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Binary(expr_a, bin_op, expr_b) => {
                self.gen_binary(base_expr, expr_a, bin_op, expr_b, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Unary(un_op, expr) => {
                self.gen_unary(base_expr, un_op, expr, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Call(callee_expr, args, generics) => {
                self.gen_call(base_expr, callee_expr, args, generics, dps_local_id)
            }
            crate::frontend::ast::ExprKind::MethodCall {
                receiver: reciever,
                method_name,
                args,
                generic_args,
            } => self.gen_method_call(
                base_expr,
                reciever,
                method_name,
                args,
                generic_args,
                dps_local_id,
            ),
            crate::frontend::ast::ExprKind::Cast(expr, target_ty) => {
                self.gen_cast(base_expr, expr, target_ty)
            }
            crate::frontend::ast::ExprKind::StructLit {
                name,
                fields,
                generic_inst_tys,
            } => self.gen_struct_lit(base_expr, name, fields, generic_inst_tys, dps_local_id),
            crate::frontend::ast::ExprKind::UnionLit(
                base_path,
                variant_ident,
                init_epxr_opt,
                generic_inst_tys,
            ) => self.gen_union_lit(
                base_expr,
                base_path,
                variant_ident,
                init_epxr_opt.as_ref(),
                generic_inst_tys,
                dps_local_id,
            ),
            crate::frontend::ast::ExprKind::ArrayLit(exprs) => {
                self.gen_array_lit(base_expr, exprs, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Index(indexee, indexer) => {
                self.gen_index(base_expr, indexee, indexer, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Field(target_val, field_ident, _) => {
                self.gen_field(base_expr, target_val, field_ident, dps_local_id)
            }
            crate::frontend::ast::ExprKind::FunctionReference(function_reference) => {
                self.gen_function_reference(base_expr, function_reference, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Match(decomp_expr) => {
                //todo!("decompose expression aren't implemented yet")
                self.gen_match(base_expr, decomp_expr, dps_local_id)
            }
            crate::frontend::ast::ExprKind::Block(statements, final_expression) => {
                for stmt in statements {
                    self.gen_stmt(stmt);
                }

                if let Some(tail) = final_expression {
                    let ty = &self.sema_expr_ty(&base_expr.id);
                    assert!(
                        *ty != Ty::Void,
                        "`void` returning tail expressions not supported. Are you missing a semicolon?"
                    );

                    // // If part of an rvalue aggregate, need to allocate stack space
                    // let alloca = if let Some(d) = dps_local_id {
                    //     d
                    // } else {
                    //     self.stack_alloc(ty, &format!("block_expr_alloc"))
                    // };

                    return self.gen_expr(&tail, dps_local_id);
                }

                return GenValResult {
                    ty: Ty::Void,
                    kind: ValueKind::RValue,
                };
            }
            crate::frontend::ast::ExprKind::If(condition, then_expr, opt_else_expr) => {
                self.gen_if_expr(base_expr, condition, then_expr, opt_else_expr, dps_local_id)
            }
            crate::frontend::ast::ExprKind::InlineWat(captured_locals, type_opt, wasm) => {
                self.gen_inline_wat(base_expr, captured_locals, type_opt, wasm, dps_local_id)
            }
        }
    }

    pub(crate) fn gen_match(
        &mut self,
        base_expr: &Expr,
        decomp_expr: &MatchExpression,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // generate base expression to match on, pushing the result to the stack
        let union_expr_result = if decomp_expr.as_ref {
            self.gen_lvalue(&decomp_expr.union_to_decomp)
        } else {
            self.gen_expr(&decomp_expr.union_to_decomp, None)
        };

        let base_addr_local = self.add_local_current(wasm_encoder::ValType::I32, "union.base.temp");
        self.emit_current(Instruction::LocalSet(base_addr_local));

        // We expect a pointer to a union here
        assert!(
            matches!(union_expr_result.ty, Ty::Ptr(ref inner) if matches!(&**inner, Ty::Union(_, _))),
            "expected union type, got {:?}",
            union_expr_result.ty
        );

        let union_expr_ty = self.sema_expr_ty(&decomp_expr.union_to_decomp.id);
        // Discrim will only be valid from literal expressions - can't use to match on
        let Ty::Union(ident, generics) = union_expr_ty else {
            panic!("expected union type for decomp expression target")
        };

        let union_def = *self.union_defs.get(&ident).expect("failed to lookup union");
        let ctx = self.create_map_from_generic_tys(&generics, &union_def.generics);
        self.generic_context.push(ctx);

        let variant_count = union_def.variants.len();

        let mut arms_ref_ordered: Vec<&MatchArm> = decomp_expr.arms.iter().map(|i| i).collect();
        arms_ref_ordered.sort_by(|a, b| {
            let i_of_a: usize = union_def.index_of(&a.variant_ident);
            let i_of_b = union_def.index_of(&b.variant_ident);
            i_of_a.cmp(&i_of_b)
        });

        // -------------------------------- //
        let variant_tys: Vec<Ty> = union_def
            .variants
            .iter()
            .map(|v| self.ast_to_ty(&v.1))
            .collect();
        let variant_offset = self.variant_offset(&variant_tys);

        let local_ptr_to_variant =
            self.add_local_current(wasm_encoder::ValType::I32, "ptr.arm.temp");
        // ------------------------------ //

        // Open default block (last in order, so first here)
        self.emit_current(Instruction::Block(BlockType::Empty));
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::Block);

        // Jump expressions go to the end of the block
        // Initialize blocks last to first
        for _arm in arms_ref_ordered.iter().rev() {
            self.emit_current(Instruction::Block(BlockType::Empty));

            self.function_ctx_stack
                .current_mut()
                .control_flow_stack
                .push(ControlLabel::Block);
        }

        // Create Dispatch block
        self.emit_current(Instruction::Block(BlockType::Empty));

        // Load pointer from gen_expr result
        // Load the address of the union as an int (since I think discrim is at offset 0, should load discrim)
        self.emit_current(Instruction::LocalGet(base_addr_local));
        self.emit_load_from_ptr(&Ty::I32);

        // for 0 - discrim max
        // if arms contains idx, add idx to array, otherwise, add default
        let mut targets = vec![];
        let index_of_default = arms_ref_ordered.len(); // indices up to arm ref max + default
        for n in 0..variant_count {
            if let Some(arm_index) = arms_ref_ordered
                .iter()
                .position(|arm| union_def.index_of(&arm.variant_ident) == n)
            {
                targets.push(arm_index as u32);
            } else {
                targets.push(index_of_default as u32);
            }
        }

        self.emit_current(Instruction::BrTable(
            std::borrow::Cow::Owned(targets),
            (index_of_default) as u32,
        ));
        // Shouldn't need an unreachable - br_table ALWAYS branches
        //self.emit_current(Instruction::Unreachable);
        self.emit_current(Instruction::End);
        // --------------------------------------------------------------- //

        // Allocate temp for result
        let expression_ty = self.sema_expr_ty(&base_expr.id);
        let expression_dps_local = if let Some(dps) = dps_local_id {
            Some(dps)
        } else {
            if expression_ty.is_composite_ty() {
                Some(self.stack_alloc(&expression_ty, "expression.dps.local.alloc"))
            } else if expression_ty != Ty::Void {
                Some(self.add_local_current(
                    self.wasm_param_type(&expression_ty),
                    "expression.dps.local",
                ))
            } else {
                None
            }
        };

        // emit blocks first to last
        let mut depth_counter = arms_ref_ordered.len();
        for arm in arms_ref_ordered.iter() {
            // ---------------- Gen body ---------------------- //
            // Create a local variable context for the arm
            self.function_ctx_stack
                .current_mut()
                .local_variables
                .push(HashMap::new());

            // let binding = (point to variant from union memory as that type)
            if let Some(binding) = &arm.binding {
                // get base addr of the union
                self.emit_current(Instruction::LocalGet(base_addr_local));
                // push variant offset
                self.emit_current(Instruction::I32Const(variant_offset as i32));
                // add to get pointer to variant
                self.emit_current(Instruction::I32Add);

                // Set localptrtovariant to the pointer to the variant
                self.emit_current(Instruction::LocalSet(local_ptr_to_variant as u32));

                // register a binding as a local var. ensure this is cleaned up later
                self.function_ctx_stack
                    .current_mut()
                    .local_variables
                    .insert_in_last(
                        binding.text.to_owned(),
                        FunctionVariable::PtrLocal(local_ptr_to_variant, true),
                    );
            }

            // Gen body, then branch
            self.gen_expr(&arm.body_expression, dps_local_id);

            // Capture / consume the return value of the expression
            if let Some(expr_dps) = expression_dps_local
                && dps_local_id.is_none()
            {
                self.emit_current(Instruction::LocalSet(expr_dps));
            }

            // Arm count (excluding default) - idx
            let arm_break_target = depth_counter;
            depth_counter -= 1;
            self.emit_current(Instruction::Br(arm_break_target as u32));
            self.emit_current(Instruction::End);

            // Clean up contexts
            self.function_ctx_stack
                .current_mut()
                .control_flow_stack
                .pop();

            // Remove temp binding
            if let Some(binding) = &arm.binding {
                let res = self
                    .function_ctx_stack
                    .current_mut()
                    .local_variables
                    .remove_from_last(&binding.text);
                assert!(
                    res.is_some(),
                    "failed to find binding {} to remove",
                    binding.text
                );
            }

            // Need to remove local variable context
            self.function_ctx_stack.current_mut().local_variables.pop();

            // ------------------------- end gen body --------------------------- //
        }

        // default block. If there is none defined, should be unreachable / trap
        if let Some(default) = &decomp_expr.default_arm {
            self.gen_expr(&default.body_expression, dps_local_id);

            // Capture / consume the return value of the expression
            if let Some(expr_dps) = expression_dps_local
                && dps_local_id.is_none()
            {
                self.emit_current(Instruction::LocalSet(expr_dps));
            }
            // Don't need to emit a break here, it should already fall through this last case
        } else {
            self.emit_current(Instruction::Unreachable);
        }

        self.emit_current(Instruction::End);
        // Pop match arms block
        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();

        let expr_result_ty = self.sema_expr_ty(&base_expr.id);

        self.generic_context.pop();

        // Need to re-emit temp value if there was no dpslocal
        if let Some(expr_dps) = expression_dps_local
            && dps_local_id.is_none()
        {
            self.emit_current(Instruction::LocalGet(expr_dps));
        }

        GenValResult {
            ty: if expr_result_ty.is_composite_ty() {
                Ty::Ptr(Box::new(expr_result_ty))
            } else {
                expr_result_ty
            },
            kind: ValueKind::RValue,
        }
    }

    pub(crate) fn gen_if_expr(
        &mut self,
        base_expr: &Expr,
        condition: &Expr,
        then_expr: &Expr,
        opt_else_expr: &Option<Box<Expr>>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // If part of an rvalue aggregate, need to allocate stack space
        let eval_ty = self.sema_expr_ty(&base_expr.id);

        // The expression result ty, not the expressions ty
        let mut ret_result_ty = Ty::Void;

        // No dps, want the result on the stack
        self.gen_expr(condition, None);

        // Is empty the correct block type here?
        self.emit_current(Instruction::If(if eval_ty == Ty::Void {
            BlockType::Empty
        } else {
            BlockType::Result(self.wasm_param_type(&eval_ty))
        }));

        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .push(ControlLabel::NoBreak);

        let then_res = self.gen_expr(then_expr, dps_local_id);
        if then_res.ty != Ty::Void {
            ret_result_ty = then_res.ty;
        }

        self.function_ctx_stack
            .current_mut()
            .control_flow_stack
            .pop();

        if let Some(else_block) = opt_else_expr {
            self.function_ctx_stack
                .current_mut()
                .control_flow_stack
                .push(ControlLabel::NoBreak);

            self.emit_current(Instruction::Else);

            let else_res = self.gen_expr(&else_block, dps_local_id);
            if else_res.ty != Ty::Void {
                ret_result_ty = else_res.ty;
            }

            self.function_ctx_stack
                .current_mut()
                .control_flow_stack
                .pop();
        }

        self.emit_current(Instruction::End);

        return GenValResult {
            ty: ret_result_ty,
            kind: ValueKind::RValue,
        };
    }

    fn gen_function_reference(
        &mut self,
        base_expr: &Expr,
        function_reference: &FunctionReference,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        if let Some(dps) = dps_local_id {
            self.emit_current(Instruction::LocalGet(dps));
        }

        let tys: Vec<Ty> = function_reference
            .generic_args
            .iter()
            .map(|a| self.ast_to_ty(a))
            .collect();
        let key = GenericInstTyKeyRef {
            name: &function_reference.path.to_string(),
            tys: &tys,
        };

        // get the function id / index
        let val = self.get_or_define_function(&key);
        // Push the index of the function in the table to the stack
        // TODO: map or something
        let index_in_tables = self
            .table_entries_to_write
            .iter()
            .position(|entry| entry.func_index == val)
            .expect("failed to get index in tables");
        self.emit_current(Instruction::I32Const(index_in_tables as i32));

        let Ty::FuncPtr(param_tys, ret_ty) = self.sema_expr_ty(&base_expr.id) else {
            panic!("expected type of function pointer");
        };

        let params = param_tys;
        let ret = ret_ty;

        if let Some(_dps) = dps_local_id {
            self.emit_load_from_ptr(&Ty::U32);
        }

        return GenValResult {
            ty: Ty::FuncPtr(params, ret),
            kind: ValueKind::RValue,
        };
    }

    fn gen_literal(
        &mut self,
        base_expr: &Expr,
        literal: &Literal,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // Push pointer for store
        if let Some(dps) = dps_local_id {
            self.emit_current(Instruction::LocalGet(dps));
        }

        let sema_ty = self.sema_expr_ty(&base_expr.id);
        let result = match literal {
            Literal::Int(value, width, _is_signed) => {
                let instruction = if *width > 32 {
                    Instruction::I64Const(*value as i64)
                } else {
                    Instruction::I32Const(*value as i32)
                };

                self.emit_current(instruction);
                GenValResult {
                    ty: sema_ty,
                    kind: ValueKind::RValue,
                }
            }
            Literal::USize(value) => {
                self.emit_current(Instruction::I32Const(*value as i32));

                GenValResult {
                    ty: sema_ty,
                    kind: ValueKind::RValue,
                }
            }
            Literal::Float(value, is_32_bit) => {
                let instruction = if *is_32_bit {
                    Instruction::F32Const((*value as f32).into())
                } else {
                    Instruction::F64Const((*value).into())
                };

                self.emit_current(instruction);

                GenValResult {
                    ty: sema_ty,
                    kind: ValueKind::RValue,
                }
            }
            Literal::Bool(value) => {
                self.emit_current(Instruction::I32Const(if *value { 1 } else { 0 }));

                GenValResult {
                    ty: sema_ty,
                    kind: ValueKind::RValue,
                }
            }
            Literal::Str(cstring) => {
                let offset = self.data_builder.add_string(&cstring);

                // Push the offset to the stack
                let instruction = Instruction::I32Const(offset as i32);

                self.emit_current(instruction);

                GenValResult {
                    ty: Ty::Ptr(Box::new(Ty::U8)), // I think the initial sema would actually be the same
                    kind: ValueKind::RValue,
                }
            }
            Literal::Null(inner_type) => {
                // Should resolve generics
                let inner_sema = self.ast_to_ty(inner_type);
                self.emit_current(Instruction::I32Const(0));
                GenValResult {
                    ty: Ty::Ptr(Box::new(inner_sema)), // I think the initial sema would actually be the same
                    kind: ValueKind::RValue,
                }
            }
            Literal::SizeOf(inner_type) => {
                let inner_sema = self.ast_to_ty(inner_type);
                //println!("Sizeof ty {:?}", inner_sema);
                let (size, _align) = self.get_size_and_align(&inner_sema);
                let instruction = Instruction::I32Const(size as i32);

                self.emit_current(instruction);

                GenValResult {
                    ty: sema_ty,
                    kind: ValueKind::RValue,
                }
            }
        };

        // Store if dps
        if let Some(_dps_local) = dps_local_id {
            let expr_ty = self.sema_expr_ty(&base_expr.id);
            self.emit_store_to_ptr(&expr_ty);
        }

        return result;
    }

    /// Local vars, functions, globals as RVALUES
    fn gen_scoped_ident(
        &mut self,
        base_expr: &Expr,
        scoped_ident: &ScopedIdent,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        let sema_ty = self.sema_expr_ty(&base_expr.id);

        // Compound types would return a pointer? In let / assign, these need to be memcopied
        // Probably shoudn't be done here - just return a pointer to the memory
        // as an r value they do need to be copied - at least until I can guerentee that it won't be written to
        // otherwise more of a global than a const. I guess that's ok though

        // Enum var lits
        if let Some(val) = scoped_ident
            .scopes
            .last()
            .and_then(|last| self.enum_variant_map.get(last).map(|e| (last, e)))
            .and_then(|f| {
                Some(
                    f.1.get(&scoped_ident.ident.text)
                        .expect("Failed to unwrap enum val")
                        .clone(),
                )
            })
        {
            if let Some(dps) = dps_local_id {
                self.emit_current(Instruction::LocalGet(dps));
                self.emit_current(Instruction::I32Const(val as i32));
                self.emit_store_to_ptr(&Ty::I32);
            } else {
                self.emit_current(Instruction::I32Const(val as i32));
            }

            return GenValResult {
                ty: Ty::I32,
                kind: ValueKind::RValue,
            };
        }

        if sema_ty.is_composite_ty() {
            if let Some(global) = self
                .global_constants
                .get(&scoped_ident.to_string())
                .cloned()
                .or_else(|| self.global_statics.get(&scoped_ident.to_string()).cloned())
            {
                let alloc_id = if let Some(a) = dps_local_id {
                    Some(a)
                } else {
                    if sema_ty.is_composite_ty() {
                        let alloc = self.stack_alloc(
                            &sema_ty,
                            &format!("sc_ident.temp.{}", scoped_ident.to_string()),
                        );
                        Some(alloc)
                    } else {
                        None
                    }
                };

                let Some(dps_id) = alloc_id else {
                    panic!("memory types need stack allocation")
                };

                // emit dest
                self.emit_current(Instruction::LocalGet(dps_id));
                // emit source

                // Load the pointer to the constant onto the stack
                // I don't currently enforce constants, so this could be mutated
                // But ideally this doesn't need a copy
                match global.ptr {
                    super::wasm_globals::GlobalPtr::Memory(ptr) => {
                        self.emit_current(Instruction::I32Const(ptr as i32));
                    }
                    super::wasm_globals::GlobalPtr::WasmGlobal(_global_id) => {
                        panic!("composite types cannot be stored in globals!")
                    }
                }

                let (size, _align) = self.get_size_and_align(&sema_ty);
                self.emit_memcpy(size);

                // Need to push result to stack if not using dps
                if dps_local_id.is_none() {
                    self.emit_current(Instruction::LocalGet(dps_id));
                }

                return GenValResult {
                    // Globals / statics only store their semantic type. Since this is a composite ty, the value on the stack is a pointer
                    ty: Ty::Ptr(Box::new(global.ty)),
                    kind: ValueKind::RValue,
                };
            }

            // If it is a compound type and not a global, it would need to be a local var
            match *self
                .function_ctx_stack
                .current()
                .local_variables
                .get_mapping(&scoped_ident.to_string())
            {
                FunctionVariable::PtrLocal(ptr_local_id, _mutable) => {
                    // Need to make a new alloca and copy contents into it
                    // This is only the case because we know it is a struct/array (mem only type) here
                    let alloca_id = if let Some(d) = dps_local_id {
                        d
                    } else {
                        self.stack_alloc(
                            &sema_ty,
                            &format!("ident.copy.{}", scoped_ident.to_string()),
                        )
                    };
                    // Push the destination onto the stack
                    self.emit_current(Instruction::LocalGet(alloca_id));

                    // Load the pointer contained in the local onto the stack
                    self.emit_current(Instruction::LocalGet(ptr_local_id));

                    let (size, _align) = self.get_size_and_align(&sema_ty);
                    // Copy bytes to destination from source
                    self.emit_memcpy(size);

                    if dps_local_id.is_none() {
                        self.emit_current(Instruction::LocalGet(alloca_id));
                    }

                    return GenValResult {
                        ty: Ty::Ptr(Box::new(sema_ty)),
                        kind: ValueKind::LValuePtr,
                    };
                }
                // I guess this would be a non-mutable pointer inside a local?
                // Though that would be passed in a pointer, so sema ty woudn't be struct
                FunctionVariable::Local(_local_id, _mutable) => {
                    //self.emit_current(Instruction::LocalGet(*local_id));
                    // TODO: They can if they are passed to the function as a copy
                    panic!("compound types can never be stored directly in locals");
                }
            }
        }
        // Else, handle the simple types
        else if let Some(local) = self
            .function_ctx_stack
            .current()
            .local_variables
            .try_get_mapping(&scoped_ident.to_string())
        {
            match local {
                // This would be for mutable vars, need to get the pointer from the local, and load the value at that pointer onto the stack
                FunctionVariable::PtrLocal(ptr_local_id, _mutable) => {
                    self.emit_load_from_local(*ptr_local_id, &sema_ty);
                }
                // The value is directly in a local, just load the local
                FunctionVariable::Local(local_id, _mutable) => {
                    self.emit_current(Instruction::LocalGet(*local_id));
                }
            }

            return GenValResult {
                ty: sema_ty,
                kind: ValueKind::RValue,
            };
        }
        // And simple types embedded in global data (data section)
        else if let Some(global) = self
            .global_constants
            .get(&scoped_ident.to_string())
            .cloned()
            .or_else(|| self.global_statics.get(&scoped_ident.to_string()).cloned())
        {
            match global.ptr {
                super::wasm_globals::GlobalPtr::Memory(ptr) => {
                    // At this point, the global should be a simple value type that we should probably load onto the stack
                    self.emit_current(Instruction::I32Const(ptr as i32));
                    self.emit_load_from_ptr(&global.ty);
                }
                super::wasm_globals::GlobalPtr::WasmGlobal(global_id) => {
                    // Otherwise, just gen a global.get
                    self.emit_current(Instruction::GlobalGet(global_id));
                }
            }

            return GenValResult {
                ty: global.ty,
                kind: ValueKind::RValue,
            };
        } else {
            panic!(
                "No global or local matching {:?} was found!",
                scoped_ident.to_string()
            );
        }
    }

    fn gen_unary(
        &mut self,
        base_expr: &Expr,
        un_op: &UnOp,
        inner_expr: &Box<Expr>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        let inner_expr_ty = self.sema_expr_ty(&inner_expr.id);
        match un_op {
            UnOp::AddrOf => {
                if let ExprKind::ScopedIdent(ident) = &inner_expr.kind {
                    let key = GenericInstTyKeyRef {
                        name: &ident.to_string(),
                        tys: &vec![],
                    };

                    // This would happend because the parser doesn't know the difference
                    // between a addrof(variable) and addrof(non-generic function)
                    if let Some(_func) = self.function_to_id_map.get(&key) {
                        return self.gen_function_reference(
                            &inner_expr,
                            &FunctionReference {
                                path: ident.clone(),
                                generic_args: vec![],
                            },
                            dps_local_id,
                        );
                    }
                }
                // Otherwise just get an lvalue to the inner expression, since that will push
                // a pointer to it to the stack
                // This won't work if the inner expression is a local variable stored in a local
                // If this evaluates to a register, may need to somehow spill to stack
                // Not sure how to keep the reference to the local in tact though?

                // Need to look out for this when using lvals as rvals
                if let Some(dps_local) = dps_local_id {
                    // Need to actually store the result
                    // Dest: the dps local id
                    self.emit_current(Instruction::LocalGet(dps_local));

                    // value: the result of gen_lvalue
                    //So this pushes a pointer to the stack. Should be an actual memory pointer
                    //This is what needs to be stored AT the location given by the dps pointer
                    let result = self.gen_lvalue(&inner_expr);
                    self.emit_store_to_ptr(&result.ty);

                    assert!(result.kind == ValueKind::LValuePtr);

                    return result;
                } else {
                    self.gen_lvalue(&inner_expr)
                }
            }
            UnOp::Neg => {
                match inner_expr_ty {
                    Ty::I32 | Ty::I16 | Ty::I8 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::I32Const(0));
                        self.emit_current(Instruction::I32Sub);
                    }
                    Ty::I64 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::I64Const(0));
                        self.emit_current(Instruction::I64Sub);
                    }
                    Ty::F32 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::F32Neg);
                    }
                    Ty::F64 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::F64Neg);
                    }
                    _ => {
                        panic!("unexpected type in unary neg {:?}", inner_expr_ty);
                    }
                }
                return GenValResult {
                    ty: inner_expr_ty,
                    kind: ValueKind::RValue,
                };
            }
            UnOp::Not => {
                match inner_expr_ty {
                    // I think bitnot would be fine for this? Not sure my sema allows this yet
                    Ty::USize | Ty::U32 | Ty::I32 | Ty::U16 | Ty::I16 | Ty::U8 | Ty::I8 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::I32Const(-1));
                        self.emit_current(Instruction::I32Xor);
                    }
                    Ty::U64 | Ty::I64 => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::I64Const(-1));
                        self.emit_current(Instruction::I64Xor);
                    }
                    Ty::Bool => {
                        self.gen_expr(&inner_expr, None);
                        self.emit_current(Instruction::I32Const(1));
                        self.emit_current(Instruction::I32Xor);
                    }
                    _ => {
                        panic!("unsupported type in unary not {:?}", inner_expr_ty);
                    }
                }
                return GenValResult {
                    ty: inner_expr_ty,
                    kind: ValueKind::RValue,
                };
            }
            UnOp::Deref => {
                assert!(matches!(inner_expr_ty, Ty::Ptr(_)));
                let Ty::Ptr(inner_ptr_ty) = &inner_expr_ty else {
                    panic!("cannot deref non pointer types")
                };

                let gen_expr_result = self.gen_expr(&inner_expr, dps_local_id);

                // pointers can only be created by stack allocations - inner will never point to a register value
                if !gen_expr_result.ty.is_composite_ty() {
                    self.emit_load_from_ptr(&inner_ptr_ty);
                }

                return gen_expr_result;
            }
            UnOp::MaybeDeref => {
                if let Ty::Ptr(_inner_ptr_ty) = inner_expr_ty {
                    return self.gen_unary(base_expr, &UnOp::Deref, inner_expr, dps_local_id);
                } else {
                    return self.gen_expr(&inner_expr, dps_local_id);
                }
            }
        }
    }

    fn gen_call(
        &mut self,
        base_expr: &Expr,
        callee_expr: &Box<Expr>,
        args: &Vec<Expr>,
        generics: &Vec<Type>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // This is handled 2 ways - for ident expressions, need to just map the ident + generics to a function id
        // then just push the args and call

        'ident: {
            if let ExprKind::ScopedIdent(ident) = &callee_expr.kind {
                let full_ident = ident.to_string();

                if self
                    .function_ctx_stack
                    .current()
                    .local_variables
                    .try_get_mapping(&full_ident)
                    .is_some()
                {
                    // This is a local variable, not a function name
                    break 'ident;
                }

                // create a function key to get or create the function type
                let tys: Vec<Ty> = generics.iter().map(|t| self.ast_to_ty(t)).collect();
                let key = GenericInstTyKeyRef {
                    name: &full_ident,
                    tys: &tys,
                };

                let function_id = self.get_or_define_function(&key);

                // let ret_ty = self.ast_to_ty(ret_ty);
                let ret_ty = self.sema_expr_ty(&base_expr.id);

                // need to alloc a stack pointer and pass it to the function as the first arg (if the return ty is mem only)
                let mut sret_opt = None;
                if ret_ty.is_composite_ty() {
                    let sret = if let Some(dps_id) = dps_local_id {
                        dps_id
                    } else {
                        self.stack_alloc(&ret_ty, "alloc.sret")
                    };
                    sret_opt = Some(sret);
                    self.emit_current(Instruction::LocalGet(sret));
                }

                // Need to make an sret pointer if return type is mem only
                for arg in args {
                    // We want these pushed to the stack, so no dps
                    self.gen_expr(arg, None);
                }

                self.emit_current(Instruction::Call(function_id));

                // If using dps, there will be no "result" on the stack
                if let Some(sret) = sret_opt {
                    if dps_local_id.is_none() {
                        self.emit_current(Instruction::LocalGet(sret));
                    }
                }

                return GenValResult {
                    ty: if sret_opt.is_some() {
                        Ty::Ptr(Box::new(ret_ty))
                    } else {
                        ret_ty
                    },
                    kind: if sret_opt.is_some() {
                        ValueKind::SretRValue
                    } else {
                        ValueKind::RValue
                    },
                };
            }
        }

        // Going to want function id pushed to stack
        let expression_ty = self.sema_expr_ty(&callee_expr.id);
        let Ty::FuncPtr(_arg_tys, ret_ty) = expression_ty else {
            panic!("Can't build call expression on non-function pointer type");
        };

        let mut sret_opt = None;
        if ret_ty.is_composite_ty() {
            let sret = if let Some(dps_id) = dps_local_id {
                dps_id
            } else {
                self.stack_alloc(&ret_ty, "alloc.sret")
            };
            sret_opt = Some(sret);

            // Push dps arg to the stack
            self.emit_current(Instruction::LocalGet(sret));
        }

        for arg in args {
            // No dps here, need args on the stack
            self.gen_expr(arg, None);
        }

        // Push the function index in table to the stack
        let expression_result = self.gen_expr(&callee_expr, None);
        // Make sure it is the right type
        assert!(matches!(expression_result.ty, Ty::FuncPtr(_, _)));
        let Ty::FuncPtr(param_tys, ret_ty) = expression_result.ty else {
            panic!("Expected function pointer");
        };

        let type_index = self.get_type_index_from_signature(&param_tys, &ret_ty);

        self.emit_current(Instruction::CallIndirect {
            type_index,
            table_index: 0,
        });

        if dps_local_id.is_none() {
            if let Some(sret) = sret_opt {
                self.emit_current(Instruction::LocalGet(sret));
            }
        }

        return GenValResult {
            ty: if sret_opt.is_some() {
                Ty::Ptr(Box::new(*ret_ty))
            } else {
                *ret_ty
            },
            kind: if sret_opt.is_some() {
                ValueKind::SretRValue
            } else {
                ValueKind::RValue
            },
        };
    }

    fn gen_method_call(
        &mut self,
        base_expr: &Expr,
        reciever: &Box<Expr>,
        method_name: &Ident,
        args: &Vec<Expr>,
        generic_args: &Vec<Type>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        let reciever_ty = self.sema_expr_ty(&reciever.id);
        let ns_string = reciever_ty
            .to_ns_string()
            .expect("failed to extract namespace from type");
        let method_name_with_scope = format!("{}::{}", ns_string, method_name.text);
        // reciver needs to resolve to a struct or struct ptr(struct)

        //let ret_ty = self.ast_to_ty(ret_ty);
        let ret_ty = self.sema_expr_ty(&base_expr.id);

        // need to alloc a stack pointer and pass it to the function as the first arg (if the return ty is mem only)
        let mut sret_opt = None;
        if ret_ty.is_composite_ty() {
            let sret = if let Some(dps_id) = dps_local_id {
                dps_id
            } else {
                self.stack_alloc(&ret_ty, "alloc.sret")
            };
            sret_opt = Some(sret);
            // Push sret ptr to stack
            self.emit_current(Instruction::LocalGet(sret));
        }

        // Need to push self to the args (after sret, if applies)
        // Lvalue, don't want a copy
        let reciever_result = self.gen_lvalue(&reciever);
        let Ty::Ptr(ref inner) = reciever_result.ty else {
            panic!("result of reciver expression must be a pointer");
        };

        // ensure self ptr is on the stack
        // function expects a mutable pointer. Not sure if I need to wrap the stack pointer .. ?
        // seems unnecessary
        match &**inner {
            Ty::Struct(_ident, _generics) => {
                // The pointer is already on the stack. Woo (wheee?)
            }
            Ty::Ptr(ptr_inner) => {
                let Ty::Struct(ref _name, ref _generics) = **ptr_inner else {
                    panic!("cannot call method on non-struct pointers")
                };

                println!("Calling method on ptr ptr struct {:?}", reciever_result);
                // If result is Ptr(Ptr(struct)) need to unbox the first one
                match reciever_result.kind {
                    ValueKind::LValuePtr => {
                        self.emit_load_from_ptr(&Ty::Ptr(Box::new(*ptr_inner.clone())));
                    }
                    ValueKind::LValueLocalId(local_id) => {
                        self.emit_current(Instruction::LocalGet(local_id));
                    }
                    _ => {
                        panic!("reciver lvalue cannot be an rvalue")
                    }
                }
            }
            _ => {
                panic!("Inner type must be struct or pointer to struct")
            }
        }

        // more generics stuff
        let mut all_generic_args = vec![];
        if let Ty::Struct(_struct_ident, struct_generics) =
            self.maybe_get_inner_struct(&reciever_ty)
        {
            all_generic_args.append(&mut struct_generics.clone());
        } else {
            panic!("Method call not on a struct");
        }

        //

        // These generics are extra / in addition to those inherited by the thing they impl
        //let generics: Vec<Ty> = generic_args.iter().map(|a| self.ast_to_ty(a)).collect();
        all_generic_args.append(&mut generic_args.iter().map(|g| self.ast_to_ty(g)).collect());
        let key = GenericInstTyKeyRef {
            name: &method_name_with_scope,
            tys: &all_generic_args,
        };

        let function_id = self.get_or_define_function(&key);

        for arg in args {
            self.gen_expr(arg, None);
        }

        self.emit_current(Instruction::Call(function_id));

        if dps_local_id.is_none() {
            if let Some(sret) = sret_opt {
                self.emit_current(Instruction::LocalGet(sret));
            }
        }

        return GenValResult {
            ty: if sret_opt.is_some() {
                Ty::Ptr(Box::new(ret_ty))
            } else {
                ret_ty
            },
            kind: if sret_opt.is_some() {
                ValueKind::SretRValue
            } else {
                ValueKind::RValue
            },
        };
    }

    pub(crate) fn maybe_get_inner_struct(&self, source_ty: &Ty) -> Ty {
        match source_ty {
            Ty::Ptr(ty) => self.maybe_get_inner_struct(&ty),
            _ => source_ty.clone(),
        }
    }

    fn gen_cast(&mut self, _base_expr: &Expr, expr: &Box<Expr>, target_ty: &Type) -> GenValResult {
        let from_ty = &self.sema_expr_ty(&expr.id);
        let target_ty = self.ast_to_ty(target_ty);
        // Would essentially be a reinterpret_cast cast for pointers. Not ideal
        self.maybe_cast_expr(&expr, from_ty, &target_ty);

        GenValResult {
            ty: target_ty,
            kind: if self.is_lvalue(&expr) {
                ValueKind::LValuePtr
            } else {
                ValueKind::RValue
            },
        }
    }

    fn gen_struct_lit(
        &mut self,
        base_expr: &Expr,
        name: &Ident,
        fields: &Vec<(Ident, Expr)>,
        generic_inst_tys: &Vec<Type>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        let ty = &self.sema_expr_ty(&base_expr.id); // Ty of the base expr should be the struct type
        let alloca = if let Some(d) = dps_local_id {
            d
        } else {
            self.stack_alloc(ty, &format!("struct_alloc_{}", name.text))
        };

        // I don't know if the generic inst tys will be used here..? I forget how I resoved those

        // For field in fields
        // GEP the field
        // Store the value generated by the expression\
        let struct_def = *self
            .struct_defs
            .get(&name.text)
            .expect("Failed to find requested struct");

        // set up generic context
        let generic_tys: Vec<Ty> = generic_inst_tys.iter().map(|t| self.ast_to_ty(t)).collect();
        let mut ctx = HashMap::new();
        for (idx, def_generic) in struct_def.generics.iter().enumerate() {
            let Type::Generic {
                generic_name,
                index_in_decl: _,
            } = def_generic
            else {
                panic!("Def generic must be generic")
            };
            ctx.insert(generic_name.text.to_owned(), generic_tys[idx].clone());
        }

        // Push generic ctx
        self.generic_context.push(ctx);

        for (ident, expr) in fields {
            let field_ty = self.sema_expr_ty(&expr.id);

            if field_ty.is_composite_ty() {
                // The alloca returns the local's id, not the pointer, so load that onto the stack
                self.emit_current(Instruction::LocalGet(alloca));
                // This loads the element pointer onto the stack. Pass in the generics to make sure offsets are correct
                self.struct_gep(&struct_def, &ident, &generic_tys);

                // Write struct field at offset (which is on the stack)
                // The type is now a pointer to this field type, since gep returns a pointer to the element
                //self.emit_copy_from_expr(&expr, &field_ty, dps_local_id);

                // Need to make a new dps since this is memory without an alloca
                let dps_local = self.add_local_current(wasm_encoder::ValType::I32, "field.dps");
                // Set the local to the value of the GEP
                self.emit_current(Instruction::LocalSet(dps_local));

                self.gen_expr(expr, Some(dps_local));
            } else {
                // Push the dest pointer onto the stack
                self.emit_current(Instruction::LocalGet(alloca));
                // This loads the element pointer onto the stack. Pass in the generics to make sure offsets are correct
                self.struct_gep(&struct_def, &ident, &generic_tys);

                // Push the value onto the stack
                self.gen_expr(expr, None);

                // store to pointer
                self.emit_store_to_ptr(&field_ty);
            }
        }

        // I THINK I don't need to do this if the dps local was used
        // Unless let expressions would evaluate to some, but I'm not doing that yet.
        if dps_local_id.is_none() {
            self.emit_current(Instruction::LocalGet(alloca));
        }

        // Pop generic ctx
        self.generic_context.pop();

        // Shouldn't this still be an r-value?
        GenValResult {
            ty: Ty::Ptr(Box::new(ty.clone())),
            kind: ValueKind::RValue,
        }
    }

    pub(crate) fn gen_union_lit(
        &mut self,
        base_expr: &Expr,
        _base_path: &ScopedIdent,
        variant_ident: &Ident,
        init_epxr_opt: Option<&Box<Expr>>,
        _generic_inst_tys: &[Type], // Should I need these?
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // all I should need to do is create a dps pointer to the bit of mem after the discrim, and store r value in it
        let ty = &self.sema_expr_ty(&base_expr.id); // Ty of the base expr should be the struct type
        let Ty::Union(base, generic_args) = self.sema_expr_ty(&base_expr.id) else {
            panic!("expected type to be sum type")
        };

        let alloca = if let Some(d) = dps_local_id {
            d
        } else {
            self.stack_alloc(ty, &format!("sum_alloc_{}:>{}", base, variant_ident.text))
        };

        // look up discriminant, and store the value at offset + 0

        let union_def = *self
            .union_defs
            .get(&base)
            .expect("Failed to find requested union");

        // Change this to lookup instead of getting sema expression ty, since that isn't always known
        let discriminant = union_def.index_of(&variant_ident) as i32;

        // todo generic map
        let ctx = self.create_map_from_generic_tys(&generic_args, &union_def.generics);
        self.generic_context.push(ctx);

        // Destination
        self.emit_current(Instruction::LocalGet(alloca));
        // Value
        self.emit_current(Instruction::I32Const(discriminant));
        self.emit_store_to_ptr(&Ty::I32);

        let variant_tys: Vec<Ty> = union_def
            .variants
            .iter()
            .map(|v| self.ast_to_ty(&v.1))
            .collect();

        if let Some(init_expr) = init_epxr_opt {
            // create dps pointer
            //let init_ty = self.sema_expr_ty(&init_expr.id);
            let init_dps = self.add_local_current(wasm_encoder::ValType::I32, "dps.sum");
            // Base pointer
            self.emit_current(Instruction::LocalGet(alloca));
            // move the pointer to after the discriminant, respecting align
            let offset_d = self.variant_offset(&variant_tys);
            //println!("variant offset for {} is {}", _base, offset_d);
            self.emit_current(Instruction::I32Const(offset_d as i32));
            self.emit_current(Instruction::I32Add);

            // Set dps to calculated pointer
            self.emit_current(Instruction::LocalSet(init_dps));
            // Should load the value into dps pointer
            self.gen_expr(&init_expr, Some(init_dps));
        } else {
            // Dunno.. Nothing ? discrim needs to be set either way
        }

        // get memory address at alloca + discriminant
        self.generic_context.pop();

        if dps_local_id.is_none() {
            self.emit_current(Instruction::LocalGet(alloca));
        }

        GenValResult {
            ty: Ty::Ptr(Box::new(ty.clone())),
            kind: ValueKind::RValue,
        }
    }

    // TODO: In let / assign, could probably optimize out the stackalloc + copy
    /// Generate an array and return the stack pointer
    fn gen_array_lit(
        &mut self,
        base_expr: &Expr,
        exprs: &Vec<Expr>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        let ty = &self.sema_expr_ty(&base_expr.id); // Ty of the base expr should be the array type
        let Ty::Array(ele_ty, len) = ty else {
            panic!("expected array type")
        };

        let alloca_local_id = if let Some(d) = dps_local_id {
            d
        } else {
            self.stack_alloc(ty, &format!("tmp_lit_alloc_x{}", len))
        };

        // Size should be for the element ty - used to calc offsets
        let (size, _align) = self.get_size_and_align(ele_ty);

        for (idx, expr) in exprs.iter().enumerate() {
            let offset = idx * size;
            // calculate and push pointer to the stack
            self.emit_current(Instruction::I32Const(offset as i32));
            // Load the value inside the alloca local (the actual pointer)
            self.emit_current(Instruction::LocalGet(alloca_local_id));
            // Add, pushing pointer to the stack
            self.emit_current(Instruction::I32Add);

            // now push the value to the stack
            if ele_ty.is_composite_ty() {
                let dps_local = self.add_local_current(wasm_encoder::ValType::I32, "field.dps");
                // Set the local to the value of the pointer arithmetic
                self.emit_current(Instruction::LocalSet(dps_local));
                self.gen_expr(expr, Some(dps_local));
            } else {
                self.gen_expr(expr, None);
                // Store the expression result to the calculated pointer
                // I think structs should be handled by their own gen_expr?
                self.emit_store_to_ptr(&ele_ty);
            }
        }

        if dps_local_id.is_none() {
            self.emit_current(Instruction::LocalGet(alloca_local_id));
        }

        // I think this is actually an r-value even though it is a pointer
        // Though since it can be dps'd and is also a pointer to (maybe) temp mem, I'm honestly not sure
        GenValResult {
            ty: Ty::Ptr(Box::new(ty.clone())),
            kind: ValueKind::RValue,
        }
    }

    // this is 99% the same as gen field
    fn gen_index(
        &mut self,
        base_expr: &Expr,
        _indexee: &Box<Expr>,
        _indexer: &Box<Expr>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        self.gen_lval_as_rval(base_expr, dps_local_id)
    }

    fn gen_field(
        &mut self,
        base_expr: &Expr, // base expression is a field expression. can use to generate lvalue
        _target_val: &Box<Expr>, // struct to index into. shouldn't actually need this since field lvalue uses it to compute pointer
        _field_ident: &Ident,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        self.gen_lval_as_rval(base_expr, dps_local_id)
    }

    fn gen_lval_as_rval(&mut self, base_expr: &Expr, dps_local_id: Option<u32>) -> GenValResult {
        let ele_ty = self.sema_expr_ty(&base_expr.id);

        let alloca_local_id = if let Some(d) = dps_local_id {
            Some(d)
        } else {
            if !ele_ty.is_composite_ty() {
                None
            } else {
                let alloc = self.stack_alloc(&ele_ty, &format!("tmp_lval2r_alloc_{:?}", ele_ty));
                Some(alloc)
            }
        };

        match alloca_local_id {
            Some(local_id_to_ptr) => {
                self.emit_current(Instruction::LocalGet(local_id_to_ptr));
                let lval_result = self.gen_lvalue(base_expr);
                let Ty::Ptr(box_ele_ty) = lval_result.ty else {
                    panic!("should return pointer to element for lvalue!")
                };
                assert!(*box_ele_ty == ele_ty);

                let (size, _align) = self.get_size_and_align(&ele_ty);
                self.emit_memcpy(size);

                if dps_local_id.is_none() {
                    // Push to stack if part of r value aggregate expression
                    // Should't it emit a load for simple types though?
                    if ele_ty.is_composite_ty() {
                        self.emit_current(Instruction::LocalGet(local_id_to_ptr));
                    } else {
                        // If it should be passed by value, need to load it
                        self.emit_current(Instruction::LocalGet(local_id_to_ptr));
                        self.emit_load_from_ptr(&ele_ty);
                    }
                }

                return GenValResult {
                    ty: Ty::Ptr(Box::new(ele_ty)),
                    kind: ValueKind::RValue,
                };
            }
            // Will never be the case for mem only types - they need stack allocations
            None => {
                let lval_result = self.gen_lvalue(base_expr);
                match lval_result.kind {
                    ValueKind::LValuePtr => {
                        self.emit_load_from_ptr(&ele_ty);
                    }
                    ValueKind::LValueLocalId(local_id) => {
                        // I don't know how this would ever be possible tbh
                        self.emit_current(Instruction::LocalGet(local_id));
                    }
                    _ => {
                        panic!("expected lvalue from gen lvalue");
                    }
                }

                return GenValResult {
                    ty: ele_ty,
                    kind: ValueKind::RValue,
                };
            }
        }
    }

    fn create_map_from_generic_tys(
        &mut self,
        generic_args: &[Ty],
        def_generics: &[Type],
    ) -> HashMap<String, Ty> {
        let mut ctx = HashMap::new();
        for (idx, def_generic) in def_generics.iter().enumerate() {
            let Type::Generic {
                generic_name,
                index_in_decl: _,
            } = def_generic
            else {
                panic!("Def generic must be generic")
            };
            ctx.insert(generic_name.text.to_owned(), generic_args[idx].clone());
        }
        ctx
    }

    pub(crate) fn gen_inline_wat(
        &mut self,
        base_expr: &Expr,
        captured_locals: &[String],
        type_arg: &Option<Type>,
        wat: &str,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        // TODO:
        let _ = captured_locals;

        let base_expr_ty = self.sema_expr_ty(&base_expr.id);
        assert!(
            matches!(
                base_expr_ty,
                Ty::I32 | Ty::I64 | Ty::F32 | Ty::F64 | Ty::Void
            ),
            "inline wasm must evaluate to a wasm value type"
        );

        // need to create a fake module to hold this inline, since the encoding functionality for instructions
        // isn't exposed in wast (very annoying, ty)
        let module_wrapped = create_wat_module_from_text(
            wat,
            type_arg
                .clone()
                .map(|t| self.wasm_param_type(&self.ast_to_ty(&t))),
        );

        //println!("parsing wat: {}", module_wrapped);
        let wasm_bytes = wat::parse_str(module_wrapped).expect("invalid wat");

        let mut instr_bytes = vec![];
        for payload in wasmparser::Parser::new(0).parse_all(&wasm_bytes) {
            match payload.unwrap() {
                wasmparser::Payload::CodeSectionEntry(function_body) => {
                    if let Ok(mut locals_reader) = function_body.get_locals_reader() {
                        while let Ok((_count, _ty)) = locals_reader.read() {}
                    }

                    // Could use this to skip locals
                    // if let Ok(operators_reader) = function_body.get_operators_reader() {
                    //     //operators_reader.rea
                    // }

                    let mut reader = function_body.get_binary_reader();
                    instr_bytes = reader
                        .read_bytes(reader.bytes_remaining())
                        .expect("failed to read function")
                        .to_vec();
                }
                _ => {}
            }
        }

        // for byte in &instr_bytes {
        //     print!("{:02X} ", byte);
        // }

        // For some reason, the body is enclosed by "unreachable ... end"
        // Don't know why, but don't want that
        assert!(!instr_bytes.is_empty(), "instruction byte array is empty!");
        assert!(*instr_bytes.first().unwrap() == 0x00);
        assert!(*instr_bytes.last().unwrap() == 0x0B);
        instr_bytes.remove(0);
        instr_bytes.pop();

        self.function_ctx_stack
            .current_mut()
            .current_function
            .add_injection_mapping(instr_bytes);

        if let Some(dps_local) = dps_local_id {
            if type_arg.is_none() {
                panic!("expression with no type args should not have a destination pointer!");
            }
            // If we expect this expression to return a value, store it in the provided dps
            self.emit_current(Instruction::LocalSet(dps_local));
        }

        GenValResult {
            ty: self.ast_to_ty(&type_arg.as_ref().unwrap_or_else(|| &Type::Void)),
            kind: ValueKind::RValue,
        }
    }
}

fn create_wat_module_from_text(text: &str, result_ty: Option<ValType>) -> String {
    let result_clause = result_ty
        .map(|ty| format!("(result {})", wasm_ty_to_string(&ty)))
        .unwrap_or_default();

    format!(
        "(module (func (export \"inline\") {} \n{}\n))",
        result_clause, text
    )
}

fn wasm_ty_to_string(ty: &ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        ValType::V128 => "vi28",
        ValType::Ref(_ref_type) => "reftype",
    }
}
