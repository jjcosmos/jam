use wasm_encoder::Instruction;

use crate::{
    frontend::{
        ast::{Expr, ExprKind, Ident, ScopedIdent, UnOp},
        sema::Ty,
    },
    wasm_backend::{
        wasm_codegen::WasmModule,
        wasm_expression::{GenValResult, ValueKind},
        wasm_function::FunctionVariable,
    },
};

impl<'a> WasmModule<'a> {
    /// Generate an assignable value given the expression.
    ///
    /// I think these should be pushing addresses to the stack.
    ///
    /// Flow would probably be gen_lvalue -> emit_store
    #[allow(dead_code)]
    pub(crate) fn gen_lvalue(&mut self, base_expr: &Expr) -> GenValResult {
        match &base_expr.kind {
            ExprKind::ScopedIdent(scoped_ident) => self.genl_scoped_ident(base_expr, scoped_ident),
            ExprKind::Unary(UnOp::Deref, expr) => self.genl_unary(base_expr, &UnOp::Deref, expr),
            ExprKind::Unary(UnOp::MaybeDeref, expr) => {
                self.genl_unary(base_expr, &UnOp::MaybeDeref, expr)
            }
            ExprKind::Index(expr, expr1) => self.genl_index(base_expr, expr, expr1),
            ExprKind::Field(expr, ident, _) => self.genl_field(base_expr, expr, ident),
            ExprKind::UnionLit(_, _, _, _)
            | ExprKind::StructLit {
                name: _,
                fields: _,
                generic_inst_tys: _,
            }
            | ExprKind::ArrayLit(_) => self.gen_lval_from_rval(base_expr),
            _ => {
                panic!(
                    "Expression kind {:?} is not an assignable lvalue",
                    base_expr.kind
                );
            }
        }
    }

    // For memory type literals, spill to the stack so that we get a memory location
    fn gen_lval_from_rval(&mut self, expr: &Expr) -> GenValResult {
        let expr_ty = self.sema_expr_ty(&expr.id);
        assert!(expr_ty.is_composite_ty());
        // Even for a literal, the pointer needs to point to actual stack memory
        let dps = self.stack_alloc(&expr_ty, "temp.lval.spill");
        let result = self.gen_expr(expr, Some(dps));
        self.emit_current(Instruction::LocalGet(dps));

        // For memory types (which this is for), the result type is already a pointer
        GenValResult {
            ty: result.ty,
            kind: ValueKind::LValuePtr,
        }
    }

    /// Determine if this is a local, and push the pointer to the stack
    /// TODO: Globals (once they aren't only const)
    fn genl_scoped_ident(&mut self, base_expr: &Expr, scoped_ident: &ScopedIdent) -> GenValResult {
        let sema_ty = self.sema_expr_ty(&base_expr.id);
        //println!("scoped ident {} ty is {:?}", scoped_ident.to_string(),sema_ty);
        // Before this can work, will need to make sure params (as locals) are added to the function context's variables
        if let Some(local) = self
            .function_ctx_stack
            .current()
            .local_variables
            .try_get_mapping(&scoped_ident.to_string())
        {
            //println!("local ty: {:?}", *local);
            match *local {
                FunctionVariable::PtrLocal(local_id, _mutable) => {
                    // the pointer is stored in a local
                    self.emit_current(Instruction::LocalGet(local_id));
                    return GenValResult {
                        ty: Ty::Ptr(Box::new(sema_ty)),
                        kind: ValueKind::LValuePtr,
                    };
                }
                FunctionVariable::Local(id_local, _mutable) => {
                    // I think this should be OK - the pointer in the local for a struct type SHOULD
                    // be a pointer owned my this function. So there is really no such thing as a
                    // const struct-by-value arg UNLESS the struct pointer passed in is NOT a copy
                    // could potentially track this and not allow this as an lvlaue

                    // This doesn't do anything? Can't get locals by index from the stack ..?
                    //self.emit_current(Instruction::I32Const(id_local as i32));
                    // println!(
                    //     "returning LOCAL value {} for {}",
                    //     id_local,
                    //     scoped_ident.to_string()
                    // );

                    return GenValResult {
                        ty: Ty::Ptr(Box::new(sema_ty)),
                        kind: ValueKind::LValueLocalId(id_local),
                    };
                }
            }
        } else if let Some(global_static) =
            self.global_statics.get(&scoped_ident.to_string()).cloned()
        {
            // We know this pointer statically, just push it to the stack
            match global_static.ptr {
                super::wasm_globals::GlobalPtr::Memory(ptr) => {
                    self.emit_current(Instruction::I32Const(ptr as i32));
                }
                super::wasm_globals::GlobalPtr::WasmGlobal(_global_id) => {
                    panic!(
                        "static {} should only live in memory, not in a local!",
                        scoped_ident.ident.text
                    );
                }
            }

            return GenValResult {
                ty: Ty::Ptr(Box::new(global_static.ty.clone())),
                kind: ValueKind::LValuePtr,
            };
        // TODO: This should be semantically invalid for assigment
        } else if let Some(global_const) = self
            .global_constants
            .get(&scoped_ident.to_string())
            .cloned()
        {
            match global_const.ptr {
                super::wasm_globals::GlobalPtr::Memory(ptr) => {
                    self.emit_current(Instruction::I32Const(ptr as i32));
                    return GenValResult {
                        ty: Ty::Ptr(Box::new(global_const.ty)),
                        kind: ValueKind::LValuePtr,
                    };
                }
                super::wasm_globals::GlobalPtr::WasmGlobal(_) => {
                    panic!("Wasm Globals cannot be lvalues.")
                }
            }
        } else {
            panic!("ident {:?} is not a valid lvalue", scoped_ident.to_string());
        }
    }

    /// This should handle deref & maybe_deref
    fn genl_unary(&mut self, _base_expr: &Expr, op: &UnOp, inner_expr: &Box<Expr>) -> GenValResult {
        // The semantic ty of the value being maybe-deref'd
        let sema_ty = self.sema_expr_ty(&inner_expr.id);

        match op {
            UnOp::Deref => {
                // TODO: This might mark everything in gen_expr as an r-value
                let result = self.gen_expr(&inner_expr, None);
                GenValResult {
                    ty: result.ty,
                    kind: ValueKind::LValuePtr,
                }
            }
            // only used in field access expressions atm
            // MaybeDeref(StructPtr/Val) . FieldIdent
            UnOp::MaybeDeref => {
                if let Ty::Ptr(inner_ty) = sema_ty {
                    assert!(
                        matches!(*inner_ty, Ty::Struct(_, _)),
                        "maybederef can only be used on struct pointers"
                    );
                    // Should be fine to just return the pointer here since it's what we want?
                    return self.gen_expr(&inner_expr, None);
                } else if matches!(
                    &inner_expr.kind,
                    ExprKind::StructLit {
                        name: _,
                        fields: _,
                        generic_inst_tys: _
                    }
                ) {
                    // Need to allocate on the stack to CREATE an lvalue for this, since it doesn't exist in mem
                    // gen_expr for struct literals
                    return self.gen_expr(&inner_expr, None);
                } else {
                    // I think this should just be a struct then?
                    assert!(matches!(sema_ty, Ty::Struct(_, _)));
                    // So just get the lval of the inner expression?
                    return self.gen_lvalue(&inner_expr);
                }
            }
            _ => {
                panic!("UnOp: {:?} cannot be used to create an l-value", op);
            }
        }
    }

    /// Handle index into array & index into pointer via index offset
    fn genl_index(
        &mut self,
        _base_expr: &Expr,
        collection: &Box<Expr>,
        index: &Box<Expr>,
    ) -> GenValResult {
        let sema_collection = self.sema_expr_ty(&collection.id);
        assert!(matches!(sema_collection, Ty::Ptr(_) | Ty::Array(_, _)));

        let ele_ty = match sema_collection {
            Ty::Ptr(inner_ty) => {
                // address of pointer + index * sizeof ty
                self.pointer_gep(&inner_ty, &collection, &index, None);
                inner_ty
            }
            Ty::Array(base_ty, _len) => {
                self.pointer_gep(&base_ty, &collection, &index, None);
                base_ty
            }
            _ => unreachable!(),
        };

        GenValResult {
            ty: Ty::Ptr(ele_ty),
            kind: ValueKind::LValuePtr,
        }
    }

    /// Generate a pointer to the field based off the offset. Essentially a GEP (since it needs to return a pointer to the field).
    fn genl_field(&mut self, base_expr: &Expr, expr: &Box<Expr>, ident: &Ident) -> GenValResult {
        let struct_sema_ty = self.sema_expr_ty(&expr.id);
        let Ty::Struct(struct_name, generics) = struct_sema_ty else {
            panic!("cannot generate field access for non-struct types");
        };

        let struct_def = *self
            .struct_defs
            .get(&struct_name)
            .expect(&format!("failed to lookup struct {}", struct_name));

        // Push the pointer to the struct onto the stack for the gep
        let gen_res = self.gen_lvalue(expr);

        // Just for safety, the generated lvalue must be a pointer to a struct
        assert!(matches!(gen_res.ty, Ty::Ptr(mem_ty) if matches!(*mem_ty, Ty::Struct(_, _))));

        // TODO: May need to set generic context here
        self.struct_gep(struct_def, ident, &generics);

        // Type SHOULD match Ptr(base_expr_type)
        let base_expr_ty = self.sema_expr_ty(&base_expr.id);
        GenValResult {
            ty: Ty::Ptr(Box::new(base_expr_ty)),
            kind: ValueKind::LValuePtr,
        }
    }

    pub fn is_lvalue(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::ScopedIdent(_) => true, // could be a variable or pointer
            ExprKind::Field(_, _, _) => true, // struct field access
            ExprKind::Index(_, _) => true,    // array indexing
            ExprKind::Unary(un_op, _) => matches!(un_op, UnOp::Deref | UnOp::MaybeDeref),
            _ => false, // everything else is not addressable
        }
    }
}
