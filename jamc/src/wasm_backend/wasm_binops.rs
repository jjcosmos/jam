use wasm_encoder::{BlockType, Instruction, ValType};

use crate::{
    frontend::{
        ast::{BinOp, Expr},
        sema::Ty,
    },
    wasm_backend::{
        wasm_codegen::WasmModule,
        wasm_expression::{GenValResult, ValueKind},
    },
};

impl<'a> WasmModule<'a> {
    pub fn gen_binary(
        &mut self,
        base_expr: &Expr,
        lhs: &Box<Expr>,
        bin_op: &BinOp,
        rhs: &Box<Expr>,
        dps_local_id: Option<u32>,
    ) -> GenValResult {
        if let Some(dps) = dps_local_id {
            self.emit_current(Instruction::LocalGet(dps));
        }

        let lhs_ty = self.sema_expr_ty(&lhs.id);
        let rhs_ty = self.sema_expr_ty(&rhs.id);

        let result_ty = self.sema_expr_ty(&base_expr.id);
        let operand_tys = self.unify_binop_types(&lhs_ty, &rhs_ty);

        // Pushes expression values to the stack
        self.maybe_cast_expr(lhs, &lhs_ty, &operand_tys);

        // Short circuit logic (not super well tested)
        if matches!(bin_op, BinOp::And) {
            self.emit_current(Instruction::If(BlockType::Result(ValType::I32)));
            self.maybe_cast_expr(rhs, &rhs_ty, &operand_tys);

            self.emit_current(Instruction::Else);
            self.emit_current(Instruction::I32Const(0));
            self.emit_current(Instruction::End);

            if let Some(_dps) = dps_local_id {
                let sema_ty = self.sema_expr_ty(&base_expr.id);
                self.emit_store_to_ptr(&sema_ty);
            }

            return GenValResult {
                ty: result_ty,
                kind: ValueKind::RValue,
            };
        } else if matches!(bin_op, BinOp::Or) {
            self.emit_current(Instruction::If(BlockType::Result(ValType::I32)));
            self.emit_current(Instruction::I32Const(1));

            self.emit_current(Instruction::Else);
            self.maybe_cast_expr(rhs, &rhs_ty, &operand_tys);
            self.emit_current(Instruction::End);

            if let Some(_dps) = dps_local_id {
                let sema_ty = self.sema_expr_ty(&base_expr.id);
                self.emit_store_to_ptr(&sema_ty);
            }

            return GenValResult {
                ty: result_ty,
                kind: ValueKind::RValue,
            };
        }

        self.maybe_cast_expr(rhs, &rhs_ty, &operand_tys);

        // Special case since they require more than one instruction
        match (&result_ty, &bin_op) {
            (Ty::F32, BinOp::Mod) => {
                self.emit_rem_floor_f32();

                return GenValResult {
                    ty: result_ty,
                    kind: ValueKind::RValue,
                };
            }
            (Ty::F64, BinOp::Mod) => {
                self.emit_rem_floor_f64();

                if let Some(_dps) = dps_local_id {
                    let sema_ty = self.sema_expr_ty(&base_expr.id);
                    self.emit_store_to_ptr(&sema_ty);
                }

                return GenValResult {
                    ty: result_ty,
                    kind: ValueKind::RValue,
                };
            }
            _ => {}
        };

        use BinOp::*;
        let op = match bin_op {
            Add => instr_add(&operand_tys),
            Sub => instr_sub(&operand_tys),
            Mul => instr_mul(&operand_tys),
            Div => instr_div(&operand_tys),
            Mod => instr_rem(&operand_tys),
            Eq => instr_eq(&operand_tys),
            Ne => instr_ne(&operand_tys),
            Lt => instr_lt(&operand_tys),
            Gt => instr_gt(&operand_tys),
            Le => instr_le(&operand_tys),
            Ge => instr_ge(&operand_tys),
            BitAnd => instr_bitand(&operand_tys),
            BitOr => instr_bitor(&operand_tys),
            BitXor => instr_bitxor(&operand_tys),
            BitShiftL => instr_shl(&operand_tys),
            BitShiftR => instr_shr(&operand_tys),
            And => Instruction::I32And,
            Or => Instruction::I32Or,
        };

        self.emit_current(op);

        if let Some(_dps) = dps_local_id {
            let sema_ty = self.sema_expr_ty(&base_expr.id);
            self.emit_store_to_ptr(&sema_ty);
        }

        GenValResult {
            ty: result_ty,
            kind: ValueKind::RValue,
        }
    }

    fn unify_binop_types(&self, lhs: &Ty, rhs: &Ty) -> Ty {
        use Ty::*;
        match (lhs, rhs) {
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (I64, _) | (_, I64) => I64,
            (U64, _) | (_, U64) => U64,
            (I32, _) | (_, I32) => I32,
            (U32, _) | (_, U32) => U32,
            (I16, _) | (_, I16) => I32,
            (U16, _) | (_, U16) => U32,
            (I8, _) | (_, I8) => I32,
            (U8, _) | (_, U8) => U32,
            (Bool, Bool) => Bool,
            (Ptr(inner1), Ptr(inner2)) if inner1 == inner2 => I32,
            (NullPtr(_), Ptr(_)) => I32,
            (Ptr(_), NullPtr(_)) => I32,
            _ => panic!("Cannot unify binop types: {:?} and {:?}", lhs, rhs),
        }
    }

    pub fn maybe_cast_expr(&mut self, expr: &Expr, from_ty: &Ty, to_ty: &Ty) {
        self.gen_expr(expr, None);
        use Instruction::*;
        use Ty::*;

        if from_ty == to_ty {
            return;
        } else if from_ty
            .int_bit_width()
            .zip(to_ty.int_bit_width())
            .map(|(a, b)| a == b)
            .unwrap_or(false)
        {
            return; // Just reinterpret
        }
        if matches!(from_ty, Ty::I8 | Ty::U8 | Ty::I16 | Ty::U16 | Ty::Bool) {
            self.normalize_to_i32(from_ty);
            self.cast_i32_to(from_ty, to_ty);
        } else if matches!(from_ty, Ty::I64 | Ty::U64)
            && matches!(to_ty, Ty::I8 | Ty::U8 | Ty::I16 | Ty::U16 | Ty::Bool)
        {
            self.truncate_long_int_to_small(to_ty);
        } else {
            match (from_ty, to_ty) {
                (I32, F32) => self.emit_current(F32ConvertI32S),
                (U32, F32) => self.emit_current(F32ConvertI32U),
                (I32, F64) => self.emit_current(F64ConvertI32S),
                (U32, F64) => self.emit_current(F64ConvertI32U),
                (I64, F32) => self.emit_current(F32ConvertI64S),
                (U64, F32) => self.emit_current(F32ConvertI64U),
                (I64, F64) => self.emit_current(F64ConvertI64S),
                (U64, F64) => self.emit_current(F64ConvertI64U),

                (F32, I32) => self.emit_current(I32TruncF32S),
                (F32, U32) => self.emit_current(I32TruncF32U),
                (F64, I32) => self.emit_current(I32TruncF64S),
                (F64, U32) => self.emit_current(I32TruncF64U),
                (F32, I64) => self.emit_current(I64TruncF32S),
                (F32, U64) => self.emit_current(I64TruncF32U),
                (F64, I64) => self.emit_current(I64TruncF64S),
                (F64, U64) => self.emit_current(I64TruncF64U),

                (I32, I64) => self.emit_current(I64ExtendI32S),
                (U32, I64) => self.emit_current(I64ExtendI32U),
                (I64, I32) => self.emit_current(I32WrapI64),
                (U64, I32) => self.emit_current(I32WrapI64),

                (F64, F32) => self.emit_current(F32DemoteF64),
                // These are the same types for this purpose. Re-interpret cast
                (Ptr(_), Ptr(_))
                | (Ptr(_), I32)
                | (NullPtr(_), I32)
                | (I32, Ptr(_))
                | (FuncPtr(_, _), I32)
                | (I32, FuncPtr(_, _)) => {}
                (Union(_, _), I32) => {
                    panic!("union casts are not yet supported")
                }
                _ => panic!("Unsupported cast from {:?} to {:?}", from_ty, to_ty),
            }
        }
    }
}

fn instr_add(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Add,
        Ty::F64 => Instruction::F64Add,
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Add,
        Ty::I64 | Ty::U64 => Instruction::I64Add,
        _ => panic!("unsupported type for add: {:?}", ty),
    }
}

fn instr_sub(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Sub,
        Ty::F64 => Instruction::F64Sub,
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Sub,
        Ty::I64 | Ty::U64 => Instruction::I64Sub,
        _ => panic!("unsupported type for sub: {:?}", ty),
    }
}

fn instr_mul(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Mul,
        Ty::F64 => Instruction::F64Mul,
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Mul,
        Ty::I64 | Ty::U64 => Instruction::I64Mul,
        _ => panic!("unsupported type for mul: {:?}", ty),
    }
}

fn instr_div(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Div,
        Ty::F64 => Instruction::F64Div,
        Ty::I32 => Instruction::I32DivS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32DivU,
        Ty::I64 => Instruction::I64DivS,
        Ty::U64 => Instruction::I64DivU,
        _ => panic!("unsupported type for div: {:?}", ty),
    }
}

fn instr_rem(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 => Instruction::I32RemS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32RemU,
        Ty::I64 => Instruction::I64RemS,
        Ty::U64 => Instruction::I64RemU,
        _ => panic!("unsupported type for mod: {:?}", ty),
    }
}

fn instr_eq(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Eq,
        Ty::F64 => Instruction::F64Eq,
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Eq,
        Ty::I64 | Ty::U64 => Instruction::I64Eq,
        _ => panic!("unsupported type for eq: {:?}", ty),
    }
}

fn instr_ne(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Ne,
        Ty::F64 => Instruction::F64Ne,
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Ne,
        Ty::I64 | Ty::U64 => Instruction::I64Ne,
        _ => panic!("unsupported type for ne: {:?}", ty),
    }
}

fn instr_lt(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Lt,
        Ty::F64 => Instruction::F64Lt,
        Ty::I32 => Instruction::I32LtS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32LtU,
        Ty::I64 => Instruction::I64LtS,
        Ty::U64 => Instruction::I64LtU,
        _ => panic!("unsupported type for lt: {:?}", ty),
    }
}

fn instr_gt(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Gt,
        Ty::F64 => Instruction::F64Gt,
        Ty::I32 => Instruction::I32GtS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32GtU,
        Ty::I64 => Instruction::I64GtS,
        Ty::U64 => Instruction::I64GtU,
        _ => panic!("unsupported type for gt: {:?}", ty),
    }
}

fn instr_le(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Le,
        Ty::F64 => Instruction::F64Le,
        Ty::I32 => Instruction::I32LeS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32LeU,
        Ty::I64 => Instruction::I64LeS,
        Ty::U64 => Instruction::I64LeU,
        _ => panic!("unsupported type for le: {:?}", ty),
    }
}

fn instr_ge(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::F32 => Instruction::F32Ge,
        Ty::F64 => Instruction::F64Ge,
        Ty::I32 => Instruction::I32GeS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32GeU,
        Ty::I64 => Instruction::I64GeS,
        Ty::U64 => Instruction::I64GeU,
        _ => panic!("unsupported type for ge: {:?}", ty),
    }
}

fn instr_bitand(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32And,
        Ty::I64 | Ty::U64 => Instruction::I64And,
        _ => panic!("unsupported type for bitand: {:?}", ty),
    }
}

fn instr_bitor(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Or,
        Ty::I64 | Ty::U64 => Instruction::I64Or,
        _ => panic!("unsupported type for bitor: {:?}", ty),
    }
}

fn instr_bitxor(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Xor,
        Ty::I64 | Ty::U64 => Instruction::I64Xor,
        _ => panic!("unsupported type for bitxor: {:?}", ty),
    }
}

fn instr_shr(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 => Instruction::I32ShrS,
        Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32ShrU,
        Ty::I64 => Instruction::I64ShrS,
        Ty::U64 => Instruction::I64ShrU,
        _ => panic!("unsupported type for shift right: {:?}", ty),
    }
}

fn instr_shl(ty: &Ty) -> Instruction<'static> {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::USize => Instruction::I32Shl,
        Ty::I64 | Ty::U64 => Instruction::I64Shl,
        _ => panic!("unsupported type for shift left: {:?}", ty),
    }
}
