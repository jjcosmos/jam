use wasm_encoder::{ConstExpr, GlobalType, ValType};

use crate::{
    frontend::{
        ast::{Expr, ExprKind, Literal},
        sema::Ty,
    },
    wasm_backend::wasm_codegen::WasmModule,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedPtr {
    pub ty: Ty,
    pub ptr: GlobalPtr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalPtr {
    Memory(u32),
    WasmGlobal(u32),
}

impl<'a> WasmModule<'a> {
    pub fn eval_const_expr(&mut self, expr: &Expr) -> Vec<u8> {
        match &expr.kind {
            ExprKind::Literal(Literal::Int(val, width, signed)) => match (width, signed) {
                (8, false) => (*val as u8).to_le_bytes().to_vec(),
                (8, true) => (*val as i8).to_le_bytes().to_vec(),
                (16, false) => (*val as u16).to_le_bytes().to_vec(),
                (16, true) => (*val as i16).to_le_bytes().to_vec(),
                (32, false) => (*val as u32).to_le_bytes().to_vec(),
                (32, true) => (*val as i32).to_le_bytes().to_vec(),
                (64, false) => (*val as u64).to_le_bytes().to_vec(),
                (64, true) => (*val as i64).to_le_bytes().to_vec(),
                _ => panic!("unsupported integer width"),
            },

            ExprKind::Literal(Literal::Float(val, is_f32)) => {
                if *is_f32 {
                    (*val as f32).to_le_bytes().to_vec()
                } else {
                    (*val as f64).to_le_bytes().to_vec()
                }
            }

            ExprKind::Literal(Literal::Bool(b)) => {
                vec![if *b { 1 } else { 0 }]
            }

            ExprKind::Literal(Literal::Str(s)) => {
                // Create a global string and return a pointer to it
                // TODO: This currently bypasses string de-duplication
                let mut bytes = s.as_bytes().to_vec();
                bytes.push(b'\0');
                bytes
            }

            ExprKind::StructLit {
                name,
                fields,
                generic_inst_tys: _generic_concrete_tys,
            } => {
                let struct_def = self.struct_defs.get(&name.text).unwrap();

                assert_eq!(
                    fields.len(),
                    struct_def.fields.len(),
                    "struct fields must all be defined!"
                );

                let mut fields_ordered = fields.clone();
                fields_ordered.sort_by(|a, b| {
                    let iofa = struct_def
                        .fields
                        .iter()
                        .position(|n| n.0.text == a.0.text)
                        .unwrap();
                    let iofb = struct_def
                        .fields
                        .iter()
                        .position(|n| n.0.text == b.0.text)
                        .unwrap();
                    iofa.cmp(&iofb)
                });

                let mut buffer = vec![];
                let mut offset = 0;

                for (_field_ident, field_expr) in fields_ordered {
                    let field_ty = &self.sema_expr_ty(&field_expr.id);
                    let (_size, align) = self.get_size_and_align(field_ty);

                    // Compute aligned offset for this field
                    let aligned_offset = self.align_to(offset, align);

                    // Zero fill the gap from offset to aligned_offset, if any
                    if aligned_offset > offset {
                        insert_bytes_at(&mut buffer, offset, &vec![0u8; aligned_offset - offset]);
                    }

                    offset = self.align_to(offset, align);

                    let bytes = self.eval_const_expr(&field_expr);
                    insert_bytes_at(&mut buffer, offset, &bytes);

                    offset += bytes.len();
                }

                buffer
            }

            _ => panic!("unsupported or non-constant initializer"),
        }
    }

    pub fn new_global(
        &mut self,
        val_type: ValType,
        init_expr: &ConstExpr,
        mutable: bool,
        name: &str,
    ) -> u32 {
        let global_id = self.global_section.len();
        self.global_section.global(
            GlobalType {
                val_type,
                mutable,
                shared: false,
            },
            init_expr,
        );

        self.dbg_name_bld.add_global(global_id, name);

        global_id
    }

    pub fn store_primitive_global(&mut self, prim_expr: &Expr, name: &str) -> u32 {
        match &prim_expr.kind {
            ExprKind::Literal(literal) => {
                match literal {
                    Literal::Int(value, bit_width, _signed) => {
                        if *bit_width > 32 {
                            self.new_global(
                                ValType::I64,
                                &ConstExpr::i64_const(*value as i64),
                                false,
                                name,
                            )
                        } else {
                            self.new_global(
                                ValType::I32,
                                &ConstExpr::i32_const(*value as i32),
                                false,
                                name,
                            )
                        }
                    }
                    Literal::USize(value) => self.new_global(
                        ValType::I32,
                        &ConstExpr::i32_const(*value as i32),
                        false,
                        "global.const",
                    ),
                    Literal::Float(value, is32bit) => {
                        if *is32bit {
                            self.new_global(
                                ValType::F32,
                                &ConstExpr::f32_const((*value as f32).into()),
                                false,
                                name,
                            )
                        } else {
                            self.new_global(
                                ValType::F64,
                                &ConstExpr::f64_const((*value as f64).into()),
                                false,
                                name,
                            )
                        }
                    }
                    Literal::Bool(value) => self.new_global(
                        ValType::I32,
                        &ConstExpr::i32_const(*value as i32),
                        false,
                        name,
                    ),
                    _ => {
                        panic!(
                            "literal type {:?} not supported in constant expressions",
                            literal
                        )
                    } // Should support this
                      //Literal::SizeOf(_) => todo!(),
                }
            }
            _ => {
                panic!("expression kind {:?} is not a prim type", prim_expr.kind)
            }
        }
    }
}

fn insert_bytes_at(buf: &mut Vec<u8>, offset: usize, data: &[u8]) {
    if buf.len() < offset + data.len() {
        buf.resize(offset + data.len(), 0);
    }
    buf[offset..offset + data.len()].copy_from_slice(data);
}
