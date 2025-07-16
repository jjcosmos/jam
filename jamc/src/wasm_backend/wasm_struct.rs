use hashbrown::HashMap;

use crate::{
    frontend::{ast::Type, sema::Ty},
    wasm_backend::wasm_codegen::WasmModule,
};

impl<'a> WasmModule<'a> {
    pub(crate) fn align_to(&self, offset: usize, align: usize) -> usize {
        (offset + align - 1) & !(align - 1)
    }

    pub(crate) fn layout_struct_ty(&mut self, fields: &[Ty]) -> (usize, Vec<usize>) {
        let mut offsets = Vec::new();
        let mut offset = 0;
        let mut max_align = 1;

        for field in fields {
            let (size, align) = self.get_size_and_align(&field);
            offset = self.align_to(offset, align);
            offsets.push(offset);
            offset += size;
            max_align = max_align.max(align);
        }

        let size = self.align_to(offset, max_align); // pad final size
        (size, offsets)
    }

    pub(crate) fn variant_offset(&mut self, tys: &[Ty]) -> usize {
        let max_align = tys
            .iter()
            .map(|t| self.get_size_and_align(t).1)
            .max()
            .unwrap_or(1);

        let (size, _) = self.get_size_and_align(&Ty::I32);
        self.align_to(size, max_align)
    }

    /// Get size and align in bytes
    /// Probably needs to respect generic context for layout - I think the funcgeneric case
    /// should handle that.
    pub(crate) fn get_size_and_align(&mut self, ty: &Ty) -> (usize, usize) {
        match ty {
            Ty::I8 | Ty::U8 | Ty::Bool => (1, 1),
            Ty::I16 | Ty::U16 => (2, 2),
            Ty::I32 | Ty::U32 | Ty::F32 | Ty::USize => (4, 4),
            Ty::I64 | Ty::U64 | Ty::F64 => (8, 8),
            Ty::Ptr(_) | Ty::NullPtr(_) => (4, 4), // 8 on wasm 64
            Ty::Array(elem_ty, count) => {
                let (elem_size, elem_align) = self.get_size_and_align(elem_ty);
                let size = self.align_to(elem_size, elem_align) * count;
                (size, elem_align)
            }
            Ty::Struct(ident, generic_types) => {
                // Was previously using generic types as fields. Should document this better
                let mut ctx = HashMap::new();
                let struct_def = self.struct_defs.get(ident).expect(&format!(
                    "failed to lookup struct {}. is it missing generics?",
                    ident
                ));
                for generic in &struct_def.generics {
                    let Type::Generic {
                        generic_name,
                        index_in_decl,
                    } = generic
                    else {
                        panic!()
                    };
                    ctx.insert(
                        generic_name.text.to_owned(),
                        generic_types[*index_in_decl].clone(),
                    );
                }
                self.generic_context.push(ctx);

                let fields: Vec<Ty> = struct_def
                    .fields
                    .iter()
                    .map(|f| self.ast_to_ty(&f.1))
                    .collect();

                self.generic_context.pop();

                let mut offset = 0;
                let mut max_align = 1;

                for field in fields {
                    let (field_size, field_align) = self.get_size_and_align(&field);
                    offset = self.align_to(offset, field_align);
                    offset += field_size;
                    max_align = max_align.max(field_align);
                }

                let size = self.align_to(offset, max_align);
                (size, max_align)
            }
            Ty::Union(ident, generic_types) => {
                let mut ctx = HashMap::new();
                let union_def = self.union_defs[ident];
                for generic in &union_def.generics {
                    let Type::Generic {
                        generic_name,
                        index_in_decl,
                    } = generic
                    else {
                        panic!("Type in generic param list must be generic")
                    };
                    ctx.insert(
                        generic_name.text.to_owned(),
                        generic_types[*index_in_decl].clone(),
                    );
                }
                self.generic_context.push(ctx);

                let variants: Vec<(usize, usize)> = union_def
                    .variants
                    .iter()
                    .map(|v| {
                        let (size, align) = self.get_size_and_align(&self.ast_to_ty(&v.1));
                        (size, align)
                    })
                    .collect();
                self.generic_context.pop();

                let (discriminant_size, discriminant_align) = self.get_size_and_align(&Ty::I32);

                let (size, align) = if variants.is_empty() {
                    // TODO: Should this be allowed?
                    let size = self.align_to(discriminant_size, discriminant_align);
                    (size, discriminant_align)
                } else {
                    let max_variant_size = variants.iter().map(|v| v.0).max().unwrap();
                    let max_variant_align = variants.iter().map(|v| v.1).max().unwrap();
                    let align_max = discriminant_align.max(max_variant_align);
                    let variant_offset = self.align_to(discriminant_size, max_variant_align);
                    let size = self.align_to(variant_offset + max_variant_size, align_max);
                    (size, align_max)
                };
                (size, align)
            }
            Ty::FuncPtr(_, _) => (4, 4), // same as pointer
            Ty::Void => (0, 1),
            Ty::FuncGeneric(name, _) => {
                let mapped = self.generic_context.get_mapping(name);
                self.get_size_and_align(&mapped.clone()) // Since self can't be reborrowed as mut
            }
            Ty::Error => unreachable!(),
        }
    }
}
