use hashbrown::{HashMap, HashSet};
use std::{fs, io::ErrorKind, path::PathBuf};
use wasm_encoder::{
    CodeSection, CustomSection, DataSection, ElementSection, ExportSection, Function,
    FunctionSection, GlobalSection, ImportSection, MemorySection, Module, NameSection,
    TableSection, TypeSection, ValType,
};

use crate::{
    SourceFile,
    frontend::{
        ast::{
            EnumDef, ExprKind, ExternalProcDef, Literal, PrimTy, ProcDef, StructDef, Type, UnionDef,
        },
        sema::{Ty, TyTable},
    },
    wasm_backend::{
        function_ctx::{FunctionCtx, GenericContext},
        wasm_debug::DebugNameBuilder,
        wasm_function::{GenericInstTyKey, TableEntry},
        wasm_globals::TypedPtr,
        wasm_mem::{BssSectionBuilder, DATA_BASE, DataSectionBuilder},
    },
};
// VERY USEFUL (except for br table)
// https://developer.mozilla.org/en-US/docs/WebAssembly/Reference

pub struct WasmModule<'a> {
    pub module: Module,

    pub addr_of_uses_by_function: &'a HashMap<String, Vec<String>>,
    pub static_strings: HashSet<String>,

    pub custom_section: CustomSection<'a>,
    pub type_section: TypeSection,
    pub import_section: ImportSection,
    pub function_section: FunctionSection,
    pub table_section: TableSection,
    pub memory_section: MemorySection,
    pub global_section: GlobalSection,
    pub export_section: ExportSection,
    pub element_section: ElementSection,
    pub code_section: CodeSection,
    pub data_section: DataSection,
    pub name_section: NameSection,

    // Code section queue
    pub code_section_queue: Vec<(u32, Function)>,

    pub function_to_id_map: HashMap<GenericInstTyKey, WasmTypeId>,
    pub table_entries_to_write: Vec<TableEntry>,
    pub sig_to_type_map: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    pub dbg_name_bld: DebugNameBuilder,

    // Ast info
    pub external_functions: HashMap<String, &'a ExternalProcDef>,
    pub internal_functions: HashMap<String, &'a ProcDef>,

    pub struct_defs: HashMap<String, &'a StructDef>,
    pub union_defs: HashMap<String, &'a UnionDef>,
    pub enum_defs: HashMap<String, &'a EnumDef>,

    pub global_constants: HashMap<String, TypedPtr>,
    pub global_statics: HashMap<String, TypedPtr>,
    pub struct_layouts: HashMap<GenericInstTyKey, (usize, Vec<usize>)>,
    pub(crate) enum_variant_map: HashMap<String, HashMap<String, u32>>,

    pub generic_context: GenericContext<Ty>,
    pub function_ctx_stack: FunctionCtxStack<'a>,

    pub ty_table: &'a TyTable,

    pub stack_ptr_id: WasmTypeId,

    pub data_builder: DataSectionBuilder,
    pub bss_builder: BssSectionBuilder,
}

pub struct FunctionCtxStack<'a> {
    pub(crate) stack: Vec<FunctionCtx<'a>>,
}

impl<'a> FunctionCtxStack<'a> {
    pub fn new() -> Self {
        Self { stack: vec![] }
    }

    pub fn push(&mut self, ctx: FunctionCtx<'a>) {
        self.stack.push(ctx);
    }

    pub fn pop(&mut self) -> Option<FunctionCtx> {
        self.stack.pop()
    }

    pub fn current_mut(&mut self) -> &mut FunctionCtx<'a> {
        self.stack.last_mut().unwrap()
    }

    pub fn current(&self) -> &FunctionCtx<'a> {
        self.stack.last().unwrap()
    }
}

pub(crate) type WasmTypeId = u32;

pub fn emit_wasm<'a>(
    sources: &Vec<SourceFile>,
    ty_table: &'a TyTable,
    addr_of_uses_by_function: &'a HashMap<String, Vec<String>>,
    static_strings: HashSet<String>,
) -> std::io::Result<()> {
    let mut wasm = WasmModule::new(ty_table, addr_of_uses_by_function, static_strings);

    let mut proc_bucket = vec![];
    let mut extern_proc_bucket = vec![];
    let mut union_def_bucket = vec![];
    let mut struct_def_bucket = vec![];
    let mut enum_def_bucket = vec![];
    let mut const_def_bucket = vec![];
    let mut static_def_bucket = vec![];
    let mut include_def_bucket = vec![];

    for source in sources {
        for item in &source.program.items {
            match item {
                crate::frontend::ast::Item::ProcDef(proc_def) => proc_bucket.push(proc_def),
                crate::frontend::ast::Item::ExternProc(external_proc_def) => {
                    extern_proc_bucket.push(external_proc_def)
                }
                crate::frontend::ast::Item::StructDef(struct_def) => {
                    struct_def_bucket.push(struct_def)
                }
                crate::frontend::ast::Item::UnionDef(union_def) => union_def_bucket.push(union_def),
                crate::frontend::ast::Item::EnumDef(enum_def) => enum_def_bucket.push(enum_def),
                crate::frontend::ast::Item::ConstDef(const_def) => const_def_bucket.push(const_def),
                crate::frontend::ast::Item::StaticDef(static_def) => {
                    static_def_bucket.push(static_def)
                }
                crate::frontend::ast::Item::IncludeStmt(include_stmt) => {
                    include_def_bucket.push(include_stmt)
                }
            }
        }
    }

    // Struct, enum, and union defs

    for struct_def in struct_def_bucket {
        wasm.struct_defs
            .insert(struct_def.name.text.to_owned(), struct_def);
    }

    for union_def in union_def_bucket {
        wasm.union_defs
            .insert(union_def.name.text.to_owned(), union_def);
    }

    for enum_def in enum_def_bucket {
        for variant in &enum_def.variants {
            if let Some(map) = wasm.enum_variant_map.get_mut(&enum_def.name.text) {
                map.insert(variant.0.text.clone(), variant.1 as u32);
            } else {
                wasm.enum_variant_map
                    .insert(enum_def.name.text.clone(), HashMap::new());
                wasm.enum_variant_map
                    .get_mut(&enum_def.name.text)
                    .unwrap()
                    .insert(variant.0.text.clone(), variant.1 as u32);
            }

            wasm.enum_defs
                .insert(enum_def.name.text.to_owned(), enum_def);
        }
    }

    // generate imports / write import section
    // TODO: Add function metadata for exports

    for extern_def in extern_proc_bucket {
        wasm.external_functions
            .insert(extern_def.name.text.to_owned(), extern_def);
        wasm.declare_external_function(&extern_def.name.text);
    }

    for const_def in const_def_bucket {
        match &const_def.ty {
            Type::Primitive(prim_ty) => match prim_ty {
                // need this to fall through
                PrimTy::CStr => {}
                _ => {
                    let global_id =
                        wasm.store_primitive_global(&const_def.value, &const_def.name.text);

                    let ptr = TypedPtr {
                        ty: wasm.ast_to_ty(&const_def.ty),
                        ptr: crate::wasm_backend::wasm_globals::GlobalPtr::WasmGlobal(global_id),
                    };
                    wasm.global_constants
                        .insert(const_def.name.text.to_owned(), ptr);
                    continue;
                }
            },
            _ => {}
        };

        let data = wasm.eval_const_expr(&const_def.value);
        let is_static_str = matches!(const_def.value.kind, ExprKind::Literal(Literal::Str(_)));
        let sema_ty = wasm.sema_expr_ty(&const_def.value.id);
        // Note that since cstrings are semantically u8 pointers, they need to align to 1
        let (_size, align) = wasm.get_size_and_align(&sema_ty);
        let ptr = wasm.data_builder.add_data(&data, is_static_str, align);
        let ty_ptr: TypedPtr = TypedPtr {
            ty: wasm.ast_to_ty(&const_def.ty),
            ptr: crate::wasm_backend::wasm_globals::GlobalPtr::Memory(ptr as u32),
        };
        wasm.global_constants
            .insert(const_def.name.text.to_owned(), ty_ptr);
    }

    for proc_def in &proc_bucket {
        wasm.internal_functions
            .insert(proc_def.name.text.to_owned(), proc_def);

        if proc_def.generics.is_empty() {
            wasm.declare_function(&proc_def.name.text);
        }
    }

    wasm.bss_builder =
        BssSectionBuilder::init_from_data_builder(&wasm.data_builder, &wasm.static_strings);

    for static_def in static_def_bucket {
        let (data_size, align) = wasm.get_size_and_align(&wasm.ast_to_ty(&static_def.ty));
        let ptr = wasm.bss_builder.add_entry(data_size, align);
        // Statics are only allowed in memory
        let ty_ptr: TypedPtr = TypedPtr {
            ty: wasm.ast_to_ty(&static_def.ty),
            ptr: super::wasm_globals::GlobalPtr::Memory(ptr as u32),
        };
        wasm.global_statics
            .insert(static_def.name.text.to_owned(), ty_ptr);
    }

    // This needs to be updated before defining procedures, as it is used to build the stack frames
    // init_stack will create the actual global though
    wasm.stack_ptr_id = wasm.global_section.len();

    for proc_def in proc_bucket {
        if proc_def.generics.is_empty() {
            wasm.define_function(&proc_def.name.text);
        }
    }

    wasm.init_mem();
    wasm.init_stack();

    // Need the length of the functions first, but since the table is written before the ele
    // section, need to make sure it is properly initialized
    wasm.init_function_table();

    // write other sections
    wasm.module.section(&wasm.custom_section);
    wasm.module.section(&wasm.type_section);
    wasm.module.section(&wasm.import_section);
    wasm.module.section(&wasm.function_section);
    wasm.module.section(&wasm.table_section);
    wasm.module.section(&wasm.memory_section);
    wasm.module.section(&wasm.global_section);
    wasm.write_export_section();
    wasm.write_elements_section();
    wasm.write_code_section();
    wasm.write_data_section();
    wasm.write_name_section();

    let wasm_bytes = wasm.module.finish();

    let out_dir: PathBuf = "out".into();
    if !out_dir.exists() {
        fs::create_dir_all(&out_dir).unwrap();
    }

    let mut failed = false;
    match wasmparser::validate(&wasm_bytes) {
        Ok(_) => {
            //println!("WASM bytes: {:?}", &wasm_bytes[..4]);
            let wasm_path = out_dir.join("main.wasm");
            fs::write(wasm_path, &wasm_bytes).expect("failed to write .wasm");
        }
        Err(e) => {
            failed = true;
            println!("validate failed: {}", e)
        }
    }

    let wat = wasmprinter::print_bytes(&wasm_bytes);
    match wat {
        Ok(string) => {
            let wat_path = out_dir.join("main.wat");
            fs::write(wat_path, string.as_bytes()).expect("failed to write .wat");
        }
        Err(e) => {
            eprintln!("wat write error: {:?}", e);
        }
    };

    //assert!(std::fs::read("out/main.wasm")?[..4] == [0, 97, 115, 109]);
    if failed {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "validation failed",
        ));
    }

    Ok(())
}

impl<'a> WasmModule<'a> {
    pub fn new(
        ty_table: &'a TyTable,
        addr_of_uses_by_function: &'a HashMap<String, Vec<String>>,
        static_strings: HashSet<String>,
    ) -> Self {
        WasmModule {
            module: Module::new(),

            addr_of_uses_by_function,
            static_strings,

            custom_section: new_custom_section(true),
            memory_section: MemorySection::new(),
            data_section: DataSection::new(),

            global_section: GlobalSection::new(),
            type_section: TypeSection::new(),
            function_section: FunctionSection::new(),
            table_section: TableSection::new(),
            export_section: ExportSection::new(),
            import_section: ImportSection::new(),
            code_section: CodeSection::new(),
            element_section: ElementSection::new(),
            name_section: NameSection::new(),

            code_section_queue: vec![],

            dbg_name_bld: DebugNameBuilder::new(),

            external_functions: HashMap::new(),
            internal_functions: HashMap::new(),

            struct_defs: HashMap::new(),
            union_defs: HashMap::new(),
            enum_defs: HashMap::new(),

            global_constants: HashMap::new(),
            global_statics: HashMap::new(),
            struct_layouts: HashMap::new(),
            enum_variant_map: HashMap::new(),

            generic_context: GenericContext::new(false),
            function_ctx_stack: FunctionCtxStack::new(),

            ty_table,

            stack_ptr_id: 0,

            data_builder: DataSectionBuilder::new(DATA_BASE),
            bss_builder: BssSectionBuilder::uninit(), // this can't be properly initialized yet

            function_to_id_map: HashMap::new(),

            sig_to_type_map: HashMap::new(),
            table_entries_to_write: vec![],
        }
    }

    /// Only works when a generic context is properly set
    pub(crate) fn full_resolve_generic(&self, source_ty: &Ty) -> Ty {
        match source_ty {
            // If the inner contains a generic, wrap in self ty and recurse
            Ty::Ptr(ty) => {
                if let Ty::FuncGeneric(g, _i) = ty.as_ref() {
                    Ty::Ptr(Box::new(
                        self.full_resolve_generic(self.generic_context.get_mapping(g)),
                    ))
                } else {
                    source_ty.clone()
                }
            }
            Ty::Array(ty, len) => {
                if let Ty::FuncGeneric(g, _i) = ty.as_ref() {
                    Ty::Array(
                        Box::new(self.full_resolve_generic(self.generic_context.get_mapping(g))),
                        *len,
                    )
                } else {
                    source_ty.clone()
                }
            }
            Ty::Struct(name, items) => {
                let generics = items
                    .iter()
                    .map(|item| self.full_resolve_generic(item))
                    .collect();

                Ty::Struct(name.to_string(), generics)
            }
            Ty::FuncPtr(args, ty) => {
                let res_args = args.iter().map(|a| self.full_resolve_generic(a)).collect();

                Ty::FuncPtr(res_args, Box::new(self.full_resolve_generic(ty)))
            }
            Ty::NullPtr(ty) => {
                if let Ty::FuncGeneric(g, _i) = ty.as_ref() {
                    Ty::NullPtr(Box::new(
                        self.full_resolve_generic(self.generic_context.get_mapping(g)),
                    ))
                } else {
                    source_ty.clone()
                }
            }
            Ty::FuncGeneric(g_name, _) => {
                self.full_resolve_generic(self.generic_context.get_mapping(&g_name))
            }
            _ => source_ty.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn mangle_generic(&self, base: &str, generics: &Vec<Ty>) -> String {
        if generics.is_empty() {
            return base.to_string();
        }

        let mut out = format!("__{}", base);
        for arg in generics {
            out.push_str(&format!("_{:?}", arg).replace(" ", "_").replace("\"", ""));
        }

        out
    }

    pub(crate) fn ast_to_ty(&self, ast_ty: &Type) -> Ty {
        match ast_ty {
            Type::Primitive(p) => match p {
                PrimTy::I32 => Ty::I32,
                PrimTy::F64 => Ty::F64,
                PrimTy::Bool => Ty::Bool,
                PrimTy::I8 => Ty::I8,
                PrimTy::I16 => Ty::I16,
                PrimTy::I64 => Ty::I64,
                PrimTy::U8 => Ty::U8,
                PrimTy::U16 => Ty::U16,
                PrimTy::U32 => Ty::U32,
                PrimTy::U64 => Ty::U64,
                PrimTy::F32 => Ty::F32,
                PrimTy::CStr => Ty::Ptr(Box::new(Ty::I8)),
                PrimTy::USize => Ty::USize,
            },
            Type::Ptr(inner) => Ty::Ptr(Box::new(self.ast_to_ty(inner))),
            Type::Array(elem, len) => Ty::Array(Box::new(self.ast_to_ty(elem)), *len),
            Type::Void => Ty::Void,
            Type::FuncPtr(params, ret) => Ty::FuncPtr(
                params.iter().map(|t| self.ast_to_ty(t)).collect(),
                Box::new(self.ast_to_ty(ret)),
            ),
            Type::Named(ident, generics) => {
                if self.struct_defs.contains_key(&ident.text) {
                    Ty::Struct(
                        ident.text.to_owned(),
                        generics.iter().map(|g| self.ast_to_ty(g)).collect(),
                    )
                } else if self.union_defs.contains_key(&ident.text) {
                    Ty::Union(
                        ident.text.to_owned(),
                        generics.iter().map(|g| self.ast_to_ty(g)).collect(),
                    )
                } else if self.enum_defs.contains_key(&ident.text) {
                    Ty::I32
                } else {
                    panic!("could not find named type for {:?}", ident);
                }
            }
            Type::Inferred(_range) => {
                panic!("Cannot convert AST to Sema Ty: Type cannot be inferred here.")
            }
            Type::Generic {
                //source_ident: _,
                generic_name,
                index_in_decl: _,
            } => {
                self.generic_context.get_mapping(&generic_name.text).clone()
                // if let Some(gctx) = self.generic_context {
                //     gctx[&generic_name.text].clone()
                // } else {
                //     panic!("generic type conversion outside of a generic context is unsupported")
                // }
            }
        }
    }

    // TODO: Could cache these
    /// Get the type of the expression, taking into account the current generic context
    pub fn sema_expr_ty(&mut self, expr_id: &(usize, usize)) -> Ty {
        return self.full_resolve_generic(self.ty_table.expr.get(expr_id).expect(&format!(
            "failed to get type from table given expression id: {:?}",
            expr_id
        )));
    }
}

/// I don't know if I'll use this or some other way to detect gui apps, but it's neat
/// Could give an option to not even pack the gui runner for a smaller binary
fn new_custom_section(is_gui: bool) -> CustomSection<'static> {
    let config = if is_gui { "gui=true" } else { "gui=false" };
    let custom = CustomSection {
        name: std::borrow::Cow::Borrowed("jam.runtime"),
        data: config.as_bytes().to_vec().into(),
    };
    custom
}
