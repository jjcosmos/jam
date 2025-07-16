use hashbrown::HashMap;
use hashbrown::HashSet;
use logos::Span;

use crate::SourceFile;
use crate::frontend::ast::{self, Block, ScopedIdent, StaticDef, Type, UnionDef};
use crate::frontend::ast::{
    BinOp, ConstDef, EnumDef, Expr, ExprKind, ExternalProcDef, Ident, Item, Literal, ProcDef,
    Program, Stmt, StructDef, UnOp,
};
use crate::wasm_backend::function_ctx::GenericContext;

type ExprId = (usize, usize);
/// SourceID, Index
type ItemId = (usize, usize);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    USize,
    Bool,
    Ptr(Box<Ty>),
    Array(Box<Ty>, usize),
    /// This itself needs to be a concrete type.
    ///
    /// Identifier, CONCRETE Generic instance types. Diff from funcptr is that this doesn't really hold the info, just the ident to map to the type
    Struct(String, Vec<Ty>),
    // Resolved union ty. Ident, Generics
    Union(String, Vec<Ty>),
    FuncPtr(Vec<Ty>, Box<Ty>),
    Void,
    NullPtr(Box<Ty>),
    FuncGeneric(String, usize),
    Error, // keeps analysis going after a mistake. Doesn't work well yet since type results get screwy
}

impl Ty {
    fn is_numeric(&self) -> bool {
        return matches!(
            *self,
            Ty::I8
                | Ty::I16
                | Ty::I32
                | Ty::I64
                | Ty::U8
                | Ty::U16
                | Ty::U32
                | Ty::U64
                | Ty::F32
                | Ty::F64
        );
    }

    pub(crate) fn to_ns_string(&self) -> Option<String> {
        match self {
            Ty::Struct(ident, _) => Some(ident.to_owned()),
            Ty::Ptr(inner) => match &**inner {
                Ty::Struct(inner_ident, _) => Some(inner_ident.to_owned()),
                _ => None,
            },
            _ => {
                println!("Cannot get type namespace from non struct type {:?}", self);
                None
            }
        }
    }

    pub(crate) fn is_signed(&self) -> bool {
        matches!(self, Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64)
    }

    pub(crate) fn is_composite_ty(&self) -> bool {
        matches!(self, Ty::Struct(_, _) | Ty::Union(_, _) | Ty::Array(_, _))
    }

    pub(crate) fn int_bit_width(&self) -> Option<usize> {
        match self {
            Ty::I8 | Ty::U8 => Some(8),
            Ty::I16 | Ty::U16 => Some(16),
            Ty::I32 | Ty::U32 => Some(32),
            Ty::I64 | Ty::U64 => Some(64),
            Ty::USize => Some(32), // TODO: ONLY FOR WASM
            _ => None,
        }
    }

    pub(crate) fn get_path_head(&self) -> Option<(String, Vec<Ty>)> {
        match self {
            Ty::Union(path_head, items) => Some((path_head.clone(), items.clone())),
            _ => None,
        }
    }
}

struct SemaScope {
    vars: HashSet<String>,
}

#[derive(Default)]
pub struct TyTable {
    pub expr: HashMap<ExprId, Ty>, // every Expr’s final type
    pub item: HashMap<ItemId, Ty>, // type of consts / functions
}

#[derive(Default)]
pub struct ResTable {
    pub expr_to_item: HashMap<ExprId, ItemId>, // Ident -> declaring Item
}

#[derive(Debug, Clone, Copy)]
pub enum Def {
    Proc(ItemId),
    Const(ItemId),
    Static(ItemId),
    Struct(ItemId),
    Union(ItemId),
    Enum(ItemId),
}
impl Def {
    pub fn item_id(&self) -> (usize, usize) {
        match *self {
            Def::Proc(i) => i,
            Def::Const(i) => i,
            Def::Struct(i) => i,
            Def::Enum(i) => i,
            Def::Union(i) => i,
            Def::Static(i) => i,
        }
    }
}

pub struct SymbolTable {
    pub globals: HashMap<String, Def>,
}

#[derive(Debug, Clone)]
pub struct SemaError {
    pub source_id: usize,
    pub span: Span,
    pub message: String,
}

pub struct Analyzer<'a> {
    programs: Vec<&'a Program>,
    syms: SymbolTable,

    tys: TyTable,
    res: ResTable,
    diags: Vec<SemaError>,

    // track this for error emission
    current_source_id: usize,

    // current function context
    locals: HashMap<String, (Ty, Span)>,

    scope_stack: Vec<SemaScope>,
    generic_context: GenericContext<Ty>,

    // For every addrof, need to mark a local as stack-allocable
    // This holds true for params too
    addr_of_by_fn: HashMap<String, Vec<String>>,
    // Which means the current function must be tracked
    current_function: String,

    // track static strings so that we can know the size of the data section before
    // function body codegen
    static_strings: HashSet<String>,

    // To allow path head elision
    elision_ctx_stack: Vec<(String, Vec<Ty>)>,
}

impl<'a> Analyzer<'a> {
    pub fn new(sources: &'a Vec<SourceFile>) -> Self {
        Self {
            programs: sources.iter().map(|s| &s.program).collect(),
            //sources,
            syms: SymbolTable {
                globals: HashMap::new(),
            },
            tys: TyTable {
                expr: HashMap::new(),
                item: HashMap::new(),
            },
            res: ResTable {
                expr_to_item: HashMap::new(),
            },
            diags: vec![],
            locals: HashMap::new(),
            current_source_id: 0,
            scope_stack: vec![],
            // Generics
            //struct_generic_map: HashMap::new(),
            //proc_generic_map: HashMap::new(),
            generic_context: GenericContext::new(false),

            addr_of_by_fn: HashMap::new(),
            current_function: String::new(),

            static_strings: HashSet::new(),

            elision_ctx_stack: vec![],
        }
    }

    fn is_in_scope(&self, name: &str) -> bool {
        for scope in &self.scope_stack {
            if scope.vars.contains(name) {
                return true;
            }
        }
        return false;
    }

    // Causes annoying borrow issues
    fn _get_program_item(&self, item: (usize, usize)) -> &Item {
        let program = self.programs[item.0];
        return &program.items[item.1];
    }

    pub fn run(
        mut self,
        sources: &Vec<SourceFile>,
    ) -> (
        SymbolTable,
        TyTable,
        HashSet<String>,
        Vec<SemaError>,
        HashMap<String, Vec<String>>,
    ) {
        // TODO: This gives all files access to the same globals
        self.scope_stack.push(SemaScope {
            vars: HashSet::new(),
        });

        for source in sources {
            self.current_source_id = source.id;
            let program_id = source.id;
            let syms = match collect_globals(&source.program, program_id) {
                Ok(s) => s,
                Err(e) => {
                    self.diags.push(e);
                    continue;
                }
            };
            self.syms.globals.extend(syms.globals);
        }

        for source in sources {
            self.current_source_id = source.id;
            let program_id = source.id;

            for (idx, it) in source.program.items.iter().enumerate() {
                match it {
                    Item::ProcDef(p) => self.check_proc((program_id, idx), p, false),
                    Item::ConstDef(c) => self.check_const((program_id, idx), c, false),
                    Item::StructDef(s) => self.check_struct((program_id, idx), s),
                    Item::UnionDef(s) => self.check_union((program_id, idx), s),
                    Item::ExternProc(e) => self.check_extern_proc((program_id, idx), e),
                    Item::EnumDef(e) => self.check_enum_def((program_id, idx), e),
                    Item::IncludeStmt(_include_stmt) => {}
                    Item::StaticDef(static_def) => self.check_static((program_id, idx), static_def),
                }
            }
        }

        if self.diags.len() > 0 {
            return (
                self.syms,
                self.tys,
                self.static_strings,
                self.diags,
                self.addr_of_by_fn,
            );
        }

        // Second pass, once top levels are resolved
        for source in sources {
            self.current_source_id = source.id;
            let program_id = source.id;

            for (idx, it) in source.program.items.iter().enumerate() {
                match it {
                    Item::ProcDef(p) => self.check_proc((program_id, idx), p, true),
                    Item::ConstDef(c) => self.check_const((program_id, idx), c, true),
                    _ => {}
                }
            }
        }
        self.scope_stack.pop();

        (
            self.syms,
            self.tys,
            self.static_strings,
            self.diags,
            self.addr_of_by_fn,
        )
    }

    // Check struct DEF, not literal / ty
    fn check_struct(&mut self, idx: ItemId, sdef: &StructDef) {
        let struct_ty = Ty::Struct(sdef.name.text.clone(), vec![]);
        self.tys.item.insert(idx, struct_ty);
    }

    // Pretty sure I don't use these anywasys. Should remove that at some point
    fn check_union(&mut self, idx: ItemId, udef: &UnionDef) {
        let union_ty = Ty::Union(udef.name.text.clone(), vec![]);
        self.tys.item.insert(idx, union_ty);
    }

    fn check_extern_proc(&mut self, idx: ItemId, extr: &ExternalProcDef) {
        let mut param_tys = Vec::new();
        for prm in &extr.params {
            let ty = self.ast_to_ty(&prm.ty);
            param_tys.push(ty.clone());
        }
        let ret_ty = self.ast_to_ty(&extr.ret_type);
        let fn_ty = Ty::FuncPtr(param_tys.clone(), Box::new(ret_ty.clone()));
        self.tys.item.insert(idx, fn_ty);
    }

    fn check_enum_def(&mut self, idx: ItemId, _edef: &EnumDef) {
        let enum_ty = Ty::U32;
        self.tys.item.insert(idx, enum_ty);
    }

    fn check_proc(&mut self, idx: ItemId, pd: &ProcDef, check_block: bool) {
        // Push procedure name to the global scope
        self.scope_stack
            .last_mut()
            .expect("procedure is outside of global scope (somehow)")
            .vars
            .insert(pd.name.text.clone());

        self.locals.clear();

        // Function scope ABOVE the block
        self.scope_stack.push(SemaScope {
            vars: pd.params.iter().map(|p| p.name.text.to_owned()).collect(),
        });

        // param types (as written)
        let mut param_tys = Vec::new();
        for prm in &pd.params {
            let ty = self.ast_to_ty(&prm.ty);
            param_tys.push(ty.clone());
            self.locals
                .insert(prm.name.text.clone(), (ty, prm.name.span.clone()));
        }
        let ret_ty = self.ast_to_ty(&pd.ret_type);
        let fn_ty = Ty::FuncPtr(param_tys.clone(), Box::new(ret_ty.clone()));
        self.tys.item.insert(idx, fn_ty);

        self.current_function = pd.name.text.to_string();
        if check_block {
            let eval_ty = self.check_expr(&pd.body);

            if eval_ty != ret_ty {
                self.error(
                    pd.name.span.clone(),
                    &format!("type mismatch: expected {:?} got {:?}", ret_ty, eval_ty),
                );
            }

            // if !matches!(pd.ret_type, Type::Void) && !block_always_returns(&pd.body) {
            //     self.error(
            //         pd.name.span.clone(),
            //         "all paths are not guaranteed to return a value",
            //     );
            // }
        }

        self.scope_stack.pop();
    }

    fn check_assign(&mut self, lhs: &Expr, rhs: &Expr) {
        let lhs_ty = self.check_expr(lhs);

        let rhs_ty = self.with_elision_ctx(&lhs_ty, |sema| sema.check_expr(rhs));

        let pointer_assign =
            matches!(lhs_ty, Ty::Ptr(_)) && matches!(rhs_ty, Ty::Ptr(_) | Ty::NullPtr(_));

        if !pointer_assign && lhs_ty != rhs_ty {
            self.error(
                lhs.span.clone(),
                &format!("cannot assign type {:?} to type {:?}", rhs_ty, lhs_ty),
            );
        }
    }

    // lol
    fn check_stmt_2(&mut self, st: &Stmt) {
        let Some(Def::Proc(procdef_id)) = self.syms.globals.get(&self.current_function) else {
            panic!()
        };
        let Item::ProcDef(ref current_proc) = self.programs[procdef_id.0].items[procdef_id.1]
        else {
            panic!()
        };
        let expected_ret = &self.ast_to_ty(&current_proc.ret_type);

        self.check_stmt(st, expected_ret);
    }

    fn check_stmt(&mut self, st: &Stmt, expected_ret: &Ty) {
        match st {
            Stmt::Let(id, ast_ty, init) => {
                let init_ty = if let Some(init_val) = init {
                    Some(self.check_expr(init_val))
                } else {
                    None
                };

                let decl_ty = if matches!(ast_ty, &ast::Type::Inferred(_)) {
                    let Some(resolved_init) = &init_ty else {
                        self.error(
                            id.span.clone(),
                            &"let expressions without init values must be explicitly typed"
                                .to_string(),
                        );
                        return;
                    };

                    // Nullptrs have no way of inferring type
                    if matches!(
                        init,
                        Some(ast::Expr {
                            kind: ExprKind::Literal(Literal::Null(_)),
                            id: _,
                            span: _
                        })
                    ) {
                        self.error(id.span.clone(), "null literals must be explicitly typed");
                        return;
                    }

                    resolved_init.clone()
                } else {
                    if let Some(init) = &init_ty {
                        self.with_elision_ctx(&init, |sema| sema.ast_to_ty(ast_ty))
                    } else {
                        self.ast_to_ty(ast_ty)
                    }
                };

                // Re-init as known type
                let init_ty = init_ty.unwrap_or(decl_ty.clone());

                // Don't allow shadowing for now
                if self.is_in_scope(&id.text) {
                    self.error(id.span.clone(), &format!("re-definition of '{}'", &id.text));
                    return;
                }

                self.scope_stack
                    .last_mut()
                    .expect("no scope for let stmt")
                    .vars
                    .insert(id.text.to_owned());

                let allowed_array = |tdecl: &Ty, tinit: &Ty| {
                    if let Ty::Array(declty, declsize) = tdecl {
                        if let Ty::Array(initty, initsize) = tinit {
                            return declty == initty
                                && (initsize == declsize || declsize == &0_usize);
                        }
                    }
                    false
                };
                // TODO: Hacking in letting any pointer assign to another pointer
                if decl_ty != init_ty
                    && init_ty != Ty::Error
                    && !matches!((&decl_ty, &init_ty), (Ty::Ptr(_), Ty::NullPtr(_)))
                    && !(matches!(&decl_ty, Ty::Ptr(_)) && matches!(&init_ty, Ty::Ptr(_)))
                    && !(matches!((&decl_ty, &init_ty), (Ty::Array(_, _), Ty::Array(_, 1))))
                    && !allowed_array(&decl_ty, &init_ty)
                {
                    let fmt_error = format!(
                        "type mismatch in let: declared='{:?} init='{:?}'",
                        decl_ty, init_ty
                    );
                    self.error(
                        if let Some(init_exp) = init {
                            init_exp.span.clone()
                        } else {
                            id.span.clone()
                        },
                        &fmt_error,
                    );
                }

                self.locals
                    .insert(id.text.clone(), (decl_ty.clone(), id.span.clone()));
            }
            Stmt::ExprStmt(e) => {
                self.check_expr(e);
            }
            Stmt::Return(Some(e)) => {
                let rt = self.check_expr(e);
                if &rt != expected_ret
                    && !(matches!(rt, Ty::NullPtr(_)) && matches!(expected_ret, Ty::Ptr(_)))
                {
                    self.error(
                        e.span.clone(),
                        &format!("bad return type. expected {:?} got {:?}", expected_ret, rt),
                    );
                }
            }
            Stmt::Return(None) => {
                if *expected_ret != Ty::Void {
                    // No span assoc with stmts yet
                    let Some(Def::Proc(id)) = self.syms.globals.get(&self.current_function) else {
                        panic!("symbol {} not found", self.current_function);
                    };
                    let Item::ProcDef(proc_def) = &self.programs[id.0].items[id.1] else {
                        panic!("symbol {} is not a proc def", self.current_function);
                    };

                    self.error(
                        proc_def.name.span.clone(),
                        &format!(
                            "void return when expected type for proc is {:?}",
                            expected_ret
                        ),
                    );
                }
            }
            Stmt::While(c, b) => {
                if self.check_expr(c) != Ty::Bool {
                    self.error(c.span.clone(), "while cond must be bool");
                }

                let unit = self.check_expr(b);
                if unit != Ty::Void {
                    self.error(
                        b.span.clone(),
                        "while body expressions must evaluate to void",
                    );
                }
            }
            Stmt::For(init_opt, condit_opt, postloop_opt, body) => {
                self.scope_stack.push(SemaScope {
                    vars: HashSet::new(),
                });

                if let Some(init) = init_opt.as_ref() {
                    self.check_stmt(init, expected_ret); // Not sure how to handle expected ret ..
                }

                if let Some(condit) = condit_opt {
                    if self.check_expr(condit) != Ty::Bool {
                        self.error(condit.span.clone(), "while cond must be bool");
                    }
                }

                if let Some(postloop) = postloop_opt.as_ref() {
                    self.check_stmt(postloop, expected_ret);
                }

                let unit = self.check_expr(&body);
                if unit != Ty::Void {
                    self.error(
                        body.span.clone(),
                        "for loop body expressions must evaluate to void",
                    );
                }

                self.scope_stack.pop();
            }
            //Stmt::Block(b) => self.check_block(b, expected_ret),
            Stmt::Break => {
                // TODO: Check that this is valid
            }
            Stmt::Continue => {
                // TODO: Check that this is valid
            }
            Stmt::Assign(lhs_expr, rhs_expr) => self.check_assign(lhs_expr, rhs_expr),
            // _ => {
            //     println!("[jamc DEBUG] statement {:?} not implemented", st);
            // }
        }
    }

    fn check_expr(&mut self, e: &Expr) -> Ty {
        if let Some(t) = self.tys.expr.get(&e.id) {
            return t.clone();
        }

        let ty = match &e.kind {
            // Blocks as expressions
            ExprKind::Block(stmts, tail) => {
                let mut ret_expression_ty = None;

                self.scope_stack.push(SemaScope {
                    vars: HashSet::new(),
                });

                for stmt in stmts {
                    self.check_stmt_2(stmt);

                    if let Stmt::Return(Some(expr)) = stmt {
                        let stmt_ret_ty = self.check_expr(expr);
                        if let Some(ret_other) = ret_expression_ty {
                            if ret_other != stmt_ret_ty {
                                self.error(expr.span.clone(), "return types do not match");
                                return Ty::Error;
                            }
                        }
                        ret_expression_ty = Some(stmt_ret_ty);
                    }
                }

                // This way empty blocks eval to void as well
                let ret = if let Some(tail_expr) = tail {
                    self.check_expr(&tail_expr)
                } else if let Some(ret_expr) = &ret_expression_ty {
                    ret_expr.clone()
                } else {
                    Ty::Void
                };

                if let Some(ret_expr_ty) = ret_expression_ty {
                    if ret_expr_ty != ret {
                        self.error(e.span.clone(), "return types do not match");
                        return Ty::Error;
                    }
                }

                self.scope_stack.pop();

                let ret = self.reresolve(&ret);
                //println!("Block evaluated to ty {:?}", ret);

                ret
            }
            // int literals
            ExprKind::Literal(Literal::Int(_, 8, false)) => Ty::U8,
            ExprKind::Literal(Literal::Int(_, 16, false)) => Ty::U16,
            ExprKind::Literal(Literal::Int(_, 32, false)) => Ty::U32,
            ExprKind::Literal(Literal::Int(_, 64, false)) => Ty::U64,

            ExprKind::Literal(Literal::Int(_, 8, true)) => Ty::I8,
            ExprKind::Literal(Literal::Int(_, 16, true)) => Ty::I16,
            ExprKind::Literal(Literal::Int(_, 32, true)) => Ty::I32,
            ExprKind::Literal(Literal::Int(_, 64, true)) => Ty::I64,

            ExprKind::Literal(Literal::USize(_)) => Ty::USize,

            // float literals
            ExprKind::Literal(Literal::Float(_, true)) => Ty::F32,
            ExprKind::Literal(Literal::Float(_, false)) => Ty::F64,

            ExprKind::Literal(Literal::Bool(_)) => Ty::Bool,
            ExprKind::Literal(Literal::Str(literal)) => {
                if !self.static_strings.contains(literal) {
                    self.static_strings.insert(literal.to_owned());
                }
                Ty::Ptr(Box::new(Ty::I8))
            }

            ExprKind::Literal(Literal::Null(null_ty)) => {
                Ty::NullPtr(Box::new(self.ast_to_ty(null_ty)))
            }

            ExprKind::Literal(Literal::SizeOf(_)) => Ty::U32, // for 32bit wasm

            ExprKind::StructLit {
                name,
                fields,
                generic_inst_tys: generic_concrete_tys,
            } => {
                let ty = if let Some(Def::Struct(i)) = self.syms.globals.get(&name.text) {
                    // Check the item type that defines this literal. It should only be a struct
                    match &self.programs[i.0].items[i.1] {
                        Item::StructDef(struct_def) => {
                            if struct_def.generics.len() != generic_concrete_tys.len() {
                                self.error(
                                    name.span.clone(),
                                    &format!(
                                        "exptected {} generic args, got {}",
                                        struct_def.generics.len(),
                                        generic_concrete_tys.len()
                                    ),
                                );
                                return Ty::Error;
                            }

                            let ctx = self.create_map_from_generic_tys(
                                &generic_concrete_tys,
                                &struct_def.generics,
                            );

                            self.generic_context.push(ctx);

                            let mut all_fields = true;

                            for (field_ident, field_expr) in fields {
                                let cannonical_field =
                                    self.lookup_field_ty(&struct_def.name.text, &field_ident.text);
                                if let Some(cannonical_ty) = cannonical_field {
                                    // set pathhead elision ctx
                                    let written_ty = self
                                        .with_elision_ctx(&cannonical_ty, |sema| {
                                            sema.check_expr(&field_expr)
                                        });

                                    let both_pointers = matches!(cannonical_ty, Ty::Ptr(_))
                                        && matches!(written_ty, Ty::Ptr(_) | Ty::NullPtr(_));

                                    if !both_pointers && cannonical_ty != written_ty {
                                        self.error(
                                            e.span.clone(),
                                            &format!(
                                                "expected type {:?} for field '{}', got {:?}",
                                                cannonical_ty, field_ident.text, written_ty
                                            ),
                                        );
                                        all_fields = false;
                                    }
                                } else {
                                    self.error(
                                        e.span.clone(),
                                        &format!(
                                            "struct {} does not have a field {}. Options: {}",
                                            &name.text,
                                            &field_ident.text,
                                            struct_def
                                                .fields
                                                .iter()
                                                .map(|f| f.0.text.clone())
                                                .collect::<Vec<String>>()
                                                .join(", ")
                                        ),
                                    );
                                    all_fields = false;
                                }
                            }

                            self.generic_context.pop();
                            if all_fields {
                                Ty::Struct(
                                    name.text.clone(),
                                    generic_concrete_tys
                                        .iter()
                                        .map(|t| self.ast_to_ty(t))
                                        .collect(),
                                )
                            } else {
                                Ty::Error
                            }
                        }
                        _ => {
                            self.error(e.span.clone(), "item referenced is not a struct");
                            Ty::Error
                        }
                    }
                } else {
                    Ty::Error
                };

                ty
            }
            ExprKind::ArrayLit(expressions) => {
                if expressions.len() < 1 {
                    //error
                    self.diags.push(SemaError {
                        source_id: self.current_source_id,
                        span: e.span.clone(),
                        message: "0 size array literals are not allowed".to_string(),
                    });
                    self.err_ty(e)
                } else {
                    let ele_ty = self.check_expr(&expressions[0]);
                    for ele_expr in expressions.iter().skip(1) {
                        let expr_ty = self.check_expr(ele_expr);

                        if expr_ty != ele_ty {
                            self.error(
                                ele_expr.span.clone(),
                                &format!("elements of an array must all be of the same type"),
                            );
                            return Ty::Error;
                        }
                    }
                    Ty::Array(Box::new(ele_ty), expressions.len())
                }
            }

            ExprKind::ScopedIdent(id) => {
                let ident_ty = self.resolve_ident(e, id);
                if let Ty::FuncGeneric(generic_name, _idx) = &ident_ty {
                    if let Some(mapping) = self.generic_context.try_get_mapping(&generic_name) {
                        //println!("Mapping ident {} to {:?}", generic_name, mapping);
                        return mapping.to_owned();
                    }
                }

                ident_ty
            }
            ExprKind::Unary(op, inner) => {
                let it = self.check_expr(inner);
                match op {
                    UnOp::Neg if it.is_numeric() => it,
                    UnOp::Neg => self.err_ty(e),
                    UnOp::Not if it == Ty::Bool => Ty::Bool,
                    UnOp::Not => self.err_ty(e),
                    UnOp::AddrOf => {
                        // TODO: Add addrof tracker for expression
                        if let ExprKind::ScopedIdent(scoped_ident) = &inner.kind {
                            // should let me use this in codegen for ensuring stack allocations
                            self.addr_of_by_fn
                                .entry(self.current_function.clone())
                                .or_insert_with(Vec::new)
                                .push(scoped_ident.to_string());
                        }
                        match it {
                            Ty::FuncPtr(_, _) => it,
                            _ => Ty::Ptr(Box::new(it)),
                        }
                    }
                    UnOp::Deref => {
                        if let Ty::Ptr(pointee) = it {
                            *pointee
                        } else {
                            self.diags.push(SemaError {
                                source_id: self.current_source_id,
                                span: e.span.clone(),
                                message: format!(
                                    "cannot dereference - expression does not evaluate to a pointer"
                                ),
                            });
                            self.err_ty(e)
                        }
                    }
                    UnOp::MaybeDeref => {
                        if let Ty::Ptr(pointee) = it {
                            *pointee
                        } else {
                            it
                        }
                    }
                }
            }
            ExprKind::Binary(l, op, r) => self.check_bin(e, *op, l, r),
            ExprKind::Call(callee, args, generic_args) => {
                // TODO: set a generic context so that return types can be resolved
                let mut ctx = HashMap::new();
                if let ExprKind::ScopedIdent(sident) = &callee.kind {
                    if let Some(local) = self.locals.get(&sident.to_string()) {
                        if let Ty::FuncPtr(fptr_args, _ret) = &local.0 {
                            if args.len() != fptr_args.len() {
                                self.error(local.1.clone(), "incorrect number of args for call");
                                return Ty::Error;
                            }

                            //return *ret.clone();
                        } else {
                            panic!(
                                "cannot call non with non function pointer locals (ty = {:?})",
                                local.0
                            );
                        }
                    } else {
                        let Some(def) = self.syms.globals.get(&sident.to_string()) else {
                            self.error(
                                sident.ident.span.clone(),
                                &format!("Could not find identifier {}", sident.to_string()),
                            );
                            return Ty::Error;
                        };
                        if let Item::ProcDef(p) =
                            &self.programs[def.item_id().0].items[def.item_id().1]
                        {
                            if p.generics.len() != generic_args.len() {
                                self.error(
                                    callee.span.clone(),
                                    &format!("expected {} generic args", p.generics.len()),
                                );
                                return Ty::Error;
                            }

                            for (idx, g_arg) in p.generics.iter().enumerate() {
                                let Type::Generic {
                                    generic_name: g_ident,
                                    index_in_decl: _,
                                } = g_arg
                                else {
                                    panic!()
                                };
                                if !matches!(&generic_args[idx], Type::Generic { generic_name, index_in_decl: _ } if generic_name == g_ident)
                                {
                                    ctx.insert(
                                        g_ident.text.to_owned(),
                                        self.ast_to_ty(&generic_args[idx]),
                                    );
                                }
                            }
                        }
                    }
                }

                if !generic_args.is_empty() {
                    self.generic_context.push(ctx);
                }

                let val = self.check_call(e, callee, args);

                if !generic_args.is_empty() {
                    self.generic_context.pop();
                }

                val
            }
            ExprKind::MethodCall {
                receiver: reciever,
                method_name,
                args,
                generic_args,
            } => {
                let mut ctx = HashMap::new();
                let recieve_ty = self.check_expr(&reciever);

                let reciver_ns_ty = if let Ty::Ptr(inner_ptr) = &recieve_ty {
                    *inner_ptr.clone()
                } else {
                    recieve_ty.clone()
                };

                let Some(ns_string) = reciver_ns_ty.to_ns_string() else {
                    self.error(reciever.span.clone(), "could not convert type to namespace");
                    return Ty::Error;
                };
                let scoped_method_name = format!("{}::{}", ns_string, method_name.text);

                let Some(def) = self.syms.globals.get(&scoped_method_name) else {
                    self.error(
                        method_name.span.clone(),
                        &format!("could not find function: {}", scoped_method_name),
                    );
                    return Ty::Error;
                };

                if let Item::ProcDef(p) = &self.programs[def.item_id().0].items[def.item_id().1] {
                    let Ty::Struct(_sname, mut sgenerics) = reciver_ns_ty else {
                        panic!()
                    };

                    sgenerics.append(
                        &mut generic_args
                            .iter()
                            .map(|g| self.ast_to_ty(g))
                            .collect::<Vec<Ty>>(),
                    );

                    if p.generics.len() != (sgenerics.len()) {
                        self.error(
                            method_name.span.clone(),
                            &format!("expected {} generic args", p.generics.len()),
                        );
                        return Ty::Error;
                    }

                    for (idx, g_arg) in p.generics.iter().enumerate() {
                        let Type::Generic {
                            generic_name,
                            index_in_decl: _,
                        } = g_arg
                        else {
                            panic!()
                        };

                        if !matches!(sgenerics[idx], Ty::FuncGeneric(ref name, _) if *name == generic_name.text)
                        {
                            ctx.insert(generic_name.text.to_owned(), sgenerics[idx].clone());
                        }
                    }
                }

                self.generic_context.push(ctx);

                let val = self.check_method_call(reciever, method_name, args);
                self.generic_context.pop();

                val
            }
            ExprKind::Cast(expr, tgt_ast) => {
                let src_ty = self.check_expr(expr);
                let tgt_ty = self.ast_to_ty(tgt_ast);
                if self.cast_ok(&src_ty, &tgt_ty) {
                    tgt_ty
                } else {
                    self.diags.push(SemaError {
                        source_id: self.current_source_id,
                        span: e.span.clone(),
                        message: format!("Failed to make cast type: {:?}", expr),
                    });
                    self.err_ty(e)
                }
            }
            //ExprKind::EnumVarLit { scope, variant } => self.check_enum_lit(e, scope, variant),
            ExprKind::Field(base, field_ident, uses_deref) => {
                let base_ty = self.check_expr(&base);

                let is_deref = matches!(base.kind, ExprKind::Unary(UnOp::Deref, _));
                let requires_deref = matches!(base_ty, Ty::Ptr(_));

                if requires_deref && !is_deref {
                    self.error(e.span.clone(), &format!("requires deref '->'"));
                }

                if !is_deref && *uses_deref {
                    self.error(
                        e.span.clone(),
                        &format!("cannot dereference non pointer type. use '.'"),
                    );
                }

                let struct_name_and_generics: (&str, &Vec<Ty>) = match &base_ty {
                    Ty::FuncGeneric(_name, _idx_ind_decl) => {
                        // Not sure if I would ever need this
                        todo!("haven't implemented this yet")
                    }
                    Ty::Struct(name, generics) => (name.as_str(), generics),
                    Ty::Ptr(inner) => match &**inner {
                        Ty::Struct(name, generics) => (name.as_str(), generics),
                        _ => {
                            self.error(
                                field_ident.span.clone(),
                                &format!("Can only get a field of a struct or struct pointer (found: {:?})", *inner),
                            );
                            return Ty::Error;
                        }
                    },
                    _ => {
                        self.error(
                            field_ident.span.clone(),
                            &format!(
                                "Can only get a field of a struct or struct pointer (found: {:?})",
                                base_ty
                            ),
                        );
                        return Ty::Error;
                    }
                };

                /////////////// generic mapping //////////////////////
                let Some(Def::Struct(i)) = self.syms.globals.get(struct_name_and_generics.0) else {
                    panic!()
                };
                // Check the item type that defines this literal. It should only be a struct
                let Item::StructDef(struct_def) = &self.programs[i.0].items[i.1] else {
                    panic!()
                };
                let mut ctx = HashMap::new();
                if struct_def.generics.len() != struct_name_and_generics.1.len() {
                    self.error(
                        base.span.clone(),
                        &format!("number of generic args do not match!"),
                    );
                    return Ty::Error;
                }

                for (idx, def_generic) in struct_def.generics.iter().enumerate() {
                    let crate::frontend::ast::Type::Generic {
                        generic_name: g_ident,
                        index_in_decl: _,
                    } = def_generic
                    else {
                        panic!()
                    };
                    let concrete = self.reresolve(&struct_name_and_generics.1[idx]);
                    if !matches!(&concrete, Ty::FuncGeneric(gname, _gindex) if *gname == g_ident.text)
                    {
                        ctx.insert(g_ident.text.clone(), concrete);
                    }
                }

                self.generic_context.push(ctx);
                ////////////////////////////////////////////////////

                let (struct_name, _generics) = struct_name_and_generics;
                let ty = self
                    .lookup_field_ty(struct_name, &field_ident.text)
                    .unwrap_or_else(|| {
                        self.error(
                            field_ident.span.clone(),
                            &format!("`{}` has no field `{}`", struct_name, field_ident.text),
                        );
                        Ty::Error
                    });
                self.generic_context.pop();

                ty
            }
            ExprKind::Index(base, idex_expr) => {
                let base_ty = self.check_expr(&base);
                let idex_ty = self.check_expr(&idex_expr);

                if !matches!(idex_ty, Ty::U32 | Ty::U64 | Ty::USize) {
                    self.error(
                        e.span.clone(),
                        &format!("cannot index with type {:?}", idex_ty),
                    );
                    Ty::Error
                } else {
                    match base_ty {
                        Ty::Array(elem_ty, arr_len) => {
                            if let ExprKind::Literal(lit) = &idex_expr.kind {
                                let val = match lit {
                                    Literal::Int(value, _, _) => *value as usize,
                                    Literal::USize(value) => *value as usize,
                                    _ => 0,
                                };
                                // static access checking. could expand to const expressions
                                if val >= arr_len {
                                    self.error(
                                        e.span.clone(),
                                        &format!(
                                            "index value {} is out of bounds for array size {}",
                                            val, arr_len
                                        ),
                                    );
                                    return Ty::Error;
                                }
                            }

                            *elem_ty.clone()
                        }
                        Ty::Ptr(elem_ty) => *elem_ty.clone(),
                        _ => {
                            self.error(
                                e.span.clone(),
                                &format!("cannot index into type {:?}", base_ty),
                            );
                            Ty::Error
                        }
                    }
                }
            }
            ExprKind::FunctionReference(func_ref) => {
                let ident_ty = self.resolve_ident(e, &func_ref.path);
                let Ty::FuncPtr(params, ret) = ident_ty else {
                    self.error(
                        e.span.clone(),
                        &format!(
                            "ident {:?} does not refer to a function",
                            func_ref.path.to_string()
                        ),
                    );
                    return Ty::Error;
                };

                // I think I need to get the def that this refers to, and map the generics from def -> funcref generics
                let Def::Proc(item_id) = self.syms.globals[&func_ref.path.to_string()] else {
                    self.error(
                        e.span.clone(),
                        &format!(
                            "ident {:?} does not refer to a function",
                            func_ref.path.to_string()
                        ),
                    );
                    return Ty::Error;
                };

                let mut ctx = HashMap::new();
                let Item::ProcDef(proc_def) = &self.programs[item_id.0].items[item_id.1] else {
                    panic!("couldn't find proc")
                };
                for (idx, generic_ty) in proc_def.generics.iter().enumerate() {
                    let Type::Generic {
                        generic_name,
                        index_in_decl: _,
                    } = generic_ty
                    else {
                        panic!("generic type is not generic")
                    };

                    // If the types aren't equal, add a mapping
                    if *generic_ty != func_ref.generic_args[idx] {
                        ctx.insert(
                            generic_name.text.to_owned(),
                            self.ast_to_ty(&func_ref.generic_args[idx]),
                        );
                    }
                }
                self.generic_context.push(ctx);
                // Do I need to add a generic context for the concretes here? Probably, otherwise the return type could be wrong
                let resolved_params: Vec<Ty> = params.iter().map(|p| self.reresolve(p)).collect();
                let resolved_ret = self.reresolve(&ret);
                self.generic_context.pop();

                Ty::FuncPtr(resolved_params, Box::new(resolved_ret))
            }
            ExprKind::UnionLit(path_head, variant_ident, init_expr_opt, generic_inst_tys) => {
                // Currently only place where elision ctx is valid
                let mut elided_generics = None;
                let path_head_str = if path_head.to_string() == "_"
                    && let Some(last) = self.elision_ctx_stack.last()
                {
                    elided_generics = Some(last.1.clone());
                    &last.0
                } else {
                    &path_head.to_string()
                };

                let Some(Def::Union(def)) = self.syms.globals.get(path_head_str).cloned() else {
                    self.error(
                        e.span.clone(),
                        &format!("sum type {:?} is not defined", path_head.to_string()),
                    );
                    return Ty::Error;
                };

                self.check_union_lit_expr(
                    e,
                    &def,
                    variant_ident,
                    init_expr_opt,
                    &generic_inst_tys,
                    &elided_generics,
                )
            }
            ExprKind::Match(decomp_expr) => {
                // Need to check the type of the expression to make sure it is a union
                let to_decomp_ty = self.check_expr(&decomp_expr.union_to_decomp);
                let Ty::Union(base_ty_ident, generics) = to_decomp_ty else {
                    self.error(
                        e.span.clone(),
                        "decomp expressions are currently only valid on sum types",
                    );
                    return Ty::Error;
                };

                // If to_decomp_ty can have generic instances defined, need to creat a context from that

                let Some(Def::Union(union_def_id)) = self.syms.globals.get(&base_ty_ident) else {
                    self.error(
                        e.span.clone(),
                        &format!("failed to find sum type def {:?}", base_ty_ident),
                    );
                    return Ty::Error;
                };

                let Item::UnionDef(union_def) =
                    &self.programs[union_def_id.0].items[union_def_id.1]
                else {
                    self.error(
                        e.span.clone(),
                        &format!("failed to lookup sum type def {:?}", base_ty_ident),
                    );
                    return Ty::Error;
                };

                let ctx = self.create_map_from_generic_tys_2(&generics, &union_def.generics);
                self.generic_context.push(ctx);

                let mut eval_tys = vec![];

                let mut all_arms: Vec<&ast::MatchArm> = decomp_expr.arms.iter().by_ref().collect();
                if let Some(default_arm) = &decomp_expr.default_arm {
                    all_arms.push(&default_arm);
                }

                let mut required_arms: HashSet<&Ident> =
                    union_def.variants.iter().map(|v| &v.0).collect();

                for (idx, arm) in all_arms.iter().enumerate() {
                    required_arms.remove(&arm.variant_ident);

                    let def_ty = if let Some(def_ty) =
                        union_def.variants.iter().find(|v| v.0 == arm.variant_ident)
                    {
                        def_ty.clone()
                    } else {
                        if decomp_expr.default_arm.is_some() && idx == all_arms.len() - 1 {
                            let ret = (Ident::dummy(Span::default()), Type::Void);
                            ret
                        } else {
                            self.error(
                                e.span.clone(),
                                &format!(
                                    "could not find variant {:?} in sum type {:?}",
                                    arm.variant_ident.text, base_ty_ident
                                ),
                            );
                            return Ty::Error;
                        }
                    };

                    // Scope for each arm
                    self.scope_stack.push(SemaScope {
                        vars: HashSet::new(),
                    });
                    if let Some(binding) = &arm.binding {
                        // I think I need to insert the type of the ident here
                        if self.is_in_scope(&binding.text) {
                            self.error(
                                binding.span.clone(),
                                &format!("re-definition of '{}'", &binding.text),
                            );
                            return Ty::Error;
                        }

                        let def_sema_ty = self.ast_to_ty(&def_ty.1);
                        self.locals
                            .insert(binding.text.to_owned(), (def_sema_ty, binding.span.clone()));

                        self.scope_stack
                            .last_mut()
                            .unwrap()
                            .vars
                            .insert(binding.text.to_owned());

                        if def_ty.1 == Type::Void {
                            self.error(
                                e.span.clone(),
                                &format!("variant {:?} expected 0 binding args", def_ty.0.text),
                            );
                            return Ty::Error;
                        }
                    } else if def_ty.1 != Type::Void {
                        self.error(
                            e.span.clone(),
                            &format!("expected a binding for variant {:?}", def_ty.0.text),
                        );
                        return Ty::Error;
                    }

                    let eval_ty = self.check_expr(&arm.body_expression);

                    // let eval_ty = if let Some(eval_expr) = &arm.eval_expr {
                    //     self.check_expr(eval_expr)
                    // } else {
                    //     Ty::Void
                    // };

                    let mut always_ret = false;
                    if let Some(prev) = eval_tys.last() {
                        // Not 100% on this
                        if expr_always_returns(&arm.body_expression) {
                            always_ret = true;
                        }

                        if eval_ty != *prev && !always_ret {
                            self.error(e.span.clone(), &format!("all arms must evaluate to the same type, or diverge. {:?} != {:?}", eval_ty, *prev));
                            return Ty::Error;
                        }
                    }

                    // Don't care about types when body always returns
                    if !always_ret {
                        eval_tys.push(eval_ty);
                    }

                    if let Some(binding) = &arm.binding {
                        // This should def be some
                        assert!(self.locals.remove(&binding.text).is_some());
                    }

                    self.scope_stack.pop();
                }

                if decomp_expr.default_arm.is_none() && !required_arms.is_empty() {
                    let as_vec: Vec<&str> =
                        required_arms.into_iter().map(|i| i.text.as_str()).collect();
                    self.error(
                        e.span.clone(),
                        &format!(
                            "match expression is non-exhaustive and contains no `default` arm. missing: `{}`",
                            as_vec.join(", ")
                        ),
                    );
                }

                if let Some(first) = eval_tys.first() {
                    if !eval_tys.iter().all(|ty| ty == first) {
                        self.error(e.span.clone(), &format!("all arms must evaluate to the same type, or diverge. (expected {:?})", first));
                        return Ty::Error;
                    }
                }

                self.generic_context.pop();
                eval_tys.first().cloned().unwrap_or_else(|| Ty::Void)
            }
            ExprKind::If(condition, then, else_opt) => {
                if self.check_expr(condition) != Ty::Bool {
                    self.error(condition.span.clone(), "condition must be bool");
                }
                let then_ty = self.check_expr(then);
                let else_ty = if let Some(else_branch) = else_opt {
                    self.check_expr(else_branch)
                } else {
                    Ty::Void
                };

                let ret_ty = if then_ty != else_ty {
                    let then_returns = expr_always_returns(&then);
                    let else_returns = if let Some(else_expr) = else_opt {
                        expr_always_returns(&else_expr)
                    } else {
                        false
                    };

                    if !(then_returns || else_returns) {
                        self.error(
                            e.span.clone(),
                            &format!("branches diverge ({:?} != {:?})", then_ty, else_ty),
                        );
                        return Ty::Error;
                    }

                    if then_returns { else_ty } else { then_ty }
                } else {
                    then_ty
                };

                // TODO: This needs testing
                ret_ty
            }
            ExprKind::InlineWat(_mappings, type_opt, _text) => {
                self.ast_to_ty(&type_opt.as_ref().unwrap_or_else(|| &Type::Void))
            }
            _ => {
                self.diags.push(SemaError {
                    source_id: self.current_source_id,
                    span: e.span.clone(),
                    message: format!(
                        "Failed to get type for: {:?}. Expression type not mapped",
                        e
                    ),
                });
                self.err_ty(e)
            } // unimplemented kinds
        };

        //let ty = self.reresolve(&ty);
        self.tys.expr.insert(e.id, ty.clone());
        ty
    }

    // TODO-Generics this won't work in fully generic functions (and shouldn't - trait bounds should be preferred)
    fn lookup_field_ty(&mut self, struct_name: &str, field_name: &str) -> Option<Ty> {
        let Def::Struct(st_id) = *self.syms.globals.get(struct_name)? else {
            return None;
        };
        let Item::StructDef(s) = &self.programs[st_id.0].items[st_id.1] else {
            return None;
        };
        s.fields
            .iter()
            .find(|(ident, _)| ident.text == field_name)
            .map(|(_, ty)| {
                // If this field is a generic, find the index of the generic in the struct def, and use it to index into the passed in generic (concrete) args
                if let crate::frontend::ast::Type::Generic {
                    //source_ident: _,
                    generic_name,
                    index_in_decl: _,
                } = ty
                {
                    if self.generic_context.has_key(&generic_name.text) {
                        self.generic_context.get_mapping(&generic_name.text).clone()
                    } else {
                        self.ast_to_ty(ty)
                    }
                } else {
                    self.ast_to_ty(&ty)
                }
            })
    }

    fn resolve_ident(&mut self, e: &Expr, id: &ScopedIdent) -> Ty {
        if let Some((last, _enum_entry)) = id // I know this is kinda cursed
            .scopes
            .last()
            .and_then(|last| self.syms.globals.get(last).map(|e| (last, e)))
            .filter(|(_, def)| matches!(def, Def::Enum(_)))
        {
            return self.check_enum_lit(e, last, &id.ident);
        } else if let Some((ty, _span)) = self.locals.get(&id.to_string()) {
            if !self.is_in_scope(&id.to_string()) {
                self.error(
                    id.ident.span.clone(),
                    &format!("identifier {} is not in scope.", &id.to_string()),
                );
                return Ty::Error;
            }

            return match ty {
                //Ty::Enum(_) => Ty::U32,
                _ => ty.clone(),
            };
            //return ty.clone();
        }
        if let Some(def) = self.syms.globals.get(&id.to_string()) {
            self.res.expr_to_item.insert(e.id, def.item_id());
            match def {
                Def::Const(idx) => self.tys.item[idx].clone(),
                Def::Static(idx) => self.tys.item[idx].clone(),
                Def::Proc(idx) => self.tys.item[idx].clone(), // already FuncPtr
                Def::Struct(idx) => self.tys.item[idx].clone(),
                Def::Union(idx) => self.tys.item[idx].clone(),
                Def::Enum(_) => Ty::I32, // enums as int for now.
            }
        } else {
            // TODO: Check for intrinsic
            self.error(
                id.ident.span.clone(),
                &format!(
                    "unknown identifier '{}'",
                    &id.to_string(), //self.syms.globals
                ),
            );
            Ty::Error
        }
    }

    fn reresolve(&self, base: &Ty) -> Ty {
        match base {
            Ty::Ptr(ty) => Ty::Ptr(Box::new(self.reresolve(ty))),
            Ty::Array(ty, size) => Ty::Array(Box::new(self.reresolve(ty)), *size),
            Ty::Struct(name, items) => Ty::Struct(
                name.to_string(),
                items.iter().map(|i| self.reresolve(i)).collect(),
            ),
            Ty::FuncPtr(items, ty) => Ty::FuncPtr(
                items.iter().map(|i| self.reresolve(i)).collect(),
                Box::new(self.reresolve(ty)),
            ),
            Ty::NullPtr(ty) => Ty::NullPtr(Box::new(self.reresolve(ty))),
            Ty::FuncGeneric(name, _idex) => {
                if let Some(mapping) = self.generic_context.try_get_mapping(&name) {
                    self.reresolve(mapping)
                } else {
                    base.clone()
                }
            }
            _ => base.clone(),
        }
    }

    // This might need a generics context to resolve generic Idents
    // Generic types need to be mapped to real types for analysis
    fn ast_to_ty(&mut self, ty: &crate::frontend::ast::Type) -> Ty {
        match ty {
            crate::frontend::ast::Type::Inferred(span) => {
                self.error(
                    span.clone(),
                    format!("type cannot be inferred in this context").as_str(),
                );
                Ty::Error
            }
            crate::frontend::ast::Type::Primitive(prim_ty) => {
                match prim_ty {
                    ast::PrimTy::I8 => Ty::I8,
                    ast::PrimTy::I16 => Ty::I16,
                    ast::PrimTy::I32 => Ty::I32,
                    ast::PrimTy::I64 => Ty::I64,
                    ast::PrimTy::U8 => Ty::U8,
                    ast::PrimTy::U16 => Ty::U16,
                    ast::PrimTy::U32 => Ty::U32,
                    ast::PrimTy::U64 => Ty::U64,
                    ast::PrimTy::F32 => Ty::F32,
                    ast::PrimTy::F64 => Ty::F64,
                    ast::PrimTy::Bool => Ty::Bool,
                    ast::PrimTy::CStr => Ty::Ptr(Box::new(Ty::I8)), // convert to (hopefully) null terminated byte ptr.
                    ast::PrimTy::USize => Ty::USize,
                }
            }
            crate::frontend::ast::Type::Ptr(p) => Ty::Ptr(Box::new(self.ast_to_ty(&p))),
            crate::frontend::ast::Type::FuncPtr(items, ret) => Ty::FuncPtr(
                items.iter().map(|i| self.ast_to_ty(i)).collect(),
                Box::new(self.ast_to_ty(&ret)),
            ),
            crate::frontend::ast::Type::Array(arr_ty, arr_size) => {
                Ty::Array(Box::new(self.ast_to_ty(&arr_ty)), *arr_size)
            }
            crate::frontend::ast::Type::Void => Ty::Void,
            crate::frontend::ast::Type::Named(ident, generics) => {
                match self.syms.globals.get(&ident.text) {
                    Some(Def::Struct(_)) => Ty::Struct(
                        ident.text.clone(),
                        generics
                            .iter()
                            .map(|ast_ty| self.ast_to_ty(ast_ty))
                            .collect(),
                    ),
                    Some(Def::Enum(_)) => Ty::U32,
                    Some(Def::Union(_)) => Ty::Union(
                        ident.text.clone(),
                        generics
                            .iter()
                            .map(|ast_ty| self.ast_to_ty(ast_ty))
                            .collect(),
                    ),
                    _ => {
                        self.error(
                            ident.span.clone(),
                            format!("unknown type name '{}'", &ident.text).as_str(),
                        );
                        Ty::Error
                    }
                }
            }
            crate::frontend::ast::Type::Generic {
                //source_ident: _,
                generic_name,
                index_in_decl,
            } => {
                if self.generic_context.has_key(&generic_name.text) {
                    self.generic_context.get_mapping(&generic_name.text).clone()
                } else {
                    Ty::FuncGeneric(generic_name.text.clone(), *index_in_decl)
                }
            }
        }
    }

    fn error(&mut self, span: std::ops::Range<usize>, arg: &str) {
        self.diags.push(SemaError {
            source_id: self.current_source_id,
            span: span,
            message: arg.to_string(),
        });
    }

    // TODO: Casting generic pointers is weird with this
    fn cast_ok(&self, src: &Ty, tgt: &Ty) -> bool {
        use Ty::*;

        if src == tgt {
            return true;
        }

        let is_int = |t: &Ty| matches!(t, I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64);
        let is_float = |t: &Ty| matches!(t, F32 | F64);
        let _is_ptr = |t: &Ty| matches!(t, Ptr(_));

        match (src, tgt) {
            // numeric cross-casts
            (s, t)
                if (is_int(s) || is_float(s) || *s == Bool)
                    && (is_int(t) || is_float(t) || *t == Bool) =>
            {
                true
            }

            // pointer <-> pointer
            (Ptr(_), Ptr(_)) => true,

            // pointer <-> integer (uintptr)
            (Ptr(_), t) if is_int(t) => true,
            (s, Ptr(_)) if is_int(s) => true,

            (Ptr(_), NullPtr(_)) => true,
            (NullPtr(_), Ptr(_)) => true,
            (FuncPtr(_, _), t) if is_int(t) => true,
            (t, FuncPtr(_, _)) if is_int(t) => true,

            _ => false,
        }
    }

    fn err_ty(&mut self, e: &Expr) -> Ty {
        self.diags.push(SemaError {
            source_id: self.current_source_id,
            span: e.span.clone(),
            message: "this type is not valid here".to_string(),
        });
        Ty::Error
    }

    fn check_enum_lit(&mut self, expr: &Expr, scope: &str, variant: &Ident) -> Ty {
        match self.syms.globals.get(scope) {
            Some(Def::Enum(e)) => {
                let item = &self.programs[e.0].items[e.1];
                if let Item::EnumDef(enum_def) = item {
                    if enum_def.variants.iter().any(|v| v.0.text == *variant.text) {
                        Ty::U32
                    } else {
                        let options = enum_def
                            .variants
                            .iter()
                            .map(|v| v.0.text.clone())
                            .collect::<Vec<String>>()
                            .join(", ");
                        self.error(
                            expr.span.clone(),
                            &format!(
                                "enum variant '{}' not defined for {}. options are: {}",
                                variant.text, scope, options
                            ),
                        );
                        Ty::Error
                    }
                } else {
                    self.error(
                        expr.span.clone(),
                        &format!("item {:?} is not an enum def", e),
                    );
                    Ty::Error
                }
            }
            _ => {
                self.error(expr.span.clone(), &format!("enum '{}' not found", &scope));
                Ty::Error
            }
        }
    }

    // TODO: This duplicates most of check_call
    fn check_method_call(&mut self, reciever: &Expr, method_name: &Ident, args: &[Expr]) -> Ty {
        // The struct being operated on
        let recieve_ty = self.check_expr(reciever);
        let is_struct_like = match &recieve_ty {
            Ty::Struct(_, _) => true,
            Ty::Ptr(inner) => matches!(&**inner, Ty::Struct(_, _)),
            _ => false,
        };

        if !is_struct_like {
            self.error(
                reciever.span.clone(),
                &format!("cannot perform method call on non-struct or struct pointer type"),
            );
        }

        // Another place where we maybe unwrap a pointer to get the base type
        let reciver_ns_ty = if let Ty::Ptr(inner_ptr) = &recieve_ty {
            *inner_ptr.clone()
        } else {
            recieve_ty.clone()
        };

        let Some(ns_string) = reciver_ns_ty.to_ns_string() else {
            self.error(reciever.span.clone(), "could not convert type to namespace");
            return Ty::Error;
        };
        let scoped_method_name = format!("{}::{}", ns_string, method_name.text);
        let Some(def) = self.syms.globals.get(&scoped_method_name) else {
            panic!("Failed to get gobal method {}", scoped_method_name);
        };

        let id = def.item_id();
        let proc_item = &self.programs[id.0].items[id.1];
        let Item::ProcDef(ProcDef {
            name: _,
            params,
            ret_type,
            body: _,
            impls,
            generics: _,
        }) = proc_item
        else {
            panic!()
        };

        let proc_params: Vec<Ty> = params.iter().map(|p| self.ast_to_ty(&p.ty)).collect();

        // Arity check
        // Check for variadic
        let impl_ty_opt = impls;

        // Arg count should always be one less because of the implied 'self'
        if proc_params.len() != (args.len() + 1) {
            self.error(
                reciever.span.clone(),
                "argument count mismatch. method calls do no include 'self' as arguments",
            );
        }

        //let arg_tys: Vec<Ty> = args.iter().map(|arg| self.check_expr(arg)).collect();

        //reciever ty needs to be the same as the first arg
        if let Some(impl_ty) = impl_ty_opt {
            if let Some(first) = proc_params.first() {
                let impl_sema_ty = self.ast_to_ty(&impl_ty);
                let Ty::Ptr(inner_first) = first else {
                    self.error(
                        reciever.span.clone(),
                        "the first arg of a mehod call must be a pointer to the type it impls",
                    );
                    return Ty::Error;
                };

                //
                if impl_sema_ty != *first && impl_sema_ty != **inner_first {
                    self.error(reciever.span.clone(), &format!("the first arg of a method call mush match the base type of the reciever ('{:?} != {:?}')", first, impl_sema_ty));
                }
            } else {
                self.error(
                    reciever.span.clone(),
                    &format!(
                        "the first arg of a method call mush match the reciever ('{:?}')",
                        impl_ty
                    ),
                );
            }
        }

        // Per-argument type check
        for (i, arg_expr) in args.iter().enumerate() {
            let arg_ty = self.check_expr(arg_expr);
            // Offset expected ty since arg couint is 1 less
            let expected = proc_params.get(i + 1).unwrap_or(&Ty::Error);

            // hack
            let both_pointers =
                matches!(arg_ty, Ty::Ptr(_) | Ty::NullPtr(_)) && matches!(expected, Ty::Ptr(_));

            if !both_pointers
                && &arg_ty != expected
                && arg_ty != Ty::Error
                && *expected != Ty::Error
            {
                self.error(
                    arg_expr.span.clone(),
                    &format!("expected {:?}, found {:?}", expected, arg_ty),
                );
            }
        }

        let ret_ty = self.ast_to_ty(&ret_type);

        // Return value type
        ret_ty.clone()
    }

    fn check_call(&mut self, call_expr: &Expr, callee_sub_expr: &Expr, args: &[Expr]) -> Ty {
        let callee_ty = self.check_expr(callee_sub_expr);
        // Must be a func-ptr type
        let (param_tys, ret_ty) = match callee_ty {
            Ty::FuncPtr(ref ps, ref r) => (ps, r.as_ref()),
            Ty::Ptr(ref inner) => match &**inner {
                Ty::FuncPtr(p_tys, r_ty) => (p_tys, r_ty.as_ref()),
                _ => {
                    return Ty::Error;
                }
            },
            Ty::Error => return Ty::Error, // propagate
            _ => {
                self.error(callee_sub_expr.span.clone(), "call of non-function value");
                return Ty::Error;
            }
        };

        let param_tys: Vec<Ty> = param_tys.iter().map(|p| self.reresolve(p)).collect();
        let param_count = param_tys.len();

        // Arity check
        // Check for variadic
        let mut variadic = false;
        if let ExprKind::ScopedIdent(i) = &callee_sub_expr.kind {
            // if let Some(intrin) = INTRINSICS.get(&i.to_string()) {
            //     param_count = intrin.required_args as usize;
            // }

            if let Some(def) = self.syms.globals.get(&i.to_string()) {
                let i = def.item_id();
                let item = &self.programs[i.0].items[i.1];
                match item {
                    Item::ProcDef(_proc_d) => {
                        variadic = false; // TODO: support user variadics
                    }
                    Item::ExternProc(proc_exd) => {
                        variadic = proc_exd.is_variadic;
                    }
                    _ => {}
                }
            }
        }
        if param_count != args.len() && !variadic {
            self.error(call_expr.span.clone(), "argument count mismatch");
        }

        // Per-argument type check
        for (i, arg_expr) in args.iter().enumerate() {
            let arg_ty = self.check_expr(arg_expr);
            let expected = param_tys.get(i).unwrap_or(&Ty::Error);

            // If ptr type matches
            let null_match = matches!(&arg_ty, Ty::NullPtr(_) if matches!(expected, Ty::Ptr(_)));

            if !null_match && &arg_ty != expected && arg_ty != Ty::Error && *expected != Ty::Error {
                eprintln!("Callee expr: {:?}", callee_sub_expr);
                self.error(
                    arg_expr.span.clone(),
                    &format!("expected {:?}, found {:?}", expected, arg_ty),
                );
            }
        }

        // Return value type
        ret_ty.clone()
    }

    fn check_bin(&mut self, e: &Expr, op: BinOp, lhs: &Expr, rhs: &Expr) -> Ty {
        use BinOp::*;
        use Ty::*;

        let lt = self.check_expr(lhs);
        let rt = self.check_expr(rhs);

        // Helper predicates
        let numeric = |t: &Ty| matches!(t, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F32 | F64);
        let same_numeric = |l: &Ty, r: &Ty| numeric(l) && l == r;
        let ptr_and_nullptr =
            |l: &Ty, r: &Ty| matches!(l, Ptr(_) | NullPtr(_)) && matches!(r, Ptr(_) | NullPtr(_));

        match op {
            //  arithmetic
            Add | Sub | Mul | Div | Mod => {
                if same_numeric(&lt, &rt) {
                    lt
                } else {
                    self.error(
                        e.span.clone(),
                        &format!(
                            "arithmetic operands must have the same numeric type ({:?} & {:?})",
                            &lt, &rt
                        ),
                    );
                    Ty::Error
                }
            }

            //  comparison (order or equality) -> bool
            Lt | Le | Gt | Ge => {
                if same_numeric(&lt, &rt) {
                    Bool
                } else {
                    self.error(
                        e.span.clone(),
                        "ordered comparison needs matching numeric types",
                    );
                    Ty::Bool // still return Bool so later checks continue
                }
            }

            Eq | Ne => {
                // permit pointer==pointer and bool==bool
                if lt == rt && (numeric(&lt) || matches!(lt, Ptr(_) | Bool))
                    || ptr_and_nullptr(&lt, &rt)
                {
                    Bool
                } else {
                    self.error(
                        e.span.clone(),
                        "equality operands must have the same primitive or pointer type",
                    );
                    Bool
                }
            }

            //  logical && / ||  -> bool
            And | Or => {
                if lt == Bool && rt == Bool {
                    Bool
                } else {
                    self.error(e.span.clone(), "logical operator expects bool operands");
                    Bool
                }
            }

            //  bitwise (| & ^)  -> same integer type
            BitAnd | BitOr | BitXor | BitShiftL | BitShiftR => {
                if lt == rt && Self::is_int(&lt) {
                    lt
                } else {
                    self.error(
                        e.span.clone(),
                        "bitwise operator expects identical integer types",
                    );
                    Ty::Error
                }
            }
        }
    }

    fn is_int(t: &Ty) -> bool {
        matches!(
            t,
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64
        )
    }

    fn check_static(&mut self, idx: ItemId, static_def: &StaticDef) {
        let ty = self.ast_to_ty(&static_def.ty);
        self.tys.item.insert(idx, ty);
        self.scope_stack
            .last_mut()
            .expect("no scope for static def")
            .vars
            .insert(static_def.name.text.to_owned());
    }

    fn check_const(&mut self, idx: ItemId, c: &ConstDef, check_expr: bool) {
        // TODO: This sucks. Calls can be const in theory, but no mechanism for that yet
        match c.value.kind {
            ExprKind::Call(_, _, _) => {
                self.error(
                    c.value.span.clone(),
                    &format!(
                        "value of type {:?} is not allowed in a const initializer",
                        c.value.kind
                    ),
                );
                return;
            }
            _ => {}
        }

        // fake value type on initial pass
        let decl_ty = self.ast_to_ty(&c.ty);
        let val_ty = if check_expr {
            self.check_expr(&c.value)
        } else {
            decl_ty.clone()
        };

        if check_expr && decl_ty != val_ty && val_ty != Ty::Error {
            self.error(
                c.value.span.clone(),
                "const initialiser type does not match its declared type",
            );
        }
        // Record the constant’s (declared) type for later uses.
        self.tys.item.insert(idx, decl_ty);
        self.scope_stack
            .last_mut()
            .expect("no scope for const def")
            .vars
            .insert(c.name.text.to_owned());
    }

    fn check_union_lit_expr(
        &mut self,
        base_expr: &Expr,
        union_def_id: &(usize, usize),
        variant_ident: &Ident,
        init_expr_opt: &Option<Box<Expr>>,
        generic_args: &[Type],
        // This is done a bit hackily, as the variant ident is resolved prior
        // while the generics are not
        elided_generics: &Option<Vec<Ty>>,
    ) -> Ty {
        let Item::UnionDef(u_def) = &self.programs[union_def_id.0].items[union_def_id.1] else {
            panic!("expected union def!") // These should hard fail here since it should be a compiler fix, not user
        };

        let variant_lookup = u_def.variants.iter().find(|v| v.0 == *variant_ident);
        let Some(variant) = variant_lookup else {
            self.error(
                variant_ident.span.clone(),
                &format!(
                    "failed to look up variant {:?} for {:?}",
                    variant_ident.text, u_def.name
                ),
            );
            return Ty::Error;
        };

        if generic_args.len() != u_def.generics.len() && elided_generics.is_none() {
            self.error(
                base_expr.span.clone(),
                &format!(
                    "exptected {} generic args, got {}",
                    u_def.generics.len(),
                    generic_args.len()
                ),
            );
            return Ty::Error;
        }

        let ctx = if let Some(elided) = elided_generics {
            self.create_map_from_generic_tys_2(elided, &u_def.generics)
        } else {
            self.create_map_from_generic_tys(generic_args, &u_def.generics)
        };

        self.generic_context.push(ctx);
        let variant_ty = self.ast_to_ty(&variant.1);
        let expected_args = if variant_ty != Ty::Void { 1 } else { 0 };
        let given_args = if init_expr_opt.is_some() { 1 } else { 0 };
        if given_args != expected_args {
            self.error(
                base_expr.span.clone(),
                &format!("expected {} args in variant literal", expected_args),
            );
            return Ty::Error;
        }

        if let Some(init_expr) = init_expr_opt {
            let ty_arg = self.check_expr(&init_expr);
            if ty_arg != variant_ty {
                self.error(
                    init_expr.span.clone(),
                    &format!("exptected type: {:?}", variant_ty),
                );
                return Ty::Error;
            }
        }

        let union_generics: Vec<Ty> = if let Some(elided) = elided_generics {
            elided.to_vec()
        } else {
            generic_args.iter().map(|a| self.ast_to_ty(a)).collect()
        };

        self.generic_context.pop();
        let _discrim = u_def
            .variants
            .iter()
            .position(|p| p.0 == *variant_ident)
            .expect("somehow failed to lookup index");

        Ty::Union(u_def.name.text.clone(), union_generics)
    }

    // Might have messed up the order here
    fn create_map_from_generic_tys(
        &mut self,
        generic_args: &[Type],
        def_generics: &[Type],
    ) -> HashMap<String, Ty> {
        if def_generics.len() != generic_args.len() {
            panic!("def generic len != generic args len.")
        }

        let mut ctx = HashMap::new();
        for (idx, generic_ty) in def_generics.iter().enumerate() {
            let Type::Generic {
                generic_name,
                index_in_decl: _,
            } = generic_ty
            else {
                panic!("generic type is not generic")
            };

            // If the types aren't equal, add a mapping
            if *generic_ty != generic_args[idx] {
                ctx.insert(
                    generic_name.text.to_owned(),
                    self.ast_to_ty(&generic_args[idx]),
                );
            }
        }
        ctx
    }

    fn create_map_from_generic_tys_2(
        &mut self,
        generic_args: &[Ty],
        def_generics: &[Type],
    ) -> HashMap<String, Ty> {
        if def_generics.len() != generic_args.len() {
            panic!("def generic len != generic args len.")
        }

        let mut ctx = HashMap::new();
        for (idx, generic_ty) in def_generics.iter().enumerate() {
            let Type::Generic {
                generic_name,
                index_in_decl: _,
            } = generic_ty
            else {
                panic!("generic type is not generic")
            };

            let generic_ty = self.ast_to_ty(generic_ty);

            // If the types aren't equal, add a mapping
            if generic_ty != generic_args[idx] {
                ctx.insert(
                    generic_name.text.to_owned(),
                    self.reresolve(&generic_args[idx]),
                );
            }
        }
        ctx
    }

    #[allow(unused)]
    fn create_context_from_ty(&mut self, ty: &Ty, def_generics: &[Type]) -> HashMap<String, Ty> {
        match ty {
            Ty::Struct(_ident, generic_instances) => {
                self.create_map_from_generic_tys_2(generic_instances, def_generics)
            }
            Ty::Union(_ident, generic_instances) => {
                self.create_map_from_generic_tys_2(generic_instances, def_generics)
            }
            _ => HashMap::new(),
        }
    }

    fn with_elision_ctx<F>(&mut self, ty: &Ty, func: F) -> Ty
    where
        F: FnOnce(&mut Self) -> Ty,
    {
        if let Some(path_head) = ty.get_path_head() {
            self.elision_ctx_stack.push(path_head);
            let result = func(self);
            self.elision_ctx_stack.pop();
            result
        } else {
            func(self)
        }
    }
}

// TODO: Check for redefinitions
pub fn collect_globals(program: &Program, source_id: usize) -> Result<SymbolTable, SemaError> {
    let mut syms = SymbolTable {
        globals: Default::default(),
    };

    for (idx, it) in program.items.iter().enumerate() {
        match it {
            Item::ProcDef(def) => {
                let name = &def.name.text;

                // if INTRINSICS.contains_key(name) {
                //     return Err(SemaError {
                //         source_id,
                //         span: it.get_ident().span,
                //         message: format!("redefinition of intrinsic '{}'", &it.get_ident().text),
                //     });
                // }

                if let Some(_prev) = syms
                    .globals
                    .insert(name.to_string(), Def::Proc((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of procedure '{}'", &it.get_ident().text),
                    });
                }
            }
            Item::ExternProc(def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(def.name.text.clone(), Def::Proc((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!(
                            "redefinition of external procedure '{}'",
                            &it.get_ident().text
                        ),
                    });
                }
            }
            Item::StructDef(def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(def.name.text.clone(), Def::Struct((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of struct '{}'", &it.get_ident().text),
                    });
                }
            }
            Item::UnionDef(def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(def.name.text.clone(), Def::Union((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of union '{}'", &it.get_ident().text),
                    });
                }
            }
            Item::EnumDef(def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(def.name.text.clone(), Def::Enum((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of enum '{}'", &it.get_ident().text),
                    });
                }
            }
            Item::ConstDef(def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(def.name.text.clone(), Def::Const((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of constant '{}'", &it.get_ident().text),
                    });
                }
            }
            Item::IncludeStmt(_include_stmt) => {
                // Currently don't care. MIGHT later if I use scope resolution for module item access
            }
            Item::StaticDef(static_def) => {
                if let Some(_prev) = syms
                    .globals
                    .insert(static_def.name.text.clone(), Def::Static((source_id, idx)))
                {
                    return Err(SemaError {
                        source_id,
                        span: it.get_ident().span,
                        message: format!("redefinition of static '{}'", &it.get_ident().text),
                    });
                }
            }
        }
    }

    Ok(syms)
}

fn stmt_always_returns(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(_) => true,
        // Stmt::Block(stmts) => block_always_returns(stmts),
        // Stmt::If(_expr, then_block, else_block) => {
        //     match else_block {
        //         Some(else_block) => {
        //             block_always_returns(then_block) && block_always_returns(else_block)
        //         }
        //         None => false, // could fall through
        //     }
        // }
        Stmt::While { .. } => false, // conservatively: might not enter
        Stmt::Break | Stmt::Continue => false, // terminates loop branch, not function
        _ => false,
    }
}

fn expr_always_returns(expr: &Expr) -> bool {
    if let ExprKind::Block(stmts, _tail) = &expr.kind {
        block_always_returns(&stmts)
    } else {
        false
    }
}

fn block_always_returns(block: &Block) -> bool {
    for stmt in block {
        if stmt_always_returns(stmt) {
            return true; // this path returns - rest is unreachable TODO: could use for dead code analysis
        }
    }
    false
}
