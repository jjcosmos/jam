use std::hash::Hash;

use logos::Span;

/// program AST node
#[derive(Clone, Debug)]
pub struct Program {
    pub(crate) items: Vec<Item>,
}

#[derive(Clone, Debug, Eq)]
pub struct Ident {
    pub text: String, // utf‑8 from src file.
    pub span: Span,   // todo: use for diagnostics.
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text //&& self.span == other.span
    }
}

impl Hash for Ident {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        //self.span.hash(state); // don't care about span
    }
}

impl Ident {
    pub fn dummy(span: Span) -> Self {
        Self {
            text: String::new(),
            span,
        }
    }
}

/// Top-level item
#[derive(Clone, Debug)]
pub enum Item {
    ProcDef(ProcDef),            // a function definition
    ExternProc(ExternalProcDef), // external function declaration
    StructDef(StructDef),        // struct type definition
    UnionDef(UnionDef),          // union type definition
    EnumDef(EnumDef),            // enum type definition
    ConstDef(ConstDef),          // global constant
    StaticDef(StaticDef),        // global static
    IncludeStmt(IncludeStmt),
}

impl Item {
    pub fn get_ident(&self) -> Ident {
        match self {
            Item::ProcDef(proc_def) => proc_def.name.clone(),
            Item::ExternProc(external_proc_def) => external_proc_def.name.clone(),
            Item::StructDef(struct_def) => struct_def.name.clone(),
            Item::EnumDef(enum_def) => enum_def.name.clone(),
            Item::ConstDef(const_def) => const_def.name.clone(),
            Item::StaticDef(static_def) => static_def.name.clone(),
            Item::IncludeStmt(inc_stmt) => inc_stmt.mod_name.clone(),
            Item::UnionDef(union_def) => union_def.name.clone(),
        }
    }
}

/// Function definition AST
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ProcDef {
    pub name: Ident,
    pub params: Vec<Param>, // list of parameters (name and type)
    pub ret_type: Type,     // return type
    pub body: Expr,         // function body (a block of statements)
    pub impls: Option<Type>,
    pub generics: Vec<Type>, // Should be all Type::Generic
}

/// External procedure signature / prototype
#[derive(Clone, Debug)]
pub struct ExternalProcDef {
    pub(crate) name: Ident,
    pub(crate) params: Vec<Param>,
    pub(crate) ret_type: Type,
    pub(crate) is_variadic: bool,
}

/// Struct definition
#[derive(Clone, Debug)]
pub struct StructDef {
    pub(crate) name: Ident,
    pub(crate) fields: Vec<(Ident, Type)>, // field name and type
    #[allow(unused)]
    pub(crate) external: bool, // I don't think I'll need this
    pub(crate) generics: Vec<Type>,        // Should be all Type::Generic
}

#[derive(Clone, Debug)]
pub struct UnionDef {
    pub(crate) name: Ident,
    pub(crate) variants: Vec<(Ident, Type)>,
    pub(crate) generics: Vec<Type>,
}

impl UnionDef {
    pub fn index_of(&self, ident: &Ident) -> usize {
        self.variants.iter().position(|v| v.0 == *ident).unwrap()
    }
}

/// Enum definition - using ints as resolved type for now
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub(crate) name: Ident,
    pub(crate) variants: Vec<(Ident, i64)>,
}

/// Constant definition
#[derive(Clone, Debug)]
pub struct ConstDef {
    pub(crate) name: Ident,
    pub(crate) ty: Type,
    pub(crate) value: Expr,
}

/// Static definition
#[derive(Clone, Debug)]
pub struct StaticDef {
    pub(crate) name: Ident,
    pub(crate) ty: Type,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IncludeStmt {
    pub(crate) mod_name: Ident,
}

/// A function parameter
#[derive(Clone, Debug)]
pub struct Param {
    pub(crate) mut_arg: bool, //i.e foo(const f: Thing*). Used to cut allocas
    pub(crate) name: Ident,
    pub(crate) ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrimTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    USize,
    F32,
    F64,
    Bool,
    CStr,
}

/// AST node for types
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Primitive(PrimTy),
    Ptr(Box<Type>), // pointer: *T
    // TODO: currently not used. All function pointers are void*
    FuncPtr(Vec<Type>, Box<Type>), // function pointer type (param types, return type)
    Array(Box<Type>, usize),
    Void,
    Named(Ident, Vec<Type>), // Track potential generics for named. like "let foo: Foo<i32>;"
    Inferred(Span),
    Generic {
        //source_ident: Ident,  // The struct or proc it is defined in
        generic_name: Ident,  // The generic name, like 'T'
        index_in_decl: usize, // The order it appears in in the declaration
    },
}

impl Type {
    pub(crate) fn as_ns_string(&self) -> String {
        match self {
            Type::Named(ident, _items) => ident.text.clone(),
            _ => panic!("Cannot get a namespace string from anthing other than a struct!"),
        }
    }
}

/// Funtion bodies and whatnot
pub type Block = Vec<Stmt>;

/// AST statement
#[derive(Clone, Debug)]
pub enum Stmt {
    ExprStmt(Expr), // expression used as a statement
    // Not expressions yet
    Let(Ident, Type, Option<Expr>), // variable declaration: let name: type = expr
    // Also not expressions
    Assign(Expr, Expr), // assignment (target must be an lvalue)
    // Could be expression always evaluating to void
    While(Expr, Expr), // while loop COULD deprecate this in favor of only for i.e. for(; i < 4 ;) {}
    // The inner types here should continue to be expression statements
    // Could also always evaluate to void
    For(Box<Option<Stmt>>, Option<Expr>, Box<Option<Stmt>>, Expr), // init statement, binary expression, post loop statement, loop body block
    // I don't actually know
    Return(Option<Expr>),
    //Block(Block),
    Break,
    Continue,
}

impl Stmt {
    #[allow(unused)]
    pub(crate) fn try_get_span(&self) -> Option<Span> {
        match self {
            Stmt::ExprStmt(expr) => Some(expr.span.clone()),
            Stmt::Let(ident, _, _expr) => Some(ident.span.clone()),
            Stmt::Assign(expr, _expr1) => Some(expr.span.clone()),
            //Stmt::If(expr, _stmts, _stmts1) => Some(expr.span.clone()),
            Stmt::While(expr, _stmts) => Some(expr.span.clone()),
            Stmt::For(_stmt, _expr, _post_stmt, _stmts) => None,
            Stmt::Return(expr) => expr.clone().map(|e| e.span),
            //Stmt::Block(_stmts) => None,
            Stmt::Break => None,    // could store a span in here
            Stmt::Continue => None, // same here
        }
    }
}

// Track expression span and source file for better output
#[derive(Clone, Debug)]
pub struct Expr {
    pub id: (usize, usize),
    pub span: Span,
    pub kind: ExprKind,
}

/// AST expressions
#[derive(Clone, Debug)]
pub enum ExprKind {
    Literal(Literal),         // literal
    ScopedIdent(ScopedIdent), // variable reference
    FunctionReference(FunctionReference),
    Binary(Box<Expr>, BinOp, Box<Expr>), // binary operation (like +, -, ==, etc.)
    Unary(UnOp, Box<Expr>),              // unary op (* deref, & address-of, - negation, ! not)
    /// function call: the function to call (could be name or func pointer expr) and arguments. Vec<Type> for generic args
    Call(Box<Expr>, Vec<Expr>, Vec<Type>),
    MethodCall {
        receiver: Box<Expr>,
        method_name: Ident,
        args: Vec<Expr>,
        generic_args: Vec<Type>,
    },
    Cast(Box<Expr>, Type), // type cast: (expr as Type)
    StructLit {
        name: Ident,
        fields: Vec<(Ident, Expr)>,
        generic_inst_tys: Vec<Type>,
    },
    ArrayLit(Vec<Expr>),
    /// Path including base, variant ident, init expr, generic inst tys
    UnionLit(ScopedIdent, Ident, Option<Box<Expr>>, Vec<Type>),
    Index(Box<Expr>, Box<Expr>),
    Field(Box<Expr>, Ident, bool), // bool uses deref (TODO: Not used rn)
    Match(MatchExpression),
    Block(Vec<Stmt>, Option<Box<Expr>>),
    // Should be expression
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>), // if/else (binexpr) (block) (else block)
    // TODO: This should not be a valid expression type for NoRT native targets
    InlineWat(Vec<String>, Option<Type>, String), // Captured locals, expression type, wasm string
}

#[derive(Debug, Clone)]
pub(crate) struct MatchExpression {
    // Needs to evaluate to a union type
    pub as_ref: bool,
    pub union_to_decomp: Box<Expr>,
    pub arms: Vec<MatchArm>,
    pub default_arm: Option<Box<MatchArm>>,
}

#[derive(Debug, Clone)]
pub(crate) struct MatchArm {
    pub variant_ident: Ident,
    pub binding: Option<Ident>,
    pub body_expression: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FunctionReference {
    pub(crate) path: ScopedIdent,
    pub(crate) generic_args: Vec<Type>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ScopedIdent {
    pub(crate) scopes: Vec<String>,
    pub(crate) ident: Ident, // Ident's eq doesn't care about span
}

impl ScopedIdent {
    #[allow(unused)]
    pub(crate) fn get_path_string(&self) -> Option<String> {
        if self.scopes.is_empty() {
            return None;
        }

        return Some(self.scopes.join("::"));
    }
    // TODO: Maybe cache result
    pub(crate) fn to_string(&self) -> String {
        let mut out = String::new();
        for scope in &self.scopes {
            out.push_str(scope);
            out.push_str("::");
        }
        out.push_str(&self.ident.text);

        out
    }
}

#[derive(Clone, Debug)]
pub enum Literal {
    ///value, width, signed
    Int(i128, u8, bool),
    USize(u64),
    ///value, is 32 bit
    Float(f64, bool),
    Bool(bool),
    /// null-terminated string (in code‑gen)
    Str(String),
    Null(Type),

    // Comptime
    SizeOf(Type),
}

/// Binary operators
#[derive(Copy, Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    BitAnd,
    BitOr,
    BitXor,
    BitShiftR,
    BitShiftL,
    And,
    Or,
}

/// Unary operators
#[derive(Copy, Clone, Debug)]
pub enum UnOp {
    Neg,
    Not,
    Deref,
    MaybeDeref,
    AddrOf,
}
