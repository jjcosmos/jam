use std::fs::read_to_string;
use std::path::Path;

use hashbrown::HashMap;

use logos::Span;

use crate::frontend::ast::{
    BinOp, ConstDef, EnumDef, Expr, ExprKind, ExternalProcDef, FunctionReference, Ident,
    IncludeStmt, Item, Literal, MatchArm, MatchExpression, Param, PrimTy, ProcDef, Program,
    ScopedIdent, StaticDef, Stmt, StructDef, Type, UnOp, UnionDef,
};
use crate::frontend::tokenizer::{Token, lex_source_span};

type PResult<T> = Result<T, ParseError>;

pub struct Parser<'src> {
    source_id: usize,
    tokens: Vec<(Token<'src>, Span)>,
    index: usize,
    //source: &'src str,
    expression_counter: usize,

    impl_context: Option<ImplContext>,
    generic_context: HashMap<String, Type>,
}

#[derive(Clone, Debug)]
struct ImplContext {
    ty: Type,
    generics: Vec<Ident>,
}

#[derive(Debug)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl<'src> Parser<'src> {
    fn peek(&self) -> Token {
        self.tokens[self.index].0
    }
    fn peek_plus_n(&self, n: usize) -> Option<Token> {
        let idex = self.index + n;
        if idex >= self.tokens.len() {
            return None;
        }

        return Some(self.tokens[self.index + n].0);
    }

    fn span(&self) -> Span {
        self.tokens[self.index].1.clone()
    }
    fn _prev(&self) -> &'src (Token, Span) {
        let index = (self.index - 1).clamp(0, usize::MAX);
        &self.tokens[index]
    }
    fn bump(&mut self) -> Token {
        let t = self.tokens[self.index].0;
        self.index += 1;
        t
    }
    fn eat(&mut self, expected: Token) -> Result<Span, ParseError> {
        if self.peek() == expected {
            let sp = self.span();
            self.bump();
            Ok(sp)
        } else {
            Err(ParseError {
                span: self.span(),
                message: format!("expected {:?} instead of {:?}", expected, self.peek()),
            })
        }
    }

    // TODO: spans are not always correct here
    fn new_expr(&mut self, span: Span, expr_kind: ExprKind) -> Expr {
        self.expression_counter += 1;
        Expr {
            span,
            id: (self.source_id, self.expression_counter),
            kind: expr_kind,
        }
    }

    fn new(src: &'src str, source_id: usize) -> Self {
        Self {
            source_id,
            tokens: lex_source_span(src, source_id),
            index: 0,
            //source: src,
            expression_counter: 0,
            generic_context: HashMap::new(),
            impl_context: None,
        }
    }
}

impl<'a> Parser<'a> {
    pub fn parse_program(src: &str, source_id: usize) -> PResult<Program> {
        let mut parser = Parser::new(src, source_id);
        let mut items = Vec::new();
        while parser.peek() != Token::Eof && parser.peek() != Token::End {
            if let Some(result) = parser.parse_item() {
                items.push(result?);
            }
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Option<PResult<Item>> {
        match self.peek() {
            Token::Impl => self.set_impl_context().err().map(Err),
            Token::RBrace => self.clear_impl_context().err().map(Err),
            Token::Include => Some(self.parse_include_stmt()),
            Token::Proc => Some(self.parse_proc_def()),
            Token::Extern => Some(self.parse_extern()),
            Token::Struct => Some(self.parse_struct(false)),
            Token::Sum => Some(self.parse_union()),
            Token::Enum => Some(self.parse_enum()),
            Token::Const => Some(self.parse_const()),
            Token::Static => Some(self.parse_static()),
            _ => {
                // I think this gives a better error?
                self.bump();
                Some(Err(ParseError {
                    span: self.span(),
                    message: format!("Unexpected token type {:?}", self.peek()),
                }))
            }
        }
    }

    fn set_impl_context(&mut self) -> PResult<()> {
        self.eat(Token::Impl)?;

        let mut generics = vec![];
        let mut generic_ctx = HashMap::new();
        let mut index_in_decl = 0;
        if self.peek() == Token::TyListOpen {
            self.eat(Token::TyListOpen)?;
            loop {
                let ident = self.parse_ident()?;
                generic_ctx.insert(
                    ident.text.clone(),
                    Type::Generic {
                        generic_name: ident.clone(),
                        index_in_decl: index_in_decl,
                    },
                );
                generics.push(ident);
                index_in_decl += 1;

                if self.peek() == Token::Comma {
                    self.eat(Token::Comma)?;
                } else {
                    self.eat(Token::TyListClose)?;
                    break;
                }
            }
        }

        self.generic_context = generic_ctx;
        let ty = self.parse_type()?;

        // TODO: Support newtype pattern
        if !matches!(ty, Type::Named(_, _)) {
            return Err(ParseError {
                span: self.span(),
                message: "You can only impl struct types".to_string(),
            });
        };

        self.generic_context.clear();

        self.impl_context = Some(ImplContext { ty, generics });

        self.eat(Token::LBrace)?;

        Ok(())
    }

    fn clear_impl_context(&mut self) -> PResult<()> {
        self.eat(Token::RBrace)?;
        self.impl_context = None;

        Ok(())
    }

    fn parse_include_stmt(&mut self) -> PResult<Item> {
        self.eat(Token::Include)?;
        let ident = if let Token::StrLiteral(literal) = self.peek() {
            literal.to_string()
        } else {
            return Err(ParseError {
                span: self.span(),
                message: "includes require a string literal".to_owned(),
            });
        };
        self.bump();
        self.eat(Token::Semicolon)?;
        Ok(Item::IncludeStmt(IncludeStmt {
            mod_name: Ident {
                text: ident.to_owned(),
                span: self.span(),
            },
        }))
    }

    fn parse_proc_def(&mut self) -> PResult<Item> {
        self.eat(Token::Proc)?;

        let mut impls = None;
        let name = self.parse_ident()?;

        // Parse generics
        let mut generic_map = HashMap::new();
        if let Some(impl_ctx) = &self.impl_context {
            // This might not be correct, since the generic in here won't have been resolved
            // In the case of impl <T> Wrap<T> {} the there was no generic ctx when parsing Wrap<T>
            impls = Some(impl_ctx.ty.clone());
            for (idx, impl_generic) in impl_ctx.generics.iter().enumerate() {
                generic_map.insert(
                    impl_generic.text.clone(),
                    Type::Generic {
                        generic_name: impl_generic.clone(),
                        index_in_decl: idx,
                    },
                );
            }
        }
        if self.peek() == Token::TyListOpen {
            self.eat(Token::TyListOpen)?;
            // This could be pre-filled by the impl
            let mut idx = generic_map.len();
            while self.peek() != Token::TyListClose {
                let ident = self.parse_ident()?;
                generic_map.insert(
                    ident.text.to_owned(),
                    Type::Generic {
                        //source_ident: name.clone(),
                        generic_name: ident,
                        index_in_decl: idx,
                    },
                );
                idx += 1;
                if self.peek() == Token::TyListClose {
                    break;
                } else {
                    self.eat(Token::Comma)?;
                }
            }
            self.eat(Token::TyListClose)?;
            //println!("Parsed generics: {:?}", generic_map);
        }

        // Might need to set a generic context here so that generic args and ret tys are mapped
        self.generic_context = generic_map.clone();

        self.eat(Token::LParen)?;
        let params = self.parse_named_param_list()?;
        self.eat(Token::RParen)?;
        let ret_type = if matches!(self.peek(), Token::Arrow) {
            self.eat(Token::Arrow)?;
            self.parse_type()?
        } else {
            Type::Void
        };
        let body = self.parse_block_expression()?; // `{ … }`

        self.generic_context.clear();

        // TODO: This should consider full namespaces when those are a thing
        let name = if let Some(impl_name) = &impls {
            Ident {
                text: format!("{}::{}", impl_name.as_ns_string(), name.text),
                span: name.span,
            }
        } else {
            name
        };

        // TODO: If impls, should probably scope to the ident i.e. name should be scope::ident
        // Not sure if doing this during parsing is correct though
        Ok(Item::ProcDef(ProcDef {
            name,
            params,
            ret_type,
            body,
            impls,
            generics: generic_map.into_values().map(|c| c).collect(),
        }))
    }

    fn parse_extern(&mut self) -> PResult<Item> {
        self.eat(Token::Extern)?;

        if self.peek() == Token::Proc {
            return self.parse_extern_proc();
        } else {
            return self.parse_struct(true);
        }
    }

    fn parse_extern_proc(&mut self) -> PResult<Item> {
        //self.eat(Token::Extern)?;
        self.eat(Token::Proc)?;
        let name = self.parse_ident()?;
        self.eat(Token::LParen)?;
        // TODO: Param list expects type names
        let params = self.parse_extern_param_list()?;

        let is_variadic = self.peek() == Token::Elipses;
        if is_variadic {
            self.eat(Token::Elipses)?;
        }

        self.eat(Token::RParen)?;
        let ret_type = if matches!(self.peek(), Token::Arrow) {
            self.eat(Token::Arrow)?;
            self.parse_type()?
        } else {
            Type::Void
        };
        self.eat(Token::Semicolon)?;

        // TODO: Lookup ABI params if defined as external

        Ok(Item::ExternProc(ExternalProcDef {
            name,
            params,
            ret_type,
            is_variadic,
        }))
    }

    fn parse_struct(&mut self, external: bool) -> PResult<Item> {
        // Parse Ident
        self.eat(Token::Struct)?;

        let name = self.parse_ident()?;

        let mut generic_map = HashMap::new();
        if self.peek() == Token::TyListOpen {
            self.eat(Token::TyListOpen)?;
            // parse generics
            let mut idx = 0;
            loop {
                let g_ty = self.parse_ident()?;
                let generic = Type::Generic {
                    //source_ident: name.clone(),
                    generic_name: g_ty.clone(),
                    index_in_decl: idx,
                };
                generic_map.insert(g_ty.text, generic);
                idx += 1;

                if self.peek() == Token::Comma {
                    self.eat(Token::Comma)?;
                } else {
                    self.eat(Token::TyListClose)?;
                    break;
                }
            }
        }

        self.eat(Token::LBrace)?;

        self.generic_context = generic_map.clone();

        let mut fields = vec![];
        while self.peek() != Token::RBrace {
            let field_name = self.parse_ident()?;
            self.eat(Token::Colon)?;
            let ty = self.parse_type()?;
            //self.eat(Token::Semicolon)?;
            self.eat(Token::Comma)?;
            fields.push((field_name, ty));
        }

        self.eat(Token::RBrace)?;

        self.generic_context.clear();

        // Foreach field, parse Ident & Type
        Ok(Item::StructDef(StructDef {
            name,
            fields,
            external,
            generics: generic_map.into_values().map(|c| c).collect(),
        }))
    }

    fn parse_union(&mut self) -> PResult<Item> {
        self.eat(Token::Sum)?;
        let name = self.parse_ident()?;

        let mut generic_map = HashMap::new();
        // This is a unique parse since it is expecting all idents
        if self.peek() == Token::TyListOpen {
            self.eat(Token::TyListOpen)?;
            // parse generics
            let mut idx = 0;
            loop {
                let g_ty = self.parse_ident()?;
                let generic = Type::Generic {
                    generic_name: g_ty.clone(),
                    index_in_decl: idx,
                };
                generic_map.insert(g_ty.text, generic);
                idx += 1;

                if self.peek() == Token::Comma {
                    self.eat(Token::Comma)?;
                } else {
                    self.eat(Token::TyListClose)?;
                    break;
                }
            }
        }

        self.eat(Token::LBrace)?;

        self.generic_context = generic_map.clone();

        let mut variants = vec![];
        while self.peek() != Token::RBrace {
            let field_name = self.parse_ident()?;

            if self.peek() == Token::Comma {
                variants.push((field_name, Type::Void));
                self.eat(Token::Comma)?;
                continue;
            }

            self.eat(Token::LParen)?;
            let ty = self.parse_type()?;
            self.eat(Token::RParen)?;
            self.eat(Token::Comma)?;
            variants.push((field_name, ty));
        }

        self.eat(Token::RBrace)?;

        self.generic_context.clear();

        let union_def = Item::UnionDef(UnionDef {
            name,
            variants,
            generics: generic_map.into_values().map(|c| c).collect(),
        });

        //println!("Parsed union: {:#?}", union_def);

        Ok(union_def)
    }

    fn parse_enum(&mut self) -> PResult<Item> {
        self.eat(Token::Enum)?;
        let name = self.parse_ident()?;
        self.eat(Token::LBrace)?;

        let mut variants = vec![];
        let mut current_val = 0;

        while self.peek() != Token::RBrace {
            let var_name = self.parse_ident()?;

            let value;
            if self.peek() == Token::Colon {
                self.bump(); // consume :
                let tok = self.bump(); // get literal token
                match tok {
                    Token::IntLit(n) => {
                        value = n.0 as i64;
                        current_val = n.0 + 1;
                    }
                    _ => {
                        return Err(ParseError {
                            span: self.span(),
                            message: "Expected integer literal after assignment".to_string(),
                        });
                    }
                }
            } else {
                value = current_val as i64;
                current_val += 1;
            }

            variants.push((var_name, value));

            if self.peek() == Token::Comma {
                self.bump();

                if self.peek() == Token::RBrace {
                    break;
                }
            }
        }

        self.eat(Token::RBrace)?;

        // allow a semicolon at the end for c-ish parity
        if self.peek() == Token::Semicolon {
            self.bump();
        }

        Ok(Item::EnumDef(EnumDef { name, variants }))
    }

    fn parse_const(&mut self) -> PResult<Item> {
        self.eat(Token::Const)?;
        let name = self.parse_ident()?;
        self.eat(Token::Colon)?;
        let ty = self.parse_type()?;
        self.eat(Token::Assign)?;
        let value = self.parse_expression()?;
        self.eat(Token::Semicolon)?;
        Ok(Item::ConstDef(ConstDef {
            name,
            ty,
            value: value,
        }))
    }

    fn parse_static(&mut self) -> PResult<Item> {
        self.eat(Token::Static)?;
        let name = self.parse_ident()?;
        self.eat(Token::Colon)?;
        let ty = self.parse_type()?;
        self.eat(Token::Semicolon)?;

        Ok(Item::StaticDef(StaticDef { name, ty }))
    }

    fn parse_named_param_list(&mut self) -> PResult<Vec<Param>> {
        let mut params = vec![];

        if self.peek() == Token::RParen {
            return Ok(params);
        }

        loop {
            let mut_arg = if let Token::Mut = self.peek() {
                self.eat(Token::Mut)?;
                true
            } else {
                false
            };

            let name = self.parse_ident()?;
            self.eat(Token::Colon)?;
            let ty = self.parse_type()?;
            params.push(Param { mut_arg, name, ty });

            match self.peek() {
                Token::Comma => {
                    self.bump();
                    if self.peek() == Token::RParen {
                        break;
                    }
                }
                Token::RParen => break,
                _ => {
                    return Err(ParseError {
                        span: self.span(),
                        message: "expected ',' or ')'".to_string(),
                    });
                }
            }
        }

        Ok(params)
    }

    fn parse_extern_param_list(&mut self) -> PResult<Vec<Param>> {
        let mut params = vec![];

        if self.peek() == Token::RParen {
            return Ok(params);
        }

        loop {
            if self.peek_plus_n(1) == Some(Token::Colon) {
                let name = self.parse_ident()?;
                self.eat(Token::Colon)?;
                let ty = self.parse_type()?;

                // We don't care about const args in externals, since they have no body
                params.push(Param {
                    mut_arg: true,
                    name,
                    ty,
                });
            } else {
                let ty = self.parse_type()?;
                params.push(Param {
                    mut_arg: true,
                    name: Ident::dummy(self.span()),
                    ty,
                });
            }

            match self.peek() {
                Token::Comma => {
                    self.bump();
                    if self.peek() == Token::RParen || self.peek() == Token::Elipses {
                        break;
                    }
                }
                Token::RParen => break,
                Token::Elipses => break,
                _ => {
                    return Err(ParseError {
                        span: self.span(),
                        message: "expected ',' or ')'".to_string(),
                    });
                }
            }
        }

        Ok(params)
    }

    fn parse_block_expression(&mut self) -> PResult<Expr> {
        self.eat(Token::LBrace)?;
        let mut stmts = vec![];

        let mut tail_expression = None;
        while self.peek() != Token::RBrace {
            if self.lookahead_no_statements_left() {
                tail_expression = Some(Box::new(self.parse_expression()?));
                break;
            } else {
                stmts.push(self.parse_statement()?);
            }
        }

        self.eat(Token::RBrace)?;
        let block_expr =
            self.new_expr(self.span().clone(), ExprKind::Block(stmts, tail_expression));
        Ok(block_expr)
    }

    fn lookahead_no_statements_left(&self) -> bool {
        let mut n = 0;
        let mut depth = 0;

        while let Some(tok) = self.peek_plus_n(n) {
            match tok {
                Token::LBrace => depth += 1,
                Token::RBrace => {
                    if depth == 0 {
                        return true; // we're about to hit the end of the current block
                    } else {
                        depth -= 1;
                    }
                }
                Token::Semicolon => {
                    if depth == 0 {
                        return false; // there's at least one statement left
                    }
                }
                _ => {}
            }
            n += 1;
        }

        false
    }

    fn parse_statement(&mut self) -> PResult<Stmt> {
        match self.peek() {
            Token::Let => self.parse_let_stmt(),
            Token::While => self.parse_while_stmt(),
            Token::For => self.parse_for_stmt(),
            Token::Return => self.parse_return_stmt(),
            // Token::LBrace => {
            //     let inner = self.parse_block()?;
            //     Ok(Stmt::Block(inner))
            // }
            // Ident could just be a function call?
            Token::Ident(_) => self.parse_ident_stmt(),
            Token::Continue => {
                self.eat(Token::Continue)?;
                self.eat(Token::Semicolon)?;
                Ok(Stmt::Continue)
            }
            Token::Break => {
                self.eat(Token::Break)?;
                self.eat(Token::Semicolon)?;
                Ok(Stmt::Break)
            }
            _ => {
                let lhs = self.parse_expression()?;
                if Self::is_assignment_adjacent(&self.peek()) {
                    self.parse_varied_assign(lhs)
                } else {
                    self.eat(Token::Semicolon)?;
                    Ok(Stmt::ExprStmt(lhs))
                }
            }
        }
    }

    fn is_assignment_adjacent(tok: &Token) -> bool {
        match tok {
            Token::Assign
            | Token::PlusEq
            | Token::MinusEq
            | Token::StarEq
            | Token::SlashEq
            | Token::PercentEq
            | Token::AmpEq
            | Token::PipeEq
            | Token::CaretEq
            | Token::Arrow2LEq
            | Token::Arrow2REq => true,
            _ => false,
        }
    }

    fn parse_ident_stmt(&mut self) -> PResult<Stmt> {
        let lhs = self.parse_expression()?;
        //eprintln!("Expression statement is {:?}", lhs);
        if matches!(
            lhs.kind,
            ExprKind::Call(_, _, _,)
                | ExprKind::MethodCall {
                    receiver: _,
                    method_name: _,
                    args: _,
                    generic_args: _,
                }
        ) {
            // This is a function call
            self.eat(Token::Semicolon)?;
            return Ok(Stmt::ExprStmt(lhs));
        }

        self.parse_varied_assign(lhs)
    }

    fn parse_varied_assign(&mut self, lhs: Expr) -> PResult<Stmt> {
        let op_span = self.span();

        match self.peek() {
            Token::Assign => {
                self.bump();
                let expr = self.parse_expression()?;
                self.eat(Token::Semicolon)?;
                Ok(Stmt::Assign(lhs, expr))
            }

            // Compound assignments
            Token::PlusEq
            | Token::MinusEq
            | Token::StarEq
            | Token::SlashEq
            | Token::PercentEq
            | Token::AmpEq
            | Token::PipeEq
            | Token::CaretEq
            | Token::Arrow2LEq
            | Token::Arrow2REq => {
                let binop = match self.peek() {
                    Token::PlusEq => BinOp::Add,
                    Token::MinusEq => BinOp::Sub,
                    Token::StarEq => BinOp::Mul,
                    Token::SlashEq => BinOp::Div,
                    Token::PercentEq => BinOp::Mod,
                    Token::AmpEq => BinOp::BitAnd,
                    Token::PipeEq => BinOp::BitOr,
                    Token::CaretEq => BinOp::BitXor,
                    Token::Arrow2LEq => BinOp::BitShiftL,
                    Token::Arrow2REq => BinOp::BitShiftR,
                    _ => unreachable!(),
                };

                self.bump(); // consume the compound operator

                let rhs = self.parse_expression()?;
                self.eat(Token::Semicolon)?;

                let bin_expr = self.new_expr(
                    op_span.clone(),
                    ExprKind::Binary(Box::new(lhs.clone()), binop, Box::new(rhs)),
                );
                Ok(Stmt::Assign(lhs, bin_expr))
            }

            _ => Err(ParseError {
                span: self.span(),
                message: "expected assignment operator".to_string(),
            }),
        }
    }

    fn _parse_assigment_stmt(&mut self) -> PResult<Stmt> {
        eprintln!("Assignment stmt");
        // Make expression for struct field / index access - l values
        let lhs = self.parse_expression()?;
        self.eat(Token::Assign)?;
        let expr = self.parse_expression()?;
        self.eat(Token::Semicolon)?;
        Ok(Stmt::Assign(lhs, expr))
    }

    fn parse_let_stmt(&mut self) -> PResult<Stmt> {
        // i.e. let x: i32 = get_int();

        self.eat(Token::Let)?;
        let ident = self.parse_ident()?;
        // TODO: make this optional for type inferance w/let
        let ty = if self.peek() == Token::Colon {
            self.eat(Token::Colon)?;
            self.parse_type()?
        } else {
            Type::Inferred(self.span())
        };
        //

        let expr = if self.peek() == Token::Assign {
            self.eat(Token::Assign)?;
            let expr = self.parse_expression()?;
            self.eat(Token::Semicolon)?;
            Some(expr)
        } else {
            self.eat(Token::Semicolon)?;
            None
        };

        // Ident, ty, expr
        Ok(Stmt::Let(ident, ty, expr))
    }

    fn parse_for_stmt(&mut self) -> PResult<Stmt> {
        self.eat(Token::For)?;
        self.eat(Token::LParen)?;

        // No init statement
        let init = if self.peek() == Token::Semicolon {
            self.eat(Token::Semicolon)?;
            None
        } else {
            Some(self.parse_statement()?)
        };

        let eval = if self.peek() == Token::Semicolon {
            self.eat(Token::Semicolon)?;
            None
        } else {
            let eval_expr = self.parse_expression()?;
            self.eat(Token::Semicolon)?;
            Some(eval_expr)
        };

        let post_loop = if self.peek() == Token::Semicolon {
            self.eat(Token::Semicolon)?;
            None
        } else {
            // This should probably be more restrictive
            let stmt = self.parse_statement()?;
            Some(stmt)
        };

        self.eat(Token::RParen)?;

        let body = self.parse_block_expression()?;

        self.eat(Token::Semicolon)?;

        Ok(Stmt::For(Box::new(init), eval, Box::new(post_loop), body))
    }

    fn parse_while_stmt(&mut self) -> PResult<Stmt> {
        // i.e.
        // while (true) {
        // inf_func();
        //};
        self.eat(Token::While)?;
        self.eat(Token::LParen)?;
        let expr = self.parse_expression()?;
        self.eat(Token::RParen)?;

        let body = self.parse_block_expression()?;

        self.eat(Token::Semicolon)?;
        // Expr, block (not expr::block)
        Ok(Stmt::While(expr, body))
    }

    fn parse_if_expression(&mut self, lhs_span: &Span) -> PResult<Expr> {
        // if (true) {
        // do_somthing();
        //}
        //else {
        // panic();
        //}

        self.eat(Token::If)?;

        self.eat(Token::LParen)?;
        let condition = self.parse_expression()?;
        self.eat(Token::RParen)?;
        let primary_block = self.parse_block_expression()?;

        let else_block = if self.peek() == Token::Else {
            self.bump();

            if self.peek() == Token::If {
                Some(Box::new(self.parse_if_expression(lhs_span)?))
            } else {
                Some(Box::new(self.parse_block_expression()?))
            }
        } else {
            None
        };

        // expr, body, optional else
        Ok(self.new_expr(
            lhs_span.clone(),
            ExprKind::If(Box::new(condition), Box::new(primary_block), else_block),
        ))
    }

    fn parse_return_stmt(&mut self) -> PResult<Stmt> {
        self.eat(Token::Return)?;

        if self.peek() == Token::Semicolon {
            self.bump();
            return Ok(Stmt::Return(None));
        }

        let expr = self.parse_expression()?;
        self.eat(Token::Semicolon)?;

        Ok(Stmt::Return(Some(expr)))
    }

    /// binding powers from lowest -> highest
    fn bp_of_bin(op: BinOp) -> (u8, u8) {
        use BinOp::*;
        match op {
            Or => (1, 2),
            And => (3, 4),
            BitOr => (5, 6),
            BitXor => (7, 8),
            BitAnd => (9, 10),
            Eq | Ne => (11, 12),
            Lt | Le | Gt | Ge => (13, 14),
            Add | Sub => (15, 16),
            Mul | Div | Mod | BitShiftL | BitShiftR => (17, 18),
        }
    }

    // entry point
    pub fn parse_expression(&mut self) -> PResult<Expr> {
        self.parse_expr_bp(0)
    }

    fn parse_type(&mut self) -> PResult<Type> {
        // base type
        let base = match self.peek() {
            // primitive keywords
            Token::Ident("i8") => {
                self.bump();
                Type::Primitive(PrimTy::I8)
            }
            Token::Ident("i16") => {
                self.bump();
                Type::Primitive(PrimTy::I16)
            }
            Token::Ident("i32") => {
                self.bump();
                Type::Primitive(PrimTy::I32)
            }
            Token::Ident("i64") => {
                self.bump();
                Type::Primitive(PrimTy::I64)
            }
            Token::Ident("u8") => {
                self.bump();
                Type::Primitive(PrimTy::U8)
            }
            Token::Ident("u16") => {
                self.bump();
                Type::Primitive(PrimTy::U16)
            }
            Token::Ident("u32") => {
                self.bump();
                Type::Primitive(PrimTy::U32)
            }
            Token::Ident("u64") => {
                self.bump();
                Type::Primitive(PrimTy::U64)
            }
            Token::Ident("usize") => {
                self.bump();
                Type::Primitive(PrimTy::USize)
            }
            Token::Ident("f32") => {
                self.bump();
                Type::Primitive(PrimTy::F32)
            }
            Token::Ident("f64") => {
                self.bump();
                Type::Primitive(PrimTy::F64)
            }
            Token::Ident("bool") => {
                self.bump();
                Type::Primitive(PrimTy::Bool)
            }
            Token::Ident("cstr") => {
                self.bump();
                Type::Primitive(PrimTy::CStr)
            }
            Token::Proc => {
                self.eat(Token::Proc)?;
                self.eat(Token::LParen)?;
                let params = self.parse_extern_param_list()?;
                self.eat(Token::RParen)?;
                let ret_ty = if matches!(self.peek(), Token::Arrow) {
                    self.eat(Token::Arrow)?;
                    self.parse_type()?
                } else {
                    Type::Void
                };

                Type::FuncPtr(params.into_iter().map(|p| p.ty).collect(), Box::new(ret_ty))
            }

            // user-defined name
            Token::Ident(ident) if ident == "_" => {
                // Treat this an a user-requested elided type
                let span = self.span();
                self.bump();
                Type::Inferred(span)
            }
            Token::Ident(_) => {
                let ident = self.parse_ident()?;

                if let Some(ty) = self.generic_context.get(&ident.text) {
                    // This happens when parsing a type within a generic struct or proc def
                    ty.clone()
                } else {
                    // Otherwise, this is a fully resolved concrete type
                    let generic_tys = self.parse_generic_args()?;

                    Type::Named(ident, generic_tys)
                }
            }

            Token::Void => {
                self.bump();
                Type::Void
            }

            _ => {
                return Err(ParseError {
                    span: self.span(),
                    message: "expected type".to_string(),
                });
            }
        };

        // postfix loop (`*` and `[N]`) gotta support int** notation
        let mut ty = base;
        loop {
            ty = match self.peek() {
                // pointer
                Token::Star => {
                    self.bump();
                    Type::Ptr(Box::new(ty))
                }

                // fixed-size array
                Token::LBracket => {
                    self.bump();
                    let size_tok = self.bump();
                    let size = match size_tok {
                        //Token::IntLit(n) => n.0 as usize,
                        Token::USizeLit(n) => n as usize,
                        _ => {
                            return Err(ParseError {
                                span: self.span(),
                                message: "expected array size integer".to_string(),
                            });
                        }
                    };
                    self.eat(Token::RBracket)?;
                    Type::Array(Box::new(ty), size)
                }

                _ => break,
            };
        }

        Ok(ty)
    }

    // Pratt / precedence climbing https://en.wikipedia.org/wiki/Operator-precedence_parser
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        // parse a prefix (atomic or unary op)
        let lhs_span = self.span();
        let mut lhs = {
            match self.peek() {
                Token::InlineWat => {
                    self.eat(Token::InlineWat)?;
                    let expression_type = if self.peek() == Token::Arrow {
                        self.eat(Token::Arrow)?;
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    self.eat(Token::LBrace)?;

                    let text = self.parse_expression()?;
                    let ExprKind::Literal(Literal::Str(inner)) = text.kind else {
                        return Err(ParseError {
                            span: self.span(),
                            message: format!(
                                "inner expression of an inline wasm expression must be a string literal"
                            ),
                        });
                    };

                    self.eat(Token::RBrace)?;
                    self.new_expr(
                        lhs_span.clone(),
                        ExprKind::InlineWat(vec![], expression_type, inner),
                    )
                }
                // literals
                Token::IntLit((value, width, signed)) => {
                    self.bump();
                    self.new_expr(
                        lhs_span,
                        ExprKind::Literal(Literal::Int(value, width, signed)),
                    )
                }
                Token::USizeLit(value) => {
                    self.bump();
                    self.new_expr(lhs_span, ExprKind::Literal(Literal::USize(value)))
                }
                Token::FloatLit((value, is_32bit)) => {
                    self.bump();
                    self.new_expr(lhs_span, ExprKind::Literal(Literal::Float(value, is_32bit)))
                }
                Token::StrLiteral(s) => {
                    // TODOne: escape string
                    let lit = self.new_expr(
                        lhs_span,
                        ExprKind::Literal(Literal::Str(
                            unescape_str(s).expect("failed to parse escape sequence"),
                        )),
                    );
                    self.bump();
                    lit
                }
                // This is pretty hacky
                Token::IncludeString => {
                    self.bump();
                    let expr = self.parse_expression()?;
                    let ExprKind::Literal(Literal::Str(file_name)) = expr.kind else {
                        return Err(ParseError {
                            span: lhs_span.clone(),
                            message: "expected string literal for file name".to_string(),
                        });
                    };

                    let path = Path::new(&file_name);
                    if !path.exists() {
                        return Err(ParseError {
                            span: lhs_span.clone(),
                            message: format!("path {:?} does not exist", path),
                        });
                    }

                    let file_content = match read_to_string(path) {
                        Ok(contents) => contents,
                        Err(_) => {
                            return Err(ParseError {
                                span: lhs_span.clone(),
                                message: format!("failed to read file {:?}", path),
                            });
                        }
                    };

                    self.new_expr(lhs_span, ExprKind::Literal(Literal::Str(file_content)))
                }
                Token::True => {
                    self.bump();
                    self.new_expr(lhs_span, ExprKind::Literal(Literal::Bool(true)))
                }
                Token::False => {
                    self.bump();
                    self.new_expr(lhs_span, ExprKind::Literal(Literal::Bool(false)))
                }
                Token::Null => {
                    self.bump();
                    self.new_expr(
                        lhs_span,
                        ExprKind::Literal(Literal::Null(Type::Ptr(Box::new(Type::Void)))),
                    )
                }

                // Array literals
                Token::LBracket => {
                    self.bump();
                    let mut elems = vec![];
                    if self.peek() != Token::RBracket {
                        loop {
                            elems.push(self.parse_expression()?);
                            if self.peek() == Token::Comma {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }

                    self.eat(Token::RBracket)?;
                    self.new_expr(lhs_span, ExprKind::ArrayLit(elems))
                }

                // identifiers / path heads
                Token::Ident(_) => {
                    let ident_expr = self.parse_ident_expr()?;

                    // Check for sum type variants
                    if let ExprKind::ScopedIdent(sc_ident) = &ident_expr.kind {
                        // Should return true even with no generics
                        let lookahead_sumvar: bool =
                            self.lookahead_token_follows_generics(&Token::SumVariant);
                        if lookahead_sumvar {
                            let generics = if self.peek() == Token::TyListOpen {
                                self.parse_generic_args()?
                            } else {
                                vec![]
                            };
                            self.eat(Token::SumVariant)?;
                            let variant = self.parse_ident()?;
                            let init = if self.peek() == Token::LParen {
                                self.eat(Token::LParen)?;
                                let expr = self.parse_expression()?;
                                self.eat(Token::RParen)?;
                                Some(Box::new(expr))
                            } else {
                                None
                            };
                            self.new_expr(
                                lhs_span,
                                ExprKind::UnionLit(sc_ident.clone(), variant, init, generics),
                            )
                        } else {
                            ident_expr
                        }
                    }
                    // TODO: Struct literals here too
                    else {
                        ident_expr
                    }
                }

                // prefix unary ops
                Token::Amp => {
                    self.bump();
                    let expr = self.parse_expr_bp(u8::MAX)?;
                    let maybe_func_ref = if let ExprKind::ScopedIdent(sc_ident) = &expr.kind {
                        if self.peek() == Token::TyListOpen {
                            let generic_args = self.parse_generic_args()?;

                            Some(self.new_expr(
                                lhs_span.clone(),
                                ExprKind::FunctionReference(FunctionReference {
                                    path: sc_ident.clone(),
                                    generic_args,
                                }),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    maybe_func_ref.unwrap_or(
                        self.new_expr(lhs_span, ExprKind::Unary(UnOp::AddrOf, Box::new(expr))),
                    )
                }
                Token::Star => {
                    self.bump();
                    let expr = self.parse_expr_bp(u8::MAX)?;
                    self.new_expr(lhs_span, ExprKind::Unary(UnOp::Deref, Box::new(expr)))
                }
                Token::Minus => {
                    self.bump();
                    let expr = self.parse_expr_bp(u8::MAX)?;
                    self.new_expr(lhs_span, ExprKind::Unary(UnOp::Neg, Box::new(expr)))
                }
                Token::Bang => {
                    self.bump();
                    let expr = self.parse_expr_bp(u8::MAX)?;
                    self.new_expr(lhs_span, ExprKind::Unary(UnOp::Not, Box::new(expr)))
                }

                // parenthesised expression
                Token::LParen => {
                    self.bump();
                    let e = self.parse_expression()?;
                    self.eat(Token::RParen)?;
                    e
                }

                Token::Cast => {
                    self.bump();
                    self.eat(Token::TyListOpen)?;
                    let ty = self.parse_type()?;
                    self.eat(Token::TyListClose)?;
                    self.eat(Token::LParen)?;
                    let expr = self.parse_expression()?;
                    self.eat(Token::RParen)?;

                    self.new_expr(lhs_span, ExprKind::Cast(Box::new(expr), ty))
                }

                Token::SizeOf => {
                    self.bump();
                    self.eat(Token::TyListOpen)?;
                    let ty = self.parse_type()?;
                    self.eat(Token::TyListClose)?;
                    self.eat(Token::LParen)?;
                    self.eat(Token::RParen)?;

                    self.new_expr(lhs_span, ExprKind::Literal(Literal::SizeOf(ty)))
                }

                Token::Match => self.parse_match_expression(&lhs_span)?,
                Token::If => self.parse_if_expression(&lhs_span)?,
                Token::LBrace => self.parse_block_expression()?,

                _ => {
                    return Err(ParseError {
                        span: self.span(),
                        message: format!("unexpected token '{:?}' in expression", self.peek()),
                    });
                }
            }
        };

        // postfix loop (call, field, index)
        loop {
            let lhs_span = self.span();

            lhs = match (self.peek(), self.peek_plus_n(1), self.peek_plus_n(2)) {
                // function call
                // This is giga wrong
                (Token::LParen, _, _) | (Token::TyListOpen, _, _) => {
                    // Only parse as a function call if it is one
                    if !self.lookahead_token_follows_generics(&Token::LParen) {
                        break;
                    }

                    //  maybe parse <TypeArg>
                    let mut type_args = self.parse_generic_args()?;

                    // now parse (arg list)
                    self.eat(Token::LParen)?;
                    let mut args = Vec::new();
                    if self.peek() != Token::RParen {
                        loop {
                            args.push(self.parse_expression()?);
                            if self.peek() == Token::Comma {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.eat(Token::RParen)?;

                    // cast
                    if let ExprKind::ScopedIdent(ref id) = lhs.kind {
                        if id.to_string() == "cast" && type_args.len() == 1 && args.len() == 1 {
                            lhs = self.new_expr(
                                lhs_span,
                                ExprKind::Cast(
                                    Box::new(args.pop().unwrap()),
                                    type_args.pop().unwrap(),
                                ),
                            );
                            continue; // restart postfix loop (allow e.g. cast<T>(x).field)
                        }
                    }

                    //println!("Parsed call expression {:?} gen: {:#?}", lhs, type_args);
                    self.new_expr(lhs_span, ExprKind::Call(Box::new(lhs), args, type_args))
                }

                // Will need to either use reciever's existing pointer, or alloca one if not
                (Token::Dot, Some(Token::Ident(_i)), Some(Token::LParen)) => {
                    self.eat(Token::Dot)?;
                    let method_name = self.parse_ident()?;

                    // TODO: parse generic args to call (in addition to any inherited by the impl)
                    let generics = self.parse_generic_args()?;

                    let mut args = vec![];
                    self.eat(Token::LParen)?;
                    if self.peek() != Token::RParen {
                        loop {
                            args.push(self.parse_expression()?);
                            if self.peek() == Token::Comma {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }

                    //println!("Parsing method call");

                    self.eat(Token::RParen)?;

                    self.new_expr(
                        lhs_span,
                        ExprKind::MethodCall {
                            receiver: Box::new(lhs),
                            method_name,
                            args,
                            generic_args: generics,
                        },
                    )
                }
                // Removed explicit deref in favor of auto deref
                // (Token::Arrow, Some(Token::Ident(_i)), Some(Token::LParen)) => {
                //     self.eat(Token::Dot)?;
                //     let method_name = self.parse_ident()?;
                //     let mut args = vec![];
                //     self.eat(Token::LParen)?;
                //     if self.peek() != Token::RParen {
                //         loop {
                //             args.push(self.parse_expression()?);
                //             if self.peek() == Token::Comma {
                //                 self.bump();
                //             } else {
                //                 break;
                //             }
                //         }
                //     }
                //     self.eat(Token::RParen)?;

                //     let deref = self.new_expr(lhs_span.clone(), ExprKind::Unary(UnOp::Deref, ()))
                //     self.new_expr(
                //         lhs_span,
                //         ExprKind::MethodCall {
                //             reciever: Box::new(lhs),
                //             method_name,
                //             args,
                //         },
                //     )
                // }

                // field access "."
                (Token::Dot, _, _) => {
                    //println!("Falling back to field access ");
                    self.bump();
                    let field = self.parse_ident()?; // must be Ident
                    let maybe_deref = self.new_expr(
                        lhs_span.clone(),
                        ExprKind::Unary(UnOp::MaybeDeref, Box::new(lhs)),
                    );
                    self.new_expr(
                        lhs_span,
                        ExprKind::Field(Box::new(maybe_deref), field, false),
                    )
                }

                // pointer‑field access "->" into deref + field access
                // removed in favor of auto deref 5/30
                // (Token::Arrow, _, _) => {
                //     //println!("Falling back to field access (deref)");

                //     self.bump();
                //     let field = self.parse_ident()?;
                //     let deref = self.new_expr(
                //         lhs_span.clone(),
                //         ExprKind::Unary(UnOp::Deref, Box::new(lhs)),
                //     );
                //     self.new_expr(lhs_span, ExprKind::Field(Box::new(deref), field, true))
                // }

                // indexing
                (Token::LBracket, _, _) => {
                    self.bump();
                    let idx_expr = self.parse_expression()?;
                    self.eat(Token::RBracket)?;
                    self.new_expr(lhs_span, ExprKind::Index(Box::new(lhs), Box::new(idx_expr)))
                }
                _ => break,
            };
        }

        // binary‑operator loop (precedence climbing)
        loop {
            let lhs_span = self.span();
            // is next token a binary op?
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                Token::AmpAmp => BinOp::And,
                Token::PipePipe => BinOp::Or,
                Token::Amp => BinOp::BitAnd,
                Token::Pipe => BinOp::BitOr,
                Token::Caret => BinOp::BitXor,
                Token::Arrow2L => BinOp::BitShiftL,
                Token::Arrow2R => BinOp::BitShiftR,
                Token::EqEq => BinOp::Eq,
                Token::NotEq => BinOp::Ne,
                Token::Lt => BinOp::Lt,
                Token::Le => BinOp::Le,
                Token::Gt => BinOp::Gt,
                Token::Ge => BinOp::Ge,
                _ => break,
            };

            let (lbp, rbp) = Self::bp_of_bin(op);
            if lbp < min_bp {
                break;
            }

            self.bump(); // consume operator
            let rhs = self.parse_expr_bp(rbp)?;
            lhs = self.new_expr(lhs_span, ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)));
        }

        //eprintln!("LHS is {:?}", lhs);

        Ok(lhs)
    }

    // TODO: This is dupicated in a lot of places
    fn parse_generic_args(&mut self) -> PResult<Vec<Type>> {
        // Empty generics
        if self.peek() != Token::TyListOpen {
            return Ok(vec![]);
        }

        self.eat(Token::TyListOpen)?;
        let mut args = vec![];
        loop {
            args.push(self.parse_type()?);
            if self.peek() == Token::Comma {
                self.eat(Token::Comma)?;
            } else {
                break;
            }
        }
        self.eat(Token::TyListClose)?;
        Ok(args)
    }

    fn parse_ident_expr(&mut self) -> PResult<Expr> {
        let span = self.span();
        let ident = self.parse_ident()?;

        // TODO: Do this AFTER potentially parsing scoped ident to support namespaced structs
        if self.peek() == Token::LBrace
            || (self.peek() == Token::TyListOpen
                && self.lookahead_token_follows_generics(&Token::LBrace))
        {
            //println!("Parsing struct literal {}", ident.text);
            self.parse_struct_lit(ident, span)
        } else if self.peek() == Token::Scope {
            let mut scopes = vec![ident.text];
            self.eat(Token::Scope)?;
            // Note, can eat '::' without parsing further paths
            loop {
                if matches!(self.peek(), Token::Ident(_))
                    && self.peek_plus_n(1) == Some(Token::Scope)
                {
                    let scope_ident: Ident = self.parse_ident()?;
                    scopes.push(scope_ident.text);
                    self.eat(Token::Scope)?;
                } else {
                    break;
                }
            }

            let ident = self.parse_ident()?;

            let scoped_ident = ExprKind::ScopedIdent(ScopedIdent { scopes, ident });

            Ok(self.new_expr(span, scoped_ident))
        } else {
            // Otherwise, just plain ident
            Ok(self.new_expr(
                span,
                ExprKind::ScopedIdent(ScopedIdent {
                    scopes: vec![],
                    ident,
                }),
            ))
        }
    }

    fn lookahead_token_follows_generics(&self, tk: &Token) -> bool {
        if self.peek() != Token::TyListOpen {
            return self.peek() == *tk;
        }

        let mut n = 1;
        let mut depth = 1;

        while let Some(tok) = self.peek_plus_n(n) {
            match tok {
                Token::TyListOpen => depth += 1,
                Token::TyListClose => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            n += 1;
        }

        self.peek_plus_n(n + 1) == Some(*tk)
    }

    fn parse_struct_lit(&mut self, name: Ident, span: Span) -> PResult<Expr> {
        let generic_concrete_tys = self.parse_generic_args()?;

        self.eat(Token::LBrace)?;

        let mut fields = vec![];
        while self.peek() != Token::RBrace {
            let field_name = self.parse_ident()?;
            self.eat(Token::Colon)?;
            let value = self.parse_expression()?;
            fields.push((field_name, value));

            if self.peek() == Token::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.eat(Token::RBrace)?;

        Ok(self.new_expr(
            span,
            ExprKind::StructLit {
                name,
                fields,
                generic_inst_tys: generic_concrete_tys,
            },
        ))
    }

    // identifier token Ident struct
    fn parse_ident(&mut self) -> PResult<Ident> {
        if let Token::Ident(name) = self.peek() {
            let id = Ident {
                text: name.into(),
                span: self.span(),
            };
            self.bump();
            Ok(id)
        } else {
            Err(ParseError {
                span: self.span(),
                message: "expected identifier".to_string(),
            })
        }
    }

    fn parse_match_expression(&mut self, lhs_span: &Span) -> PResult<Expr> {
        self.eat(Token::Match)?;
        self.eat(Token::LParen)?;
        let as_ref = if self.peek() == Token::Ref {
            self.bump();
            true
        } else {
            false
        };
        let to_decomp = self.parse_expression()?;
        self.eat(Token::RParen)?;
        self.eat(Token::LBrace)?;

        let mut arms = vec![];
        let mut default_arm = None;
        while self.peek() != Token::RBrace {
            // handle default special case, which can't have bindings
            if self.peek() == Token::Default {
                self.eat(Token::Default)?;
                self.eat(Token::ThickArrow)?;
                let expression = self.parse_expression()?;
                self.eat(Token::Comma)?;
                let arm = MatchArm {
                    variant_ident: Ident::dummy(lhs_span.clone()),
                    binding: None,
                    body_expression: expression,
                };
                default_arm = Some(Box::new(arm));
                continue;
            }

            let variant_ident = self.parse_ident()?;

            let mut binding = None;
            if self.peek() == Token::LParen {
                self.eat(Token::LParen)?;
                binding = Some(self.parse_ident()?);
                self.eat(Token::RParen)?;
            }

            self.eat(Token::ThickArrow)?;

            let body_expression = self.parse_expression()?;
            self.eat(Token::Comma)?;

            let arm = MatchArm {
                variant_ident,
                binding,
                body_expression,
            };

            arms.push(arm);
        }

        self.eat(Token::RBrace)?;

        let decomp = MatchExpression {
            as_ref,
            union_to_decomp: Box::new(to_decomp),
            arms,
            default_arm,
        };
        Ok(self.new_expr(lhs_span.clone(), ExprKind::Match(decomp)))
    }
}

// ehhh ...
fn unescape_str(input: &str) -> Result<String, String> {
    let mut result = String::new();
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('0') => result.push('\0'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('\\') => result.push('\\'),
                Some('x') => {
                    let hi = chars.next();
                    let lo = chars.next();
                    if let (Some(h), Some(l)) = (hi, lo) {
                        let hex = format!("{}{}", h, l);
                        if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                            result.push(byte as char);
                        } else {
                            return Err(format!("Invalid hex escape: \\x{}", hex));
                        }
                    } else {
                        return Err("Incomplete \\x escape".into());
                    }
                }
                Some('u') => {
                    if chars.next() != Some('{') {
                        return Err("Invalid \\u escape: expected '{{'".into());
                    }
                    let mut unicode = String::new();
                    while let Some(ch) = chars.next() {
                        if ch == '}' {
                            break;
                        }
                        unicode.push(ch);
                    }
                    if let Ok(codepoint) = u32::from_str_radix(&unicode, 16) {
                        if let Some(ch) = char::from_u32(codepoint) {
                            result.push(ch);
                        } else {
                            return Err(format!("Invalid unicode escape: \\u{{{}}}", unicode));
                        }
                    } else {
                        return Err(format!("Invalid unicode digits: \\u{{{}}}", unicode));
                    }
                }
                Some(unknown) => return Err(format!("Unknown escape: \\{}", unknown)),
                None => return Err("Incomplete escape sequence".into()),
            }
        } else {
            result.push(c);
        }
    }
    Ok(result)
}
