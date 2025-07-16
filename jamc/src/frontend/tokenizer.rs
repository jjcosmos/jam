use logos::{Logos, Span};

use crate::report_error;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t\n\f\r]+")]
pub enum Token<'src> {
    // Define patterns for tokens
    #[token("@include")]
    Include,
    #[token("#wat")]
    InlineWat,
    // Hacky
    #[token("#inc_str")]
    IncludeString,
    #[token("sum")]
    Sum,
    #[token("match")]
    Match,
    #[token(":>")]
    SumVariant,
    #[token("=>")]
    ThickArrow,
    #[token("ref")]
    Ref,
    #[token("default")]
    Default,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("extern")]
    Extern,
    #[token("impl")]
    Impl,
    #[token("proc")]
    Proc,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("static")]
    Static,
    #[token("const")]
    Const,
    #[token("mut")]
    Mut,
    #[token("let")]
    Let,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[regex("//[^\n]*", logos::skip)]
    #[regex("/\\*([^*]|\\*[^/])*\\*/", logos::skip)] // For block comments
    Comment,
    // Operators and punctuation
    #[token("<?")]
    TyListOpen,
    #[token("?>")]
    TyListClose,
    // Compound
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AmpEq,
    #[token("|=")]
    PipeEq,
    #[token("^=")]
    CaretEq,
    #[token("<<=")]
    Arrow2LEq,
    #[token(">>=")]
    Arrow2REq,

    #[token("<<")]
    Arrow2L,
    #[token(">>")]
    Arrow2R,
    #[token("::")]
    Scope,
    #[token("->")]
    Arrow,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("&")]
    Amp, // address-of or bitwise AND
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("...")]
    Elipses,
    #[token(".")]
    Dot,
    #[token("!")]
    Bang,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("=")]
    Assign,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("cast")]
    Cast,
    #[token("sizeof")]
    SizeOf,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("return")]
    Return,
    #[token("\0")]
    Eof,
    #[end]
    End,

    // Literals
    // Optional decimal part (e.g., "42f32" or "42.5f32")
    #[regex(r"-?[0-9]+(\.[0-9]+)?f32", |lex| float_tok(lex, true ))] // f32
    #[regex(r"-?[0-9]+(\.[0-9]+)?f64", |lex| float_tok(lex, false))] // f64
    FloatLit((f64, bool)), //value, is 32 bit

    // Radix based literals
    #[regex(r"0x[0-9a-fA-F]+", |lex| parse_int_literal_radix(lex, 16, 2, 4))] // hex
    #[regex(r"0b[01]+",        |lex| parse_int_literal_radix(lex, 2,  2, 1))] // binary
    #[regex(r"0o[0-7]+",       |lex| parse_int_literal_radix(lex, 8,  2, 3))] // ocatal
    #[regex(r"[0-9]+u8",       |lex| int_tok(lex,  8, false))]
    #[regex(r"[0-9]+u16",      |lex| int_tok(lex, 16, false))]
    #[regex(r"[0-9]+u32",      |lex| int_tok(lex, 32, false))]
    #[regex(r"[0-9]+u64",      |lex| int_tok(lex, 64, false))]
    #[regex(r"-?[0-9]+i8",       |lex| int_tok(lex,  8,  true))]
    #[regex(r"-?[0-9]+i16",      |lex| int_tok(lex, 16, true))]
    #[regex(r"-?[0-9]+i32",      |lex| int_tok(lex, 32, true))]
    #[regex(r"-?[0-9]+i64",      |lex| int_tok(lex, 64, true))]
    #[regex(r"'([^'\\]|\\.)'", |lex| char_lit_2_int(lex))]
    //#[regex(r"-?[0-9]+",         |lex| int_tok(lex, 64, true))] // default i64
    IntLit((i128, u8, bool)), // value, width, signed
    // Trying this as a separate token
    #[regex(r"-?[0-9]+usize",      |lex| usize_tok(lex))]
    USizeLit(u64),

    // TODO: gerneric number for type inferance?
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        // strip quotes and handle escapes for string literal
        let slice = lex.slice();
        let content = &slice[1..slice.len()-1];
        // need to unescape content here?
        content
    })]
    StrLiteral(&'src str),
    #[regex("[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice())]
    Ident(&'src str),

    #[token("void")]
    Void,
    #[token("null")]
    Null,
}

fn parse_int_literal_radix<'a>(
    lex: &mut logos::Lexer<'a, Token<'a>>,
    radix: u32,
    prefix_len: usize,
    bits_per_digit: usize,
) -> Option<(i128, u8, bool)> {
    let slice = &lex.slice()[prefix_len..]; // Strip prefix (e.g., 2 for "0x")
    let int_width = (slice.len() * bits_per_digit).next_power_of_two();

    let value = i128::from_str_radix(slice, radix).ok()?;
    // clamp int width to 8 bits minimum unless supporting smaller ints (maybe TODO?)
    Some((value, int_width.max(8) as u8, false))
}

fn char_lit_2_int<'a>(lex: &mut logos::Lexer<'a, Token<'a>>) -> Option<(i128, u8, bool)> {
    let slice = lex.slice(); // 'A'

    if slice.len() < 3 {
        return None;
    }

    let inner = &slice[1..slice.len() - 1]; // strip quotes

    let c = if inner.starts_with('\\') {
        match &inner[1..] {
            "n" => '\n',
            "r" => '\r',
            "t" => '\t',
            "\\" => '\\',
            "'" => '\'',
            "\"" => '"',
            "0" => '\0',
            _ => return None, // unknown escape
        }
    } else {
        inner.chars().next()?
    };

    Some((c as i128, 32, true))
}

fn int_tok<'a>(lex: &mut logos::Lexer<'a, Token<'a>>, bits: u8, signed: bool) -> (i128, u8, bool) {
    let raw = trim_type_suffix(lex.slice());
    (
        raw.parse::<i128>()
            .expect(&format! {"{} is not valid form for a int", &raw}),
        bits,
        signed,
    )
}

fn usize_tok<'a>(lex: &mut logos::Lexer<'a, Token<'a>>) -> u64 {
    let raw = trim_type_suffix(lex.slice());

    raw.parse::<u64>()
        .expect(&format! {"{} is not valid form for a int", &raw})
}

fn float_tok<'a>(lex: &mut logos::Lexer<'a, Token<'a>>, is_f32: bool) -> (f64, bool) {
    //let raw = lex.slice().trim_end_matches('f'); // strip optional 'f'
    let raw = trim_type_suffix(lex.slice());
    (
        raw.parse::<f64>()
            .expect(&format! {"{} is not valid form for a float", &raw}),
        is_f32,
    )
}

// Unused, remove later
fn _parse_float<'a>(lex: &mut logos::Lexer<'a, Token<'a>>) -> f64 {
    let raw = lex.slice();
    let numeric = raw.strip_suffix(|c| c == 'f').unwrap_or(raw);
    numeric.parse::<f64>().unwrap()
}

fn trim_type_suffix(s: &str) -> &str {
    match s.find(|c| c == 'u' || c == 'i' || c == 'f') {
        Some(index) => &s[..index],
        None => s,
    }
}

// Unused, remove later
pub fn _lex_source(source: &str) -> Vec<Token> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();
    while let Some(token_res) = lexer.next() {
        if let Ok(token) = token_res {
            tokens.push(token);
        } else {
            panic!(
                "Unexpected token {:?} at position {}",
                lexer.slice(),
                lexer.span().start
            );
        }
    }
    tokens
}

pub fn lex_source_span(source: &str, source_id: usize) -> Vec<(Token, Span)> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();
    while let Some(token_res) = lexer.next() {
        if let Ok(token) = token_res {
            tokens.push((token, lexer.span()));
        } else {
            let msg = format!(
                "Unexpected token {:?} at position {}",
                lexer.slice(),
                lexer.span().start
            );

            // TODO: Fix file name. Also I think report_error uses this span wrong?
            report_error(&source, &lexer.span(), &msg, &format!("file:{}", source_id));
            panic!(
                "Unexpected token {:?} at position {}",
                lexer.slice(),
                lexer.span().start
            );
        }
    }
    tokens.push((Token::Eof, lexer.span()));
    tokens
}
