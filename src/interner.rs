use std::sync::{Arc, RwLock};

use lazy_static::lazy_static;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

lazy_static! {
    pub static ref INTERNER: Arc<RwLock<StringInterner<DefaultBackend>>> =
        Arc::new(RwLock::new(StringInterner::default()));
}

pub fn intern_string(s: &str) -> DefaultSymbol {
    let mut interner = INTERNER.write().unwrap();
    interner.get_or_intern(s)
}

pub fn get_string(s: &str) -> Option<DefaultSymbol> {
    let interner = INTERNER.read().unwrap();
    interner.get(s)
}

pub fn resolve_string<'a>(s: DefaultSymbol) -> Option<String> {
    let interner = INTERNER.read().unwrap();
    interner.resolve(s).map(|s| s.to_owned())
}
