use crate::simple_parser::Module;

pub struct ModuleIndex {
    docs: String,
    scope: ScopeIndex,
    attributes: Vec<String>,
    functions: Vec<String>,
    macros: Vec<String>,
}

pub struct ScopeIndex {
    parent: Option<Box<ScopeIndex>>,
    imports: Vec<String>,
    aliases: Vec<String>,
    requires: Vec<String>,
    variables: Vec<String>,
    modules: Vec<String>,
}

pub fn build_module_index(module: Module) -> ModuleIndex {
    todo!();
}

#[cfg(test)]
mod tests {
    #[test]
    fn index_module() {
        assert_eq!(true, false)
    }
}
