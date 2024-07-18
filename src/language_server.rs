use std::sync::{Arc, Mutex};

use tokio::io::{Stdin, Stdout};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::info;

use crate::document_store::DocumentStore;

#[derive(Debug)]
struct Backend {
    client: Client,
    document_store: Arc<Mutex<DocumentStore>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        will_save: None,
                        will_save_wait_until: None,
                        save: Some(TextDocumentSyncSaveOptions::Supported(true)),
                    },
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions::default()),
                definition_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "server initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    /// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion
    async fn completion(&self, _: CompletionParams) -> Result<Option<CompletionResponse>> {
        Ok(Some(CompletionResponse::Array(vec![
            CompletionItem::new_simple("Hello".to_string(), "Some detail".to_string()),
            CompletionItem::new_simple("Bye".to_string(), "More detail".to_string()),
        ])))
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.get(0) {
            if let Ok(mut store) = self.document_store.lock() {
                store.upsert_document(params.text_document.uri.to_string(), &change.text);
                info!("updated document on store!",);
            }
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        if let Ok(mut store) = self.document_store.lock() {
            store.upsert_document(
                params.text_document.uri.to_string(),
                &params.text_document.text,
            );
            info!("inserted document into store!",);
        }
    }

    async fn hover(&self, _: HoverParams) -> Result<Option<Hover>> {
        Ok(Some(Hover {
            contents: HoverContents::Scalar(MarkedString::String("You're hovering!".to_string())),
            range: None,
        }))
    }
}

pub async fn init(stdin: Stdin, stdout: Stdout) {
    let document_store = Arc::new(Mutex::new(DocumentStore::new()));
    let (service, socket) = LspService::new(|client| Backend {
        document_store,
        client,
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
