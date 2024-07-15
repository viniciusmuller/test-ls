use std::error::Error;

use actix::Addr;
use log::{info, trace};
use lsp_server::{Connection, Message, RequestId};
use lsp_types::{CompletionItemKind, OneOf, ServerCapabilities};
use serde_json::{json, Value};

use crate::{
    completion_engine::CursorContext,
    completion_engine_actor::{CompletionEngineActor, CompletionEngineMessage},
};

pub struct LanguageServer {
    completion_engine_addr: Addr<CompletionEngineActor>,
    connection: Connection,
}

impl LanguageServer {
    pub fn new(
        completion_engine_addr: Addr<CompletionEngineActor>,
        connection: Connection,
    ) -> Self {
        Self {
            completion_engine_addr,
            connection,
        }
    }
}

pub async fn start_server(
    completion_engine_addr: Addr<CompletionEngineActor>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    // Note that  we must have our logging only write out to stderr.
    info!("Starting generic LSP server");

    // Create the transport. Includes the stdio (stdin and stdout) versions but this could
    // also be implemented to use sockets or HTTP.
    let (connection, io_threads) = Connection::stdio();

    let language_server = LanguageServer::new(completion_engine_addr, connection);

    // Run the server and wait for the two threads to end (typically by trigger LSP Exit event).
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(lsp_types::CompletionOptions {
            resolve_provider: None,
            trigger_characters: Some(vec![".".to_owned()]),
            all_commit_characters: None, // TODO
            work_done_progress_options: lsp_types::WorkDoneProgressOptions {
                work_done_progress: None,
            },
            completion_item: Some(lsp_types::CompletionOptionsCompletionItem {
                label_details_support: Some(false), // TODO
            }),
        }),
        ..Default::default()
    })
    .unwrap();

    match language_server.connection.initialize(server_capabilities) {
        Ok(it) => it,
        Err(e) => {
            if e.channel_is_disconnected() {
                io_threads.join()?;
            }
            return Err(e.into());
        }
    };
    lsp_loop(&language_server).await?;
    io_threads.join()?;

    // Shut down gracefully.
    info!("Shutting down server");
    Ok(())
}

async fn lsp_loop(language_server: &LanguageServer) -> Result<(), Box<dyn Error + Sync + Send>> {
    eprintln!("starting example main loop");
    for msg in &language_server.connection.receiver {
        trace!("got msg: {msg:?}");
        match msg {
            Message::Request(ref req) => {
                let lsp_server::Request { id, method, params } = req;

                if language_server.connection.handle_shutdown(req)? {
                    return Ok(());
                }

                match method.as_str() {
                    "textDocument/completion" => {
                        handle_completion_request(id.clone(), &language_server, &params).await
                    }
                    _ => (),
                }
            }
            Message::Response(resp) => {
                trace!("got response: {resp:?}");
            }
            Message::Notification(not) => {
                trace!("got notification: {not:?}");
            }
        }
    }
    Ok(())
}

/// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion
async fn handle_completion_request(
    id: RequestId,
    language_server: &LanguageServer,
    params: &Value,
) {
    let position = params.get("position").unwrap();
    let line = position.get("line").unwrap().as_u64().unwrap();
    let col = position.get("character").unwrap().as_u64().unwrap();
    let text_document = params
        .get("textDocument")
        .unwrap()
        .get("uri")
        .unwrap()
        .as_str()
        .unwrap();
    let file_abspath = text_document.strip_prefix("file://").unwrap();

    trace!(
        "got completion request! {:?}, {}",
        (line, col),
        file_abspath
    );

    let items = language_server
        .completion_engine_addr
        .send(CompletionEngineMessage::Query(CursorContext::Module(
            "Comp".to_owned(),
        )))
        .await
        .unwrap();

    let items_lsp = items
        .into_iter()
        .map(|i| match i {
            crate::completion_engine::CompletionItem::Module(module_name) => {
                lsp_types::CompletionItem {
                    label: module_name,
                    label_details: None,
                    kind: Some(CompletionItemKind::MODULE),
                    detail: None,
                    documentation: None,
                    deprecated: None,
                    preselect: None,
                    sort_text: None,
                    filter_text: None,
                    insert_text: None,
                    insert_text_format: None,
                    insert_text_mode: None,
                    text_edit: None,
                    additional_text_edits: None,
                    command: None,
                    commit_characters: None,
                    data: None,
                    tags: None,
                }
            }
            crate::completion_engine::CompletionItem::Function(_) => todo!(),
            crate::completion_engine::CompletionItem::ModuleFunction(_) => todo!(),
        })
        .collect::<Vec<_>>();

    language_server
        .connection
        .sender
        .send(Message::Response(lsp_server::Response {
            id,
            result: Some(json!(items_lsp)),
            error: None,
        }))
        .unwrap();
}
