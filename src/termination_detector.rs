use std::sync::atomic::{AtomicUsize, Ordering};

use ferrompi::Communicator;
use futures::{FutureExt as _, select};
use serde::{Deserialize, Serialize};
use smol::lock::Mutex;
use tracing::trace;

use crate::{
    CircleError,
    async_mpi::{self, SOURCE_ANY, Tag},
};

// Token color for Dijkstra's distributed termination algorithm.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TokenColor {
    #[default]
    White = 0,
    Black = 1,
}

/// A token sent between ranks to detect termination.
#[derive(Debug, Serialize, Deserialize)]
struct Token {
    color: TokenColor,
    originating_rank: i32,
}

/// State for Dijkstra's distributed termination detection algorithm. This is used to track the state of the algorithm
/// on each rank.
///
/// NOTE: This implementation requires we DO NOT make progress on termination detection until we are idle!
#[derive(Debug)]
pub struct TerminationDetectionState {
    /// The rank of our parent. When idle, we send receipt of our idle status.
    parent_rank: Option<i32>,
    /// Receipts for messages we've sent to others, but have not yet received acknowledgments for.
    /// When this becomes zero, it means there are no messages in transit, and we notify our parent of our idle status.
    pending_receipts: AtomicUsize,
    // The number of tokens we are waiting to receive from our children before we can check for termination.
    pending_tokens: AtomicUsize,
    /// The color of the token we hold. Used in Dijkstra's distributed termination algorithm.
    token_color: Mutex<TokenColor>,
    /// Tokens we've received from our children, in the case of the root, we collect all white tokens until we find a
    /// black token or have received a white token from every other rank. If we have received a black token, we tell the
    /// originating rank to reset.
    queued_tokens: Mutex<Vec<Token>>,
}

impl TerminationDetectionState {
    pub fn new(comm: &Communicator) -> Self {
        // There are a few different options for how to initialize the tree structure.
        // For now, we make rank 0 the root, and the parent of rank i be i-1, so we have a simple chain.
        // This means that rank i will be the only rank required to be reset when failing to detect termination.
        let (parent_rank, pending_tokens) = if comm.rank() == 0 {
            (None, 1)
        } else {
            (Some(comm.rank() - 1), 0)
        };
        // The last rank starts with a token
        let queued_tokens = if comm.rank() == comm.size() - 1 {
            Mutex::new(vec![Token {
                color: TokenColor::White,
                originating_rank: comm.rank(),
            }])
        } else {
            Mutex::new(Vec::new())
        };

        Self {
            parent_rank,
            pending_receipts: AtomicUsize::new(0),
            pending_tokens: AtomicUsize::new(pending_tokens),
            token_color: Mutex::new(TokenColor::White),
            queued_tokens,
        }
    }

    /// When we send a message to another rank, we mark ourselves as having sent a message, and having thus potentially
    /// dirtied the tree of idle processes. This will mark our, and our parents as having sent a message, so we will
    /// know to try and detect termination again.
    pub async fn on_message_sent(&self) {
        self.pending_receipts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut token_color = self.token_color.lock().await;
        *token_color = TokenColor::Black;
    }

    /// We've received acknowledgement of a message we've sent containing work, so we can mark that message as no longer
    /// in transit. We keep track of this so that we can ensure that there's no in-flight messages when we claim
    /// termination.
    pub fn on_receipt_received(&self) {
        self.pending_receipts
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub async fn forward_tokens(&self, comm: &Communicator) -> Result<(), CircleError> {
        let Some(parent_rank) = self.parent_rank else {
            // The root rank never forwards tokens through this method.
            return smol::future::pending().await;
        };
        loop {
            let mut queued_tokens = self.queued_tokens.lock().await;
            if queued_tokens.is_empty() {
                // If we have no tokens to forward, we can just wait until we have some tokens to forward.
                drop(queued_tokens);
                smol::future::yield_now().await;
                continue;
            }
            trace!(
                "Have {} tokens to forward to parent rank {parent_rank}",
                queued_tokens.len()
            );
            // If we have tokens to forward, we can forward them to our parent and clear our queue.
            // If we're black, we need to recolor the token to black.
            let mut token_color = self.token_color.lock().await;
            for mut token in queued_tokens.drain(..) {
                token.color = match token.color {
                    TokenColor::Black => {
                        *token_color = TokenColor::Black;
                        TokenColor::Black
                    }
                    TokenColor::White => *token_color,
                };
                let token_buf = postcard::to_allocvec(&token)?;
                async_mpi::isend(comm, &token_buf, parent_rank, Tag::TerminationToken as i32)
                    .await?;
            }
            // After we've forwarded the token, we can reset our color to white.
            trace!("Resetting token to white after having forwarded to parent rank {parent_rank}");
            *token_color = TokenColor::White;
        }
    }

    async fn reset_tree(&self, comm: &Communicator) -> Result<(), CircleError> {
        let mut token_color = self.token_color.lock().await;
        self.pending_tokens
            .store(self.queued_tokens.lock().await.len(), Ordering::SeqCst);
        *token_color = TokenColor::White;
        let mut queued_tokens = self.queued_tokens.lock().await;
        trace!("Resetting tree with {} pending tokens", queued_tokens.len());
        self.pending_tokens
            .store(queued_tokens.len(), Ordering::SeqCst);

        for mut token in queued_tokens.drain(..) {
            token.color = TokenColor::White;
            let token_buf = postcard::to_allocvec(&token)?;
            async_mpi::isend(
                comm,
                &token_buf,
                token.originating_rank,
                Tag::TerminationToken as i32,
            )
            .await?;
        }
        trace!("Finished resetting tree");
        Ok(())
    }

    async fn on_token_received(
        &self,
        comm: &Communicator,
        mut token: Token,
    ) -> Result<(), CircleError> {
        match self.parent_rank {
            Some(parent_rank) => {
                // If we're not the root, we just forward the token to our parent, potentially recoloring it if we've
                // sent messages. After we've forwarded the token, we can reset our color to white.
                let token = match token.color {
                    TokenColor::Black => {
                        // If the token is black, it stays black and we reset our color.
                        let mut token_color = self.token_color.lock().await;
                        let current_receipts = self.pending_receipts.load(Ordering::SeqCst);
                        if current_receipts == 0 {
                            *token_color = TokenColor::White;
                        }
                        token
                    }
                    TokenColor::White => {
                        // If the token is white, it becomes black iff we've sent messages
                        let mut token_color = self.token_color.lock().await;
                        token.color = *token_color;
                        let current_receipts = self.pending_receipts.load(Ordering::SeqCst);
                        if current_receipts == 0 {
                            trace!("Resetting token to white");
                            *token_color = TokenColor::White;
                        }
                        token
                    }
                };
                trace!(
                    "Forwarding token originating from rank {} with color {:?} to parent rank {parent_rank}",
                    token.originating_rank, token.color
                );
                let token_buf = postcard::to_allocvec(&token)?;
                async_mpi::isend(comm, &token_buf, parent_rank, Tag::TerminationToken as i32)
                    .await?;
            }
            None => {
                // If we're the root, we collect tokens until we've received a black token. After we've received a black
                // token, we send a reset notification to the originating rank.
                let mut queued_tokens = self.queued_tokens.lock().await;
                queued_tokens.push(token);
                self.pending_tokens.fetch_sub(1, Ordering::SeqCst);
                let current_tokens = self.pending_tokens.load(Ordering::SeqCst);
                if current_tokens != 0 {
                    trace!(
                        "Waiting for {current_tokens} pending tokens before validating termination"
                    );
                    return Ok(());
                }
                trace!("Received all pending tokens, validating termination");
                // If our own token is black, we reset, no need to check other tokens
                {
                    let token_color = self.token_color.lock().await;
                    if *token_color == TokenColor::Black {
                        trace!("Root has a black token, resetting tree");
                        drop(token_color);
                        drop(queued_tokens);
                        self.reset_tree(comm).await?;
                        return Ok(());
                    }
                }
                // If there are no pending tokens, we can check for termination by looking for any black tokens in
                // the queue. If there are no black tokens, we can claim termination and broadcast a termination
                // notification. If there is a black token, we need to reset the tree by sending a reset
                // notification to the originating rank of all tokens.
                if queued_tokens.iter().all(|t| t.color == TokenColor::White) {
                    trace!("Termination detected, broadcasting termination notification");
                    for rank in 0..comm.size() {
                        async_mpi::isend(comm, &[0], rank, Tag::TerminationConfirmed as i32)
                            .await?;
                    }
                    return Ok(());
                }
                // Otherwise, we need to reset the tree by sending a reset token to the originating rank of each
                // token.
                drop(queued_tokens);
                trace!("One of the tokens is black, resetting tree");
                self.reset_tree(comm).await?;
            }
        }
        Ok(())
    }

    /// Checks whether termination has been detected, processing any incoming tokens as necessary.
    pub async fn monitor_termination(&self, comm: &Communicator) -> Result<(), CircleError> {
        loop {
            trace!("Progressing termination detection");
            select! {
                // We've received a token from a child
                token = async_mpi::receive_tagged(comm, SOURCE_ANY, Tag::TerminationToken as i32).fuse() => {
                    let (source, token) = token?;
                    let token: Token = postcard::from_bytes(&token)?;
                    trace!("Received token from {} originating from rank {} with color {:?}", source, token.originating_rank, token.color);
                    self.on_token_received(comm, token).await?;
                },
                // We've received a termination notification, so we can exit gracefully.
                _ = async_mpi::receive_tagged::<u8>(comm, SOURCE_ANY, Tag::TerminationConfirmed as i32).fuse() => {
                    trace!("Received termination notification, exiting gracefully");
                    return Ok(());
                },
                // We can make some progress on forwarding tokens up the tree, if we have any tokens to forward.
                _ = self.forward_tokens(comm).fuse() => {
                    trace!("Making progress on forwarding tokens up the tree");
                }
            }
        }
    }
}
