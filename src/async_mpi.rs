use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use ferrompi::{Communicator, MpiDatatype};
use tracing::trace;

pub const SOURCE_ANY: i32 = -1;

// TODO: There are libraries that make this kind of thing saner. Use one.
/// Tag for sending and receiving messages between ranks.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub(crate) enum Tag {
    WorkRequest = 10,
    WorkResponse,
    WorkAck,
    TerminationToken,
    TerminationConfirmed,
}

impl From<i32> for Tag {
    fn from(value: i32) -> Self {
        match value {
            10 => Tag::WorkRequest,
            11 => Tag::WorkResponse,
            12 => Tag::WorkAck,
            13 => Tag::TerminationToken,
            14 => Tag::TerminationConfirmed,
            _ => panic!("Invalid tag value: {value}"),
        }
    }
}

struct WrappedRequest(Option<ferrompi::Request>);

impl Future for WrappedRequest {
    type Output = Result<(), ferrompi::Error>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let request = self.0.as_mut().expect("Request should only be polled once");
        match request.test() {
            Ok(true) => {
                let request = self.0.take().expect("Request should only be polled once");
                Poll::Ready(request.wait())
            }
            Ok(false) => {
                _cx.waker().wake_by_ref(); // Wake up the task to poll again later
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

pub async fn wrap_request(request: ferrompi::Request) -> Result<(), ferrompi::Error> {
    WrappedRequest(Some(request)).await
}

/// Checks whether a matching message is available without blocking. Returns the status of the probe, if successful.
pub async fn probe_tag(
    comm: &Communicator,
    source: i32,
    tag: i32,
) -> Result<ferrompi::Status, ferrompi::Error> {
    trace!(
        "Probing for message from source {source} with tag {tag:?}",
        tag = Tag::from(tag)
    );
    loop {
        match comm.iprobe::<u8>(source, tag) {
            Ok(Some(status)) => return Ok(status),
            // TODO: Investigate better strategies for waking.
            Ok(None) => smol::future::yield_now().await,
            Err(e) => return Err(e),
        }
    }
}

pub async fn irecv(
    comm: &Communicator,
    buffer: &mut [u8],
    source: i32,
    tag: i32,
) -> Result<(), ferrompi::Error> {
    trace!(
        "Initiating non-blocking receive from source {source} with tag {tag:?}",
        tag = Tag::from(tag)
    );
    let request = comm.irecv(buffer, source, tag)?;
    wrap_request(request).await
}

pub async fn isend(
    comm: &Communicator,
    buffer: &[u8],
    dest: i32,
    tag: i32,
) -> Result<(), ferrompi::Error> {
    trace!(
        "Initiating non-blocking send to destination {dest} with tag {tag:?}",
        tag = Tag::from(tag)
    );
    let request = comm.isend(buffer, dest, tag)?;
    wrap_request(request).await
}

/// Receives a tagged message of vectored type `T` from the specified source and tag. `T` must be an `MpiDatatype`, and
/// the number of elements received is determined by the count returned by a probe.
pub async fn receive_tagged<T>(
    comm: &Communicator,
    source: i32,
    tag: i32,
) -> ferrompi::Result<(i32, Vec<T>)>
where
    T: MpiDatatype,
{
    let status = probe_tag(comm, source, tag).await?;
    let mut buf: Vec<T> = Vec::with_capacity(status.count as usize);
    let (actual_source, actual_tag, actual_count) = comm.recv(
        unsafe { buf.spare_capacity_mut().assume_init_mut() },
        status.source,
        tag,
    )?;
    assert!(
        actual_tag == tag,
        "Received message with unexpected tag: expected {tag:?}, got {actual_tag:?}"
    );
    assert!(
        actual_count == status.count,
        "Received message with unexpected count: expected {}, got {}",
        status.count,
        actual_count
    );
    assert!(
        actual_source == status.source,
        "Received message from unexpected source: expected {}, got {}",
        status.source,
        actual_source
    );
    trace!(
        "Successfully received tagged message from source {} with tag {tag:?}",
        status.source,
        tag = Tag::from(tag)
    );
    unsafe {
        buf.set_len(status.count as usize);
    }
    Ok((status.source, buf))
}
