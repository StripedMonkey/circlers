use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use ferrompi::{Communicator, MpiDatatype};
use tracing::trace;

use crate::Tag;

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

pub async fn receive_tagged<T>(
    comm: &Communicator,
    source: i32,
    tag: i32,
) -> ferrompi::Result<Vec<T>>
where
    T: MpiDatatype,
{
    let status = probe_tag(comm, source, tag).await?;
    let mut buf: Vec<T> = Vec::with_capacity(status.count as usize);
    wrap_request(comm.irecv(
        unsafe { buf.spare_capacity_mut().assume_init_mut() },
        source,
        tag,
    )?)
    .await?;
    trace!(
        "Successfully received tagged message from source {source} with tag {tag:?}",
        tag = Tag::from(tag)
    );
    unsafe {
        buf.set_len(status.count as usize);
    }
    Ok(buf)
}
