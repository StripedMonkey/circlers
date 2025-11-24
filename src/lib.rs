use mpi::{topology::SimpleCommunicator, traits::Communicator};

struct Circle {
    comm: SimpleCommunicator,
}

impl Circle {
    fn new(comm: &dyn Communicator) -> Self {
        let comm = comm.duplicate();
        Circle { comm }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
