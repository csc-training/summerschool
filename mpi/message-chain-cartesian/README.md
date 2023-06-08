## Message chain

See [the earlier message chain exercise](../message-chain) for the description of the message chain.

### Message chain with Cartesian communicator

1. Create a Cartesian topology for the chain. Utilize `MPI_Cart_shift` for finding
   the neighbouring ranks and implement the communication with MPI point-to-point routines
   (either blocking or non-blocking).
   You may start from scratch, or use the skeleton code or
   your solution from [the earlier message chain exercise](../message-chain)
   as a starting point.

2. Make a version where the chain is periodic, i.e. task `ntasks-1` sends to task `0`
   and every task receives.
