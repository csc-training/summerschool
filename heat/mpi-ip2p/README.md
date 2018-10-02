## Non-blocking communication in heat equation

Implement the halo exchange in the heat equation solver using non-blocking
communication. For some additional challenge, you can overlap the update
(`evolve`) of the interior part of the domain (those independent on the halo 
areas) with the exchange routine. This approach implies that you will need to 
divide these into two stages both. The more straightforward approach is to 
replace just the communication operations in `exchange` with non-blocking 
operations.
