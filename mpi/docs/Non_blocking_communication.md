---
title:  Non-blocking communication 
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Non-blocking communication 

* Non-blocking communication operations return immediately and perform sending/receiving in the background
	- `MPI_Isend` & `MPI_Irecv`
* Enables some computing concurrently with communication
* Avoids many common dead-lock situations
* Also collective operations available as non-blocking versions

# Non-blocking send

**`MPI_Isend`(`buffer`{.input}, `count`{.input}, `datatype`{.input}, `dest`{.input}, `tag`{.input}, `comm`{.input}, `request`{.output})**  
Parameters similar to `MPI_Send` but has an additional request parameter  
` `    
`buffer`{.input}	send buffer that must not be written to until one has checked that the operation is over   
` `  
`request`{.output} 	a handle that is used when checking if the operation has finished (integer in Fortran,MPI_Request in C)  

# Non-blocking receive


**`MPI_Irecv`(`buffer`{.output}, `count`{.input}, `datatype`{.input}, `source`{.input}, `tag`{.input}, `comm`{.input}, `request`{.output})**  
Parameters similar to `MPI_Recv` but has no status parameter  
` `    
`buffer`{.output} receive buffer guaranteed to contain the data only after one has checked that the operation is over   	
` `  
`request`{.output} 	a handle that is used when checking if the operation has finished

# Non-blocking communication

* Important: Send/receive operations have to be finalized
	- `MPI_Wait`, `MPI_Waitall`,…
		* Waits for the communication started with `MPI_Isend` or `MPI_Irecv` to finish (blocking)
	- `MPI_Test`,…
		* Tests if the communication has finished (non-blocking)
* You can mix non-blocking and blocking routines
	- e.g., receive a message sent by `MPI_Isend` with `MPI_Recv`

# Wait for non-blocking operation

**`MPI_Wait`(`request`{.input}, `status`{.output})**  

`request`{.input}	handle of the non-blocking communication  
`status`{.output} 	status of the completed communication, see `MPI_Recv`

A call to `MPI_WAIT` returns when the operation identified by request is complete

# Non-blocking test for non-blocking operations


