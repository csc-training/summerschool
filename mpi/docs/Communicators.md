---
title: User-defined communicators 
author: CSC Summerschool
date:   2019-06
lang:   en
---

# Communicators

* The communicator determines the "communication universe" 
	- The source and destination of a message is identified by process rank within the communicator
* So far: `MPI_COMM_WORLD`
* Processes can be divided into subcommunicators
	- Task level parallelism with process groups performing separate tasks Parallel I/O

# Communicators

* Communicators are dynamic
* A task can belong simultaneously to several communicators
	- In each of them it has a unique ID, however
	- Communication is normally within the communicator

# Creating a new communicator

![](images/communicator.svg){.center width=80%}

# Creating new communicator

* `MPI_Comm_split` creates new communicators based on 'colors' and 'keys'   
	
**`MPI_Comm_split`(`comm`{.input}, `color`{.input}, `key`{.input}, `newcomm`{.output})**  
`comm`{.input} 		communicator handle  
`color`{.input}		control of subset assignment, processes with the same color belong to the same new communicator  
`key`{.input} 		control of rank assignment    
`newcomm`{.output}	new communicator handle  

**If color = `MPI_UNDEFINED`, a process does not belong to any of the new communicators**

# Creating new communicator

<div class=column>
```c
if (myid%2 == 0) {
  color = 1;
} else {
  color = 2;
}
MPI_Comm_split(MPI_COMM_WORLD, color
	, myid, &subcomm);

MPI_Comm_rank(subcomm, &mysubid);

printf ("I am rank %d in MPI_COMM_WORLD,but 
	%d in Comm %d.\n", myid, mysubid, color);
```

</div>

<div class=column>
```
I am rank 2 in MPI_COMM_WORLD, but 1 in Comm 1.
I am rank 7 in MPI_COMM_WORLD, but 3 in Comm 2.
I am rank 0 in MPI_COMM_WORLD, but 0 in Comm 1.
I am rank 4 in MPI_COMM_WORLD, but 2 in Comm 1.
I am rank 6 in MPI_COMM_WORLD, but 3 in Comm 1.
I am rank 3 in MPI_COMM_WORLD, but 1 in Comm 2.
I am rank 5 in MPI_COMM_WORLD, but 2 in Comm 2.
I am rank 1 in MPI_COMM_WORLD, but 0 in Comm 2.
```

![](images/communicator.svg){.center width=80%}

</div>

# Using an own communicator

![](images/colorcomm1.svg){.center width=30%}

```c
if (myid%2 == 0) {
  color = 1;
} else {
  color = 2;
}
MPI_Comm_split(MPI_COMM_WORLD, color, myid, &subcomm);
MPI_Comm_rank(subcomm, &mysubid);
MPI_Bcast(sendbuf, 8, MPI_INT, 0, subcomm);

```

![](images/colorcomm2.svg){.center width=30%}

# Communicator manipulation

* **MPI_Comm_size** 	
	- Returns number of processes in communicator's group  
* **MPI_Comm_rank** 			
	- Returns rank of calling process in communicator's group  
* **MPI_Comm_compare** 
	- Compares two communicators  
* **MPI_Comm_dup** 	
	- Duplicates a communicator  
* **MPI_Comm_free** 			
	- Marks a communicator for deallocation  


# Summary 

* Defining new communicators usually required in real-world programs
	- Task parallelism, using libraries, I/O,...
* We introduced one way of creating new communicators via `MPI_Comm_split`
	- Tasks assigned with a color, which can be `MPI_UNDEFINED` if the task is excluded in all resulting communicators
	- Other ways (via MPI groups) exist
