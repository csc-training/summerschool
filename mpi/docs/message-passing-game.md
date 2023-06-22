---
title:  Message-Passing Game
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# Message-passing game {.section}

# Message-passing game

* Illustrate MPI execution and data model
* Four volunteers: processes
* Lecturer: MPI runtime

# One-dimensional acyclic chain {.section}

# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()

my_name = input()
left_name = "unknown"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)

</pre>

# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()
<div style="color:lightgray">
my_name = input()
left_name = "unknown"

if rank < (size - 1):
  right = rank + 1
  <span>mpi_send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>



# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()

my_name = input()
left_name = "unknown"
<div style="color:lightgray">
if rank < (size - 1):
  right = rank + 1
  <span>mpi_send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>


# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()

my_name = input()
left_name = "unknown"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)
<div style="color:lightgray">
if rank > 0:
  left = rank - 1
  <span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>



# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()

my_name = input()
left_name = "unknown"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)
<div style="color:lightgray">
print("My left neighbour is", left_name)
</div>
</pre>


# One-dimensional acyclic chain

<pre style="color:black; padding:1ex">
size = <span style="color:var(--csc-blue)">mpi_get_size</span>()
rank = <span style="color:var(--csc-blue)">mpi_get_rank</span>()

my_name = input()
left_name = "unknown"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)

</pre>


# One-dimensional cyclic chain {.section}

# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)
<span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)

</pre>

# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...
<div style="color:lightgray">
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span>mpi_send</span>(my_name, right)
<span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>


# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1
<div style="color:lightgray">
if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span>mpi_send</span>(my_name, right)
<span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>

# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1
<div style="color:lightgray">
<span>mpi_send</span>(my_name, right)
<span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>

# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)
<div style="color:lightgray"><span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>




# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-orange)">mpi_send</span>(my_name, right)  <span style="color:var(--csc-orange)">! deadlock</span>
<div style="color:lightgray"><span>mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)
</div>
</pre>

# One-dimensional cyclic chain

<pre style="color:black; padding:1ex">
...

if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-orange)">mpi_send</span>(my_name, right)  <span style="color:var(--csc-orange)">! deadlock</span>
<span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)

print("My left neighbour is", left_name)

</pre>


# One-dimensional cyclic chain without deadlock

<pre style="color:black; padding:1ex">
...

if (rank % 2 == 0):
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)
  <span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)
else:
  <span style="color:var(--csc-blue)">mpi_recv</span>(left_name, left)
  <span style="color:var(--csc-blue)">mpi_send</span>(my_name, right)

print("My left neighbour is", left_name)

</pre>

