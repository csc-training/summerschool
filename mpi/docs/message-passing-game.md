---
title:  Message-Passing Game
event:  CSC Summer School in High-Performance Computing 2023
lang:   en
---

# Message-passing game {.section}

# One-dimensional acyclic chain {.section}

# One-dimensional acyclic chain

<pre>
...
my_name = input()
left_name = "nobody"










</pre>

# One-dimensional acyclic chain

<pre>
...
my_name = input()
left_name = "nobody"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">send</span>(my_name, right)






</pre>

# One-dimensional acyclic chain

<pre>
...
my_name = input()
left_name = "nobody"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span style="color:var(--csc-blue)">recv</span>(left_name, left)


</pre>


# One-dimensional acyclic chain

<pre>
...
my_name = input()
left_name = "nobody"

if rank < (size - 1):
  right = rank + 1
  <span style="color:var(--csc-blue)">send</span>(my_name, right)

if rank > 0:
  left = rank - 1
  <span style="color:var(--csc-blue)">recv</span>(left_name, left)

print("My left neighbour is", left_name)
</pre>


# One-dimensional cyclic chain {.section}

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1










</pre>

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1





</pre>

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-blue)">send</span>(my_name, right)



</pre>

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-orange)">send</span>(my_name, right)  <span style="color:var(--csc-orange)">! deadlock</span>



</pre>

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-orange)">send</span>(my_name, right)  <span style="color:var(--csc-orange)">! deadlock</span>
<span style="color:var(--csc-blue)">recv</span>(left_name, left)


</pre>

# One-dimensional cyclic chain

<pre>
...
if rank == (size - 1):
  right = 0
else:
  right = rank + 1

if rank == 0:
  left = size - 1
else:
  left = rank - 1

<span style="color:var(--csc-orange)">send</span>(my_name, right)  <span style="color:var(--csc-orange)">! deadlock</span>
<span style="color:var(--csc-blue)">recv</span>(left_name, left)

print("My left neighbour is", left_name)
</pre>


# One-dimensional cyclic chain without deadlock

<pre>
...
if (rank % 2 == 0):
  <span style="color:var(--csc-blue)">send</span>(my_name, right)
  <span style="color:var(--csc-blue)">recv</span>(left_name, left)
else:
  <span style="color:var(--csc-blue)">recv</span>(left_name, left)
  <span style="color:var(--csc-blue)">send</span>(my_name, right)

print("My left neighbour is", left_name)
</pre>

