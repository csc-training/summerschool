## Multiple thread communication

Write a simple hybrid program where each OpenMP thread communicates using
MPI.

Implement a test case where the threads of task 0 send their thread ID to
the corresponding threads in other tasks (see the picture). Remember to employ
the **MPI_THREAD_MULTIPLE** thread support mode.

![img](communication-pattern.png)
