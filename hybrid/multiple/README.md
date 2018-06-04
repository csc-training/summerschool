## Multiple thread communication

Write a simple hybrid program where each OpenMP thread communicates using
MPI. Implement a test case where the threads of task 0 send their thread id to
corresponding threads in other tasks (see the picture). Remember to employ the
**MPI_THREAD_MULTIPLE** thread support mode (on Sisu it needs to be enabled by
setting the environment variable `MPICH_MAX_THREAD_SAFETY=multiple`).

![img](communication-pattern.png)
