// Class for basic parallelization information
namespace heat {
struct ParallelData {
    int size;            // Number of MPI tasks
    int rank;
    int nup, ndown;      // Ranks of neighbouring MPI tasks

    ParallelData();      // Constructor
};
} // namespace heat
