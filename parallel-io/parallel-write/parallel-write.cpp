#include <vector>
#include <cstdio>
#include <string>

#include <mpi.h>


// How many integers to write, total from all MPI processes
static constexpr size_t numElements = 32;

// Enables or disables debug printing of file contents. Set to false if numElements is very large (>> 100)
static constexpr bool doDebugPrint = true;


// Debugging helper, prints out file contents. You don't have to modify this
void debug_read_file(const char* filename);

void single_writer(const std::vector<int>& localData, const char* filename) {
    // Gets called from all MPI ranks. 'localData' contains different data on each rank.
    // TODO: Gather contents of 'localData' to one MPI process and write it all to file 'filename' ("spokesperson" strategy).
    // The output should be ordered such that data from rank 0 comes first, then rank 1, and so on

    // You can assume that 'localData' has same length in all MPI processes:
    const size_t numElementsPerRank = localData.size();

}

void collective_write(const std::vector<int>& localData, const char* filename) {
    // TODO: Like single_writer(), but implement a parallel write using MPI_File_write_at_all()

}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (numElements % ntasks != 0) {
        if (rank == 0) {
            fprintf(stderr, "numElements (%zu) must be divisible by the number of MPI tasks (%d)!\n", numElements, ntasks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t numElementsPerRank = numElements / ntasks;

    // Create data array, each element initialized to value 'rank'
    std::vector<int> localData(numElementsPerRank, rank);

    // Print some statistics
    if (rank == 0) {
        printf("Writing total of %zu integers, %zu from each rank.\n", numElements, numElementsPerRank);
        const size_t bytes = numElements * sizeof(int);
        printf("Total bytes to write: %zu (%zu MB)\n", bytes, bytes / 1024 / 1024);
        fflush(stdout);
    }


    // ########## "Spokesperson" write
    std::string filename = "single_writer.dat";

    single_writer(localData, filename.c_str());

    if (rank == 0 && doDebugPrint) {
        printf("[%s] file contents:\n", filename.c_str());
        debug_read_file(filename.c_str());
    }

    // ########## Collective write
    filename = "collective_write.dat";

    collective_write(localData, filename.c_str());

    if (rank == 0 && doDebugPrint) {
        printf("[%s] file contents:\n", filename.c_str());
        debug_read_file(filename.c_str());
    }

    //~

    MPI_Finalize();
    return 0;
}


void debug_read_file(const char* filename) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        FILE *fileptr = fopen(filename, "rb");

        if (fileptr != NULL) {

            int value;
            while (fread(&value, sizeof(int), 1, fileptr) == 1) {
                printf("%d", value);
            }

            fclose(fileptr);

            printf("\n");
            fflush(stdout);
        } else {
            fprintf(stderr, "Failed to open file %s for debug printing!\n", filename);
        }
    }
}
