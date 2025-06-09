#include <vector>
#include <cstdio>
#include <string>

#include <mpi.h>


// How many integers to write, total from all MPI processes
static constexpr size_t numElements = 32;

/* Enables or disables debug printing of file contents. Printing is not practical for large files,
so we enable/disable this based on 'numElements'. */
static constexpr bool doDebugPrint = (numElements <= 100);


// Debugging helper, prints out file contents. You don't have to modify this
void debug_read_file(const char* filename);

void single_writer(const std::vector<int>& localData, const char* filename) {

    // Get MPI rank and world size
    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // We assume that each rank has the same amount of data
    const size_t numElementsPerRank = localData.size();

    // "Spokesperson strategy": Send all data to rank == 0 and write it from there.
    // Rank 0 has to allocate a receive/gather buffer to hold the full data.
    // Note that the receive buffer for MPI_Gather can be unallocated in other processes.
    // This saves memory and is OK for MPI

    const size_t totalNumElements = ntasks * numElementsPerRank;

    std::vector<int> receiveBuffer;
    if (rank == 0) {
        receiveBuffer.resize(totalNumElements);
    }

    // Gather data to rank 0, each rank sends 'numElementsPerRank' integers.
    // Note that MPI_Gather automatically orders the data by sending rank
    MPI_Gather(localData.data(), numElementsPerRank, MPI_INT,
        receiveBuffer.data(), numElementsPerRank, MPI_INT, 0, MPI_COMM_WORLD
    );

    // Standard C-style write from rank 0
    if (rank == 0) {

        FILE* fileptr = fopen(filename, "wb");

        if (fileptr == NULL) {
            // Failed to open file for whatever reason
            fprintf(stderr, "Error opening file [%s]\n", filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fwrite(receiveBuffer.data(), sizeof(int), receiveBuffer.size(), fileptr);
        fclose(fileptr);
    }
}

void collective_write(const std::vector<int>& localData, const char* filename) {
    // We assume that each rank has the same amount of data
    const size_t numElementsPerRank = localData.size();

    // Get MPI rank of this process, used to calculate write offset
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Offset is always calculated in bytes
    MPI_Offset offset = static_cast<MPI_Offset>(rank * numElementsPerRank * sizeof(int));

    MPI_File_write_at_all(file, offset, localData.data(), numElementsPerRank, MPI_INT, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
}

int main(int argc, char** argv) {

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

    // Repeat time measurements this many times
    constexpr int repeatCount = 5;

    for (int i = 0; i < repeatCount; i++) {

        // Start time measurement
        double startTime = MPI_Wtime();

        single_writer(localData, filename.c_str());

        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;

        if (rank == 0) {
            printf("i = %d : Time taken for 'single_writer': %g seconds\n", i, elapsedTime);
        }
    }

    if (rank == 0 && doDebugPrint) {
        printf("[%s] file contents:\n", filename.c_str());
        debug_read_file(filename.c_str());
    }

    // ########## Collective write

    filename = "collective_write.dat";

    for (int i = 0; i < repeatCount; i++) {

        // Start time measurement
        double startTime = MPI_Wtime();

        collective_write(localData, filename.c_str());

        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;

        if (rank == 0) {
            printf("i = %d : Time taken for 'collective_write': %g seconds\n", i, elapsedTime);
        }
    }

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
        FILE* fileptr = fopen(filename, "rb");

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
