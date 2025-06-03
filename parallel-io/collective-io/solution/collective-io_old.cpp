#include <vector>
#include <string>

// C-style headers
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include <mpi.h>

// Debugging helper, prints out file contents
void debug_read_file(const char* filename);

void single_writer(const std::vector<int>& localData, const char* filename) {

    // Get MPI rank and world size
    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // We assume that each rank has the same amount of data
    const size_t numElementsPerRank = localData.size();

    // "Spokesperson strategy": Send all data to rank == 0 and write it from there.
    // Rank 0 needs to allocate a receive/gather buffer to hold the full data

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

        FILE *fileptr = fopen(filename, "wb");

        if (fileptr == NULL) {
            // Failed to open file for whatever reason
            fprintf(stderr, "Error: %d (%s)\n", errno, strerror(errno));
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


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <megabytes> <debug_mode>\n", argv[0]);
            fprintf(stderr, "<megabytes> (integer) : How many megabytes (MB) will be written to disk in total.\n");
            fprintf(stderr, "<debug_mode> (0 or 1) : If 1, will perform a smaller write and print the file contents. Use to check that your parallel IO is working as intended\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse the command line arguments
    size_t megabytes = static_cast<size_t>(std::stoul(argv[1]));
    const bool isDebugMode = static_cast<bool>(std::stoul(argv[2]));

    size_t bytes = megabytes * 1024 * 1024;

    if (bytes % ntasks != 0 && !isDebugMode) {
        if (rank == 0) {
            fprintf(stderr, "Number of bytes to write (%zu) must be divisible by the number of MPI tasks (%d)\n", bytes, ntasks);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    size_t numElements = bytes / sizeof(int);
    size_t numElementsPerRank = numElements / ntasks;


    if (isDebugMode) {
        // For debugging we instead do a random small number of writes to keep the output readable. Get and broadcast a random integer:
        std::srand(static_cast<unsigned int>(MPI_Wtime()));
        int randomInt = std::rand() % 10 + 1;

        MPI_Bcast(&randomInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

        numElementsPerRank = static_cast<size_t>(randomInt);
        numElements = numElementsPerRank * ntasks;

        bytes = numElements * sizeof(int);
        megabytes = bytes / 1024 / 1024;
    }


    // Print some statistics
    if (rank == 0) {

        if (isDebugMode) {
            printf("\n\n#### RUNNING IN DEBUG MODE! 'megabytes' input will be ignored ####\n\n");
        }

        printf("Writing total of %zu integers, %zu from each rank.\n", numElements, numElementsPerRank);
        const size_t bytes = numElements * sizeof(int);
        printf("Total bytes to write: %zu (%zu MB)\n", bytes, megabytes);
        fflush(stdout);
    }

    // Create data array, each element initialized to value 'rank'.
    std::vector<int> data(numElementsPerRank, rank);


    // ########## "Spokesperson" write
    std::string filename = "single_writer.dat";

    // Start time measurement. MPI_Wtime() has no built-in synchronization so we add manual barriers
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    single_writer(data, filename.c_str());

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;

    if (rank == 0) {
        printf("\nTime taken for 'single_writer': %g seconds\n", elapsedTime);

        if (isDebugMode) {
            printf("File contents:\n");
            debug_read_file(filename.c_str());
        }
        // Remove the file to avoid cluttering storage with unused stuff
        remove(filename.c_str());
    }

    // ########## Collective write

    filename = "collective_write.dat";

    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    collective_write(data, filename.c_str());

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    elapsedTime = endTime - startTime;

    if (rank == 0) {
        printf("\nTime taken for 'collective_write': %g seconds\n", elapsedTime);

        if (isDebugMode) {
            printf("File contents:\n");
            debug_read_file(filename.c_str());
        }
        remove(filename.c_str());
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
            while (fread(&value, sizeof(int), 1, fileptr) == 1)
            {
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
