#include <cstdio>
#include <vector>

#include <omp.h>

/* Code for demonstrating OpenMP task dependencies.
* We run the following 3 functions as OpenMP tasks:
*    - initialize(): Sets array size and values
*    - modify(): Modifies values in the array
*    - sum(): Calculates the sum of array elements.
* Without dependencies or explicit synchronization, the order of task
* execution is not fixed. The main function demonstrates this by running the
* tasks first without dependencies, then once more using the `depend` clause.
*/


// Sets the input array contents to { 1, 2 }
void initialize(std::vector<int>& data)
{
    data.resize(2);
    data[0] = 1;
    data[1] = 2;

    printf("Finished initialize()\n");
}

// Increments each array element by one
void modify(std::vector<int>& data)
{
    for (int i = 0; i < data.size(); i++)
    {
        data[i] = data[i] + 1;
    }

    printf("Finished operate()\n");
}

// Calculates sum of the input array elements
void sum(const std::vector<int>& data)
{
    int result = 0;
    for (int i = 0; i < data.size(); i++)
    {
        result += data[i];
    }

    printf("Finished sum(), result = %d\n", result);
}

int main()
{
    // Create an empty array of integers
    std::vector<int> data;

    printf("\n### Running tasks without dependencies ###\n\n");

    #pragma omp parallel
    #pragma omp single
    {
        // The following has a data race: order of task execution not fixed

        #pragma omp task
        initialize(data);

        #pragma omp task
        modify(data);

        #pragma omp task
        sum(data);
    }

    printf("\n### Repeat with task dependencies ###\n\n");
    data.resize(0);

    #pragma omp parallel
    #pragma omp single
    {
        // Variable `data` is modified by this task, but the input value does not matter => specify `out` dependency.
        #pragma omp task depend(out: data)
        initialize(data);

        // `data` is both read and modified by this task => Should depend on all previously defined out/inout tasks
        #pragma omp task depend(inout: data)
        modify(data);

        // `data` is read but not modified => Depends on all previously defined out/inout tasks
        #pragma omp task depend(in: data)
        sum(data);
    }

    return 0;
}
