#include <omp.h>  //openmp library for parallel processing
#include <stdlib.h> // Standard library for memory allocation and random number generation.
#include <array>
#include <chrono> // for working with time, clocks and timers
#include <functional> // predefined functions for working with functions in a more flexible and generic way
#include <iostream>
#include <string>
#include <vector>

using std::chrono::duration_cast; // converting between different durations (eg. milisecs to secs)
using std::chrono::high_resolution_clock; // used for high resolution timing measurements
using std::chrono::milliseconds; // representing a period of time in milliseconds.
using namespace std;

void s_bubble(int *, int); // sequential buble-sort function declaration
void p_bubble(int *, int); // parallel buble-sort function declaration
void swap(int &, int &);

// sequential bubble-sort
void s_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// parallel bubble-sort
void p_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
#pragma omp parallel for shared(a, first) num_threads(16)
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// swapping two numbers
void swap(int &a, int &b) {
    int test;
    test = a;
    a = b;
    b = test;
}

// measures the execution time of a given function using traverse_fn
std::string bench_traverse(std::function<void()> traverse_fn) {
    auto start = high_resolution_clock::now();
    traverse_fn();
    auto stop = high_resolution_clock::now();

    // Subtract stop and start timepoints and cast it to required unit.
    // Predefined units are nanoseconds, microseconds, milliseconds, seconds,
    // minutes, hours. Use duration_cast() function.
    auto duration = duration_cast<milliseconds>(stop - start);

    // To get the value of duration use the count() member function on the
    // duration object
    return std::to_string(duration.count());
}

// argc: argument count(represents the number of command-line arguments passed to the program)
// **argv: representing the command-line arguments themselves.
int main(int argc, const char **argv) {
    // checks if the number of command-line arguments is less than 3. 
    // If so, it prints a message asking the user to specify the array length and maximum random value 
    // and then exits the program with a return code of 1
    if (argc < 3) {
        std::cout << "Specify array length and maximum random value\n";
        return 1;
    }
    int *a, n, rand_max;

    n = stoi(argv[1]); // first command line arguument (array length)
    rand_max = stoi(argv[2]); // second command line argument (maximum random value)
    a = new int[n]; // dynamic array of size n

    //  fills the array a with random integer values between 0 and rand_max.
    for (int i = 0; i < n; i++) {
        a[i] = rand() % rand_max;
    }

    // Another array b is dynamically allocated and initialized with the same contents as array a.
    int *b = new int[n];
    copy(a, a + n, b);

    // Displaying message
    cout << "Generated random array of length " << n << " with elements between 0 to " << rand_max
         << "\n\n";


    // performs a sequential bubble sort on array a using the s_bubble function and measures the time it takes using bench_traverse.
    std::cout << "Sequential Bubble sort: " << bench_traverse([&] { s_bubble(a, n); }) << "ms\n";
    // printing sorted array
    cout << "Sorted array is =>\n";
    for (int i = 0; i < n; i++) {
        cout << a[i] << ", ";
    }
    cout << "\n\n";

    // sets the number of threads to 16 for OpenMP parallelization
    omp_set_num_threads(16);

    // performs a parallel bubble sort on array b using the p_bubble function, measuring the time with bench_traverse
    std::cout << "Parallel (16) Bubble sort: " << bench_traverse([&] { p_bubble(b, n); }) << "ms\n";
    // printing sorted array
    cout << "Sorted array is =>\n";
    for (int i = 0; i < n; i++) {
        cout << b[i] << ", ";
    }
    return 0;
}

/*

OUTPUT:
Generated random array of length 100 with elements between 0 to 200

Sequential Bubble sort: 0ms
Sorted array is =>
2, 3, 8, 11, 11, 12, 13, 14, 21, 21, 22, 26, 26, 27, 29, 29, 34, 42, 43, 46, 49, 51, 56, 57, 58, 59,
60, 62, 62, 67, 69, 73, 76, 76, 81, 84, 86, 87, 90, 91, 92, 94, 95, 105, 105, 113, 115, 115, 119,
123, 124, 124, 125, 126, 126, 127, 129, 129, 130, 132, 135, 135, 136, 136, 137, 139, 139, 140, 145,
150, 154, 156, 162, 163, 164, 167, 167, 167, 168, 168, 170, 170, 172, 173, 177, 178, 180, 182, 182,
183, 184, 184, 186, 186, 188, 193, 193, 196, 198, 199,

Parallel (16) Bubble sort: 1ms
Sorted array is =>
2, 3, 8, 11, 11, 12, 13, 14, 21, 21, 22, 26, 26, 27, 29, 29, 34, 42, 43, 46, 49, 51, 56, 57, 58, 59,
60, 62, 62, 67, 69, 73, 76, 76, 81, 84, 86, 87, 90, 91, 92, 94, 95, 105, 105, 113, 115, 115, 119,
123, 124, 124, 125, 126, 126, 127, 129, 129, 130, 132, 135, 135, 136, 136, 137, 139, 139, 140, 145,
150, 154, 156, 162, 163, 164, 167, 167, 167, 168, 168, 170, 170, 172, 173, 177, 178, 180, 182, 182,
183, 184, 184, 186, 186, 188, 193, 193, 196, 198, 199,


OUTPUT:

Generated random array of length 100000 with elements between 0 to 100000

Sequential Bubble sort: 16878ms
Parallel (16) Bubble sort: 2914ms

*/

// Compiling & Running program:
//  g++ -fopenmp program.cpp -o program 
// ./program