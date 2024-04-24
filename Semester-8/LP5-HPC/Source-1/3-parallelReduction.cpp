#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void min_red(vector<int> &arr)
{
    int min_value = arr[0];
    #pragma omp parallel for reduction(min : min_value)
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] < min_value)
        {
            min_value = arr[i];
        }
    }
    cout << "\n Minimum value=" << min_value << endl;
}

void max_red(vector<int> &arr)
{
    int max_value = arr[0];
#pragma omp parallel for reduction(max : max_value)
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
        }
    }
    cout << "\n Maximum value=" << max_value << endl;
}

void sum_red(vector<int> &arr)
{
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++)
    {
        sum = sum + arr[i];
    }
    cout << "\n Sum=" << sum << endl;
}

void avg_red(vector<int> &arr)
{
    int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++)
    {
        sum = sum + arr[i];
    }
    cout << "\n Average=" << (double)sum / arr.size() << endl;
}

int main()
{
    vector<int> arr = {5, 2, 9, 1, 7, 6, 8, 3, 4};
    min_red(arr);
    max_red(arr);
    sum_red(arr);
    avg_red(arr);
    return 0;
}

// Compiling & Running program:
//  g++ -fopenmp program.cpp -o program 
// ./program