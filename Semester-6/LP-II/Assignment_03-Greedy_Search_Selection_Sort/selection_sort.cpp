#include<iostream>
#include<vector>
using namespace std;

void selectionSort(vector<int>& arr, int n)
{   
    for(int i = 0; i < n-1; i++ ) {
        int minIndex = i;
        
        for(int j = i+1; j<n; j++) {
            
            if(arr[j] < arr[minIndex]) 
                minIndex = j;
            
        }
        swap(arr[minIndex], arr[i]);
    }
}

int main()
{
    int n;
    cout<<"\n Enter the size of the array: ";
    cin>>n;

    vector<int>arr;

    cout<<"\n Enter the unsorted array \n";
    for(int i=0; i<n; i++)
    {
        int temp;
        cout<<"\n Enter number: ";
        cin>>temp;
        arr.push_back(temp);
    }

    selectionSort(arr, n);

    cout<<"\n \n Displaying the sorted array \n";
    for(int i=0; i<n; i++)
    {
        cout<<arr[i]<< " ";
    }
}