#include<iostream>

using namespace std;

int partition(int * arr, int p, int r){
	int x = arr[r];
    cout<<x<<" ";
	int i = p - 1;
	for(int j = p; j < r; j++){
		if(arr[j] <= x){
			i++;
			swap(arr[i],arr[j]);
		}
	}
	swap(arr[i + 1], arr[r]);
	return i + 1;
}

void quickSort(int * arr, int p, int r){
	if (p < r){
		int q = partition(arr, p, r);
		quickSort(arr, p, q - 1);
		quickSort(arr, q + 1, r);
	}
}

//Randomized
int randomPartition(int * arr, int p, int r){
	int randomIndex = rand() % ((r - p) + 1) + p;
    cout<<randomIndex<<" ";
    swap(arr[randomIndex], arr[r]);
    
    int x = arr[r];
	int i = p-1;
	for(int j = p; j < r; j++){
		if(arr[j] <= x){
			i++;
			swap(arr[i],arr[j]);
		}
	}
	swap(arr[i + 1], arr[r]);
	return i + 1;
}

void randomizedQuickSort(int * arr, int p, int r){
	if (p < r){
		int q = randomPartition(arr, p, r);
		randomizedQuickSort(arr, p, q - 1);
		randomizedQuickSort(arr, q + 1, r);
	}
}

int main(){
	int A[] = {2, 1, 3, 4, 5, 6, 7, 8, 9, 10};
	int n = sizeof(A)/ sizeof(A[0]);
    quickSort(A, 0, n-1);
	// randomizedQuickSort(A, 0, n - 1);
    cout<<"\n Printing sorted output"<<endl;
	for(int i : A){
		cout << i << " ";
	}
	cout << '\n';
	return 0;
}