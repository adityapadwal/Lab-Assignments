// Main code

#include <bits/stdc++.h> 
using namespace std;

int solve(vector<int>&weight, vector<int>&value, int index, int capacity)
{
	// base case
	if(index == 0)
	{
		if(weight[0] <= capacity)
		{
			return value[0];
		}
		else
		{
			return 0;
		}
	}

	int include = 0;
	if(weight[index] <= capacity)
	{
		include = value[index] + solve(weight, value, index-1, capacity-weight[index]);
	}

	int exclude = 0;
	exclude = 0 + solve(weight, value, index-1, capacity);

	int ans = max(include, exclude);
	return ans;
}

int solveMem(vector<int>&weight, vector<int>&value, int index, int capacity, vector<vector<int>>&dp)
{
	// base case
	if(index == 0)
	{
		if(weight[0] <= capacity)
		{
			return value[0];
		}
		else
		{
			return 0;
		}
	}
	if(dp[index][capacity] != -1)
	{
		return dp[index][capacity];
	}

	int include = 0;
	if(weight[index] <= capacity)
	{
		include = value[index] + solveMem(weight, value, index-1, capacity-weight[index], dp);
	}

	int exclude = 0;
	exclude = 0 + solveMem(weight, value, index-1, capacity, dp);

	dp[index][capacity] = max(include, exclude);
	return dp[index][capacity];
}

int solveTab(vector<int> weight, vector<int> value, int n, int capacity)
{
	// Step 1: DP array
	vector<vector<int>>dp(n, vector<int>(capacity+1, 0));

	// Step 2: Analyse base case
	for(int w=weight[0]; w<=capacity; w++)
	{
		if(weight[0] <= capacity)
		{
			dp[0][w] = value[0];
		}
		else
		{
			dp[0][w] = 0;
		}
	}

	// Step 3: Take care of remaining recursive calls
	for(int index=1; index<n; index++)
	{
		for(int w=0; w<=capacity; w++)
		{
			int include = 0;
			if(weight[index] <= w)
			{
				include = value[index] + dp[index-1][w-weight[index]];
			}

			int exclude = 0;
			exclude = 0 + dp[index-1][w];

			dp[index][w] = max(include, exclude);
		}
	}

	return dp[n-1][capacity];
}

int knapsack(vector<int> weight, vector<int> value, int n, int maxWeight) 
{
	// Recursion
	// int finalAns;
	// finalAns = solve(weight, value, n-1, maxWeight);
	// return finalAns;

	// Memoization
	// int finalAns;
	// vector<vector<int>>dp(n, vector<int>(maxWeight+1, -1));
	// finalAns = solveMem(weight, value, n-1, maxWeight, dp);
	// return finalAns;

	// Tabulation
	int finalAns;
	finalAns = solveTab(weight, value, n, maxWeight);
	return finalAns;
}

int main() {

    // ye khud likh lo bhai
    return 0;
}

// Alternative code
#include<iostream>

using namespace std;

int main(){
    int capacity = 10;
    int items = 4;
    int price[items + 1] = {0, 3, 7, 2, 9};
    int wt[items + 1] = {0, 2, 2, 4, 5};
    int dp[items + 1][capacity + 1];
    
    for(int i = 0; i <= items; i++){
        for(int j = 0; j <= capacity; j++){
            if(i == 0 || j == 0){
                //There's nothing to add to Knapsack
                dp[i][j] = 0;
            }
            else if(wt[i] <= j){
                //Choose previously maximum or value of current item + value of remaining weight
                dp[i][j] = max(dp[i - 1][j], price[i] + dp[i - 1][j - wt[i]]);
            }
            else{
                //Add previously added item to knapsack
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
	

    cout << "Maximum Profit Earned: " << dp[items][capacity] << "\n";
    return 0;
}

/*
0/1 Knapsack :
Time Complexity: O(N*W). 
where ‘N’ is the number of weight element and ‘W’ is capacity. As for every weight element we traverse through all weight capacities 1<=w<=W.
Auxiliary Space: O(N*W). 
The use of 2-D array of size ‘N*W’.
*/