#include <bits/stdc++.h> 
#include<iostream>
using namespace std;

void addSolution(vector<vector<int>>&board, vector<vector<int>> &ans, int n)
{
    vector<int>temp;
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            temp.push_back(board[i][j]);
        }
    }
    ans.push_back(temp);
}

bool isSafe(int row, int col, vector<vector<int>>&board, int n)
{
    int x = row;
    int y = col;
    
    // check for same row 
    while(y >= 0)
    {
        if(board[x][y] == 1)
        {
            return false;
        }
        y--;
    }
    
    x = row;
    y = col;
    
    // check for the diagonal 
    while(x >= 0 && y >= 0)
    {
        if(board[x][y] == 1)
        {
            return false;
        }
        x--;
        y--;
    }
    
    x = row;
    y = col;
    
    while(x < n && y >= 0)
    {
        if(board[x][y] == 1)
        {
            return false;
        }
        x++;
        y--;
    }
    
    return true;
}

void solve(int col, vector<vector<int>>&ans, vector<vector<int>> &board, int n)
{
    // base case 
    if(col == n)
    {
        addSolution(board, ans, n);
        return;
    }
    
    // solve 1 case and the rest will be taken care of by the recursion 
    for(int row = 0; row < n; row++)
    {
        if(isSafe(row, col, board, n))
        {
            board[row][col] = 1;
            solve(col+1, ans, board, n);
    // backtrack (queen ko hatao)
            board[row][col] = 0;
        }
    }
}

vector<vector<int>> nQueens(int n)
{
    // creating the board and initializing it with 0
	vector<vector<int>> board(n, vector<int>(n, 0));
    
    // for the ans 
    vector<vector<int>> ans;
    
    solve(0, ans, board, n);
    
    return ans;
}

int main()
{
    int n;
    cout<<"'\n Enter the value of n: ";
    cin>>n;

    vector<vector<int>> finalAns = nQueens(n);

    cout<<"\n <==== Displaying the possible combinations of output ====>";
    for(int i=0; i<n; i++)
    {
        cout<<"\n";
        vector<int>temp = finalAns[i];

        for(int j=0, x=0; j<n*n; j++)
        {
            cout<<" "<<temp[j];
            x++;
            if(x % n == 0)
            {
                cout<<endl;
            }
        }

        // for(auto j: temp)
        // {
        //     cout<<" "<<j;
        // }

        cout<<"\n";
    }
}