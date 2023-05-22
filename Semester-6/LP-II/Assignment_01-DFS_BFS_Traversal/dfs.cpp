#include<iostream>
#include<unordered_map>
#include<vector>
#include<set>
#include<queue>

using namespace std;

void findAdjList(unordered_map<int, set<int>>&adjList, vector<pair<int, int>> edges)
{
    // traverse all the edges
    for(int i=0; i<edges.size(); i++)
    {
        // find u and v
        int u = edges[i].first;
        int v = edges[i].second;

        // as the graph is undiredted
        adjList[u].insert(v);
        adjList[v].insert(u);
    }
}

void find_dfs(unordered_map<int, set<int>>&adjList, unordered_map<int, bool>&visited, vector<int>&ans, int node)
{
    // Step 1: Push node into the answer
    ans.push_back(node);

    // Step 2: Make that node visited
    visited[node] = true;

    // Step 3: Look for its adjacent node
    for(auto i: adjList[node])
    {
        if(!visited[i])
        {
            find_dfs(adjList, visited, ans, i);
        }
    }
}

vector<int> DFS(int vertex, vector<pair<int, int>> edges)
{
    // for the adjacency list
    unordered_map<int, set<int>>adjList;

    // for the final ans 
    vector<int>ans;

    // for the visited DS
    unordered_map<int, bool>visited;

    // Step 1: Find the Adjacency List 
    findAdjList(adjList, edges);

    // Step 2: Traverse all the vertices (this is because the graph can be disconnected)
    for(int i=0; i<vertex; i++)
    {
        if(!visited[i])
        {
            // Step 3: call the bfs function
            find_dfs(adjList, visited, ans, i);
        }
    }
    return ans;
}

int main()
{
    int v = 0; // represents number of vertices
    int e = 0; // represents number of edges

    cout<<"\n Enter total number of vertices: ";
    cin>>v;

    cout<<"\n Enter total number of edges: ";
    cin>>e;

    vector<pair<int, int>> edges; // for storing the edges
    cout<<"\n Enter all the edges";
    cout<<"\n";
    for(int i=0; i<e; i++)
    {
        int a;
        int b;

        cout<<"\n From: ";
        cin>>a;
        cout<<"\n To: ";
        cin>>b;

        edges.push_back(make_pair(a, b));
    }

    vector<int>finalAns; // for the final answer
    finalAns = DFS(v, edges);

    cout<<"\n \n Displaying DFS traversal of the graph";
    cout<<"\n";

    for(int i=0; i<v; i++)
    {
        cout<<" -> "<<finalAns[i];
    }
}