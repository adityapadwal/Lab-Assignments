#include<iostream>
#include<unordered_map>
#include<vector>
#include<list>
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

void find_bfs(unordered_map<int, set<int>>adjList, unordered_map<int, bool>&visited, vector<int>&ans, int node)
{
    // create a queue
    queue<int>q;
    // enter starting node inside the queue
    q.push(node);
    // mark visited of the starting node as true
    visited[node] = true;

    while(!q.empty())
    {
        // get frontNode
        int frontNode = q.front();
        // pop out the frontNode from the queue
        q.pop();
        // mark visited of frontNode as true
        visited[frontNode] = true;
        // enter frontNode into the ans
        ans.push_back(frontNode);

        // check frontNode's adjacent elements 
        for(auto i: adjList[frontNode])
        {
            if(!visited[i])
            {
                q.push(i);
                visited[i] = true;
            }
        }
    }
}

vector<int> BFS(int vertex, vector<pair<int, int>> edges)
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
            find_bfs(adjList, visited, ans, i);
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
    finalAns = BFS(v, edges);

    cout<<"\n \n Displaying BFS traversal of the graph";
    cout<<"\n";

    for(int i=0; i<v; i++)
    {
        cout<<" -> "<<finalAns[i];
    }
}