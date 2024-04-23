#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

const int MAX = 100000;

vector<int> graph[MAX];
bool visited[MAX];

void dfs(int node)
{
    stack<int>s;
    s.push(node);
    while (!s.empty())
    {
        int curr_node = s.top();
        if (!visited[curr_node])
        {
            visited[curr_node] = true;
            s.pop();
            cout << curr_node << " ";
            #pragma omp parallel for
            for (int i=0; i<graph[curr_node].size(); i++)
            {
                int adj_node = graph[curr_node][i];
                if (!visited[adj_node])
                {
                    s.push(adj_node);
                }
            }
        }
    }
}
int main()
{
    int n; 
    int m;
    int start_node;
    cout<<"<=== Program for DFS of graph ===>"<<endl;
    cout<<"\n Enter number of nodes => ";
    cin>>n;
    cout<<"\n Enter number of edges => ";
    cin>>m;
    cout<<"\n Enter the starting node of the graph => ";
    cin>>start_node;
    
    cout<<endl;
    cout << "\n Enter pair of node and edges "<<endl;
    for(int i=0; i<m; i++)
    {
        int u, v;
        cout<<"\n Enter pair "<<i+1<<"/"<<m<<" => ";
        cin >> u >> v;

        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    cout<<endl;

    #pragma omp parallel for
    for (int i=0; i<n; i++)
    {
        visited[i] = false;
    }

    cout<<"\n <=== Displaying DFS traversal of the graph ===> "<<endl;
    dfs(start_node);
    return 0;
}