#include<bits/stdc++.h> 
#include<iostream>
#include<unordered_map>
#include<list>
#include<set>
#include<vector>
using namespace std;

void createAdj(unordered_map<int, list<pair<int, int>>>&adj, vector<vector<int>>&vec, int edges)
{
    for(int i=0; i<edges; i++)
    {
        int u = vec[i][0];
        int v = vec[i][1];
        int weight = vec[i][2];

        adj[u].push_back(make_pair(v, weight));
        adj[v].push_back(make_pair(u, weight));
    }
}

vector<int> dijkstra(vector<vector<int>> &vec, int vertices, int edges, int source) {
    // adjacency list on basis (node, distance)
    unordered_map<int, list<pair<int, int>>>adj;
    createAdj(adj, vec, edges);

    // creating distance array
    vector<int>dist(vertices);
    for(int i=0; i<vertices; i++)
    {
        dist[i] = INT_MAX;
    }

    // Creating of set on basis (distance, node)
    set<pair<int, int>>st;

    // initailizations
    dist[source] = 0;
    st.insert(make_pair(0, source));

    while(!st.empty())
    {
        // fetch top of the set
        auto top = *(st.begin());
        int nodeDistance = top.first;
        int topNode = top.second;

        // now, remove the top record
        st.erase(st.begin());

        // traverse on neighbours of node
        for(auto neighbour: adj[topNode])
        {
            if(nodeDistance + neighbour.second < dist[neighbour.first])
            {
                // finding if a record is not already present in the set
                auto record = st.find(make_pair(dist[neighbour.first], neighbour.first));
                // if record found, then erase it
                if(record != st.end())
                {
                    // record found
                    st.erase(record);
                }
                
                // update the distance
                dist[neighbour.first] = nodeDistance +  neighbour.second;
                
                // push record in the set
                st.insert(make_pair(dist[neighbour.first], neighbour.first));
            }
        }
    }
    return dist;
}

int main()
{
    int v; // vertices
    cout<<"\n Enter number of vertices: ";
    cin>>v;

    int e; // edges
    cout<<"\n Enter number of edges: ";
    cin>>e;

    vector<vector<int>>vec(e, vector<int>(3));

    for(int i=0; i<e; i++) {
        int a;
        cout<<"\n From: ";
        cin>>a;

        int b;
        cout<<"\n To: ";
        cin>>b;

        int c;
        cout<<"\n Value: ";
        cin>>c;

        vec[i][0] = a;
        vec[i][1] = b;
        vec[i][2] = c;            
    }

    int source = 0;
    cout<<"\n Enter Source: ";
    cin>>source;

    // for the final answer 
    vector<int>finalAns(v);

    finalAns = dijkstra(vec, v, e, source);

    cout<<"\n\n Displaying the Shortest path from source to all other vertices";
    cout<<"\n";
    for(int i=0; i<v; i++)
    {
        cout<<"  "<<finalAns[i];
    }
}

// https://www.codingninjas.com/codestudio/problems/dijkstra-s-shortest-path_920469?leftPanelTab=1&utm_source=youtube&utm_medium=affiliate&utm_campaign=Lovebabbar