#include<iostream>
#include<stdlib.h>
#include <omp.h>
#include<queue>
using namespace std;

// node structure of the tree
class node 
{
    public:
        node* left; // left pointer
        node* right; // right pointer
        int data; // holds data
};

class BreadthFs
{
    public: 
        node* insert(node*, int); // member function declaration
        void bfs(node*); // member function declaration
};

// function for inserting a node in the tree
node* insert(node* root, int data) {
    if(!root) {
        root = new node;
        root->left = NULL;
        root->right = NULL;
        root->data = data;
        return root;
    }

    queue<node*>q;
    q.push(root);

    while(!q.empty())
    {
        node* temp = q.front();
        q.pop();

        if(temp->left == NULL)
        {
            temp->left = new node;
            temp->left->left = NULL;
            temp->left->right = NULL;
            temp->left->data = data;
            return root;
        }
        else 
        {
            q.push(temp->left);
        }

        if(temp->right == NULL) 
        {
            temp->right = new node;
            temp->right->left = NULL;
            temp->right->right = NULL;
            temp->right->data = data;
            return root;
        } 
        else
        {
            q.push(temp->right);
        }
    }
    return root;
}

// function for performing bfs on the tree/graph
void bfs(node* head) {
    queue<node*>q;
    q.push(head);

    int qSize;

    while(!q.empty())
    {
        qSize = q.size();
        #pragma omp parallel for
        for (int 
        i=0; i<qSize; i++)
        {
            node* currNode;
            #pragma omp critical
            {
                currNode = q.front();
                q.pop();
                cout<<"\t"<<currNode->data;
            }
            #pragma omp critical
            if(currNode->left)
            {
                q.push(currNode->left);
            }
            if(currNode->right)
            {
                q.push(currNode->right);
            }
        }
    }
}

int main() {
    node* root = NULL;
    int data;
    char ans;

    cout<<endl;
    cout<<"<=== Building the Tree ===>";
    cout<<endl;
    do {
        cout<<"\n Enter data => ";
        cin>>data;

        root = insert(root, data);

        cout<<"\n Do you want to insert one more node (y/n)? => ";
        cin>>ans;
    } while(ans == 'y' || ans == 'Y');

    cout<<endl;
    cout<<"<=== Displaying BFS traversal of the tree using Parallel Programming ===> ";
    cout<<endl;
    bfs(root);
    return 0;
}