'''
############################################ Graph ###########################

785. Is Graph Bipartite?
207. Course Schedule
210. Course Schedule II
684. Redundant Connection

'''

'''
785. Is Graph Bipartite?
Medium

Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j for which the edge between nodes i and j exists.  Each node is an integer between 0 and graph.length - 1.  There are no self edges or parallel edges: graph[i] does not contain i, and it doesn't contain any element twice.

Example 1:
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true
Explanation: 
The graph looks like this:
0----1
|    |
|    |
3----2
We can divide the vertices into two groups: {0, 2} and {1, 3}.
Example 2:
Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
Output: false
Explanation: 
The graph looks like this:
0----1
| \  |
|  \ |
3----2
We cannot find a way to divide the set of nodes into two independent subsets.
 

Note:

graph will have length in range [1, 100].
graph[i] will contain integers in range [0, graph.length - 1].
graph[i] will not contain i or duplicate values.
The graph is undirected: if any element j is in graph[i], then i will be in graph[j].
'''

# Color the nodes. For one red node, all its connected nodes should be blue, and for a blue node,
#   all connected nodes should be red.

# BFS won't work, because it traverses the graph layer by layer, and if we reach a node where it
#   is not connected to other nodes, we will not be able to traverse the rest of node. For ex, 
#   if node 0 and 1 are connected to eachother, and node 2, 3, 4 are all connected to each other,
#   the method would return true as it only looked at first 2 nodes. However, it should return 
#   false as the second part of the graph is not bipartite. 

# Use DFS (Stack) to traverse through the graph

# Use a dictionary, or list, to stored the color of visited nodes
# tuple in the queue: (node, shouldBeRed)

# Time: O(N + E) where N is num of nodes, E is num of edges
# Space: O(N) the space we use to store the color
class Solution:
    def isBipartite(self, graph: [[int]]) -> bool:
        
        # color stores an int for each node, 1 if it's red, 2 if it's blue. None for unvisited ones
        color = [None] * len(graph)
        # Use stack to do DFS in Iteration way
        stack = []
        
        # iterative through all not-visited nodes in graph 
        for i in range(len(graph)):
            if not color[i]:
                # Make first node in each node group red by default (or false, it doesn't matter)
                color[i] = 1
                stack.append(i)

                # Traverse through all connected nodes
                while stack:
                    curr = stack.pop()
                    currCol = color[curr]
                    for n in graph[curr]:
                        if color[n] == currCol:
                            # Find wrong one, then return False
                            return False
                        if not color[n]:
                            color[n] = 3 - currCol
                            stack.append(n)
        
        return True
                    

'''
207. Course Schedule
Medium

There are a total of numCourses courses you have to take, labeled from 0 to numCourses-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

 
Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
 

Constraints:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
1 <= numCourses <= 10^5
'''

class Solution2:
    # Recursive DFS
    # Time: O(N + E) where N is number of classes, E is number of relations
    # Space: O(N + E)
    def canFinish(self, numCourses: int, prerequisites: [[int]]) -> bool:
        checked = set()
        processing = [False] * numCourses
        
        # We need to pre-process the prerequisites list to make a map that can search
        # prerequisitesMap contains [curr, next], where course 'next' has to be taken after finish
        #   course 'curr'
        prerequisitesMap = [[] for _ in range(numCourses)]
        for p in prerequisites:
            prerequisitesMap[p[1]].append(p[0])
        
        # the dfs traversal to check starting at class c, if course is valid
        def dfs(c, processing):
            result = True
            # Base case, if c is already checked, it has to be valid
            if c in checked:
                return True
            # Base case, if c is still processing, we've reached a loop
            if processing[c]:
                return False
            # add current into processing queue so that if we reach it again in current's children, we
            #   know that we reach a loop
            processing[c] = True
            # Iterate through children
            for child in prerequisitesMap[c]:
                result = dfs(child, processing)
                if not result:
                    break
            # switch processing back to False after we finish the process
            processing[c] = False
            # Add to check so that we don't have to calculate it again
            checked.add(c)
            return result
        
        for i in range(numCourses):
            if i not in checked:
                if not dfs(i, processing):
                    return False
        return True


'''
210. Course Schedule II
Medium

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:

Input: 2, [[1,0]] 
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] .
Example 2:

Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
'''

class Solution3:
    # Time: O(N + E)  where N is number of classes, and E is number of relationships
    # Space: O(N)  which is the size of the queue
    def findOrder(self, numCourses: int, prerequisites: [[int]]) -> [int]:
        ans = []
        
        # We need to pre-process the prerequisites list to make a map
        # We want each course as a key, and all of its next courses as its value
        prerequisitesMap = [[] for _ in range(numCourses)]
        # The depth order of nodes. The courses with no prerequisites has inDegree of 0, and courses 
        #   that can be taken after taking inDegree 0 classes, has inDegree of 1, and so on...
        inDegree = [0] * numCourses
        
        for p in prerequisites:
            prerequisitesMap[p[1]].append(p[0])
            inDegree[p[0]] += 1

        # We will start with the first course with inDegree == 0, and put it in our result, then, for
        #   all classes that are connected to the current class, we reduce its inDegree by 1. Then
        #   we continue with all remaining classes with inDegree == 0
        queue = []
        
        for i in range(numCourses):
            if not inDegree[i]:
                queue.append(i)
        
        # BFS to go through all items in queue, and add new item to queue when it has inDegree == 0
        for c in queue:
            ans.append(c)
            for nextClass in prerequisitesMap[c]:
                inDegree[nextClass] -= 1
                if not inDegree[nextClass]:
                    queue.append(nextClass)
        
        return ans if len(ans) == numCourses else []


'''
684. Redundant Connection
Medium

In this problem, a tree is an undirected graph that is connected and has no cycles.

The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] with u < v, that represents an undirected edge connecting nodes u and v.

Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge [u, v] should be in the same format, with u < v.

Example 1:
Input: [[1,2], [1,3], [2,3]]
Output: [2,3]
Explanation: The given undirected graph will be like this:
  1
 / \
2 - 3
Example 2:
Input: [[1,2], [2,3], [3,4], [1,4], [1,5]]
Output: [1,4]
Explanation: The given undirected graph will be like this:
5 - 1 - 2
    |   |
    4 - 3
Note:
The size of the input 2D-array will be between 3 and 1000.
Every integer represented in the 2D-array will be between 1 and N, where N is the size of the input array.

Update (2017-09-26):
We have overhauled the problem description + test cases and specified clearly the graph is an undirected graph. For the directed graph follow up please see Redundant Connection II). We apologize for any inconvenience caused.
'''

# We will make use of a data structure called Disjoint Set Union (DSU). With DSU, we need to
#   store parents of any node. When check if two nodes are connected, we check their parent
#   value to be the same or not. Two nodes with the same parent value are connected and two
#   nodes with different parent values are not connected.
# We need to implement two methods, find, and union. 
#   - find(n) will return the parent value of node n
#   - union(n1, n2) will union n1 and n2. If the set n1 is in has more members, we update 
#   everything in n2 with parent value of what n1 set has, and vise versa. 

# Time: O(N) See solution for explaination   Space: O(N)
class DSU:
    def __init__(self, size):
        self.parents = list(range(size))
        self.ranks = [0] * size
    
    # return the parent value of node
    def find(self, n):
        if self.parents[n] != n:
            self.parents[n] = self.find(self.parents[n])
        return self.parents[n]
    
    # union two sets
    def union(self, n1, n2):
        # parents of two nodes
        par1 = self.find(n1)
        par2 = self.find(n2)
        # rank is based on node's parent only
        rank1 = self.ranks[par1]
        rank2 = self.ranks[par2]
        # If they are already in one set, return false
        if par1 == par2:
            return False
        else:
            # update the parent of eigher n1 or n2 to the other one's parent based on rank
            if rank1 > rank2:
                self.parents[par2] = par1
            elif rank1 < rank2:
                self.parents[par1] = par2
            else:
                self.parents[par2] = par1
                self.ranks[par1] += 1
            return True
        
class Solution4:
    def findRedundantConnection(self, edges: [[int]]) -> [int]:
        dsu = DSU(len(edges) + 1)
        for e in edges:
            if not dsu.union(e[0], e[1]):
                return e
