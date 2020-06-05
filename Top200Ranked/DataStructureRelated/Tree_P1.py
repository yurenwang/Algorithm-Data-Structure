'''
##################################### Tree Part I ###############################
From Top (recursion solutions) to the end of Traversing a Tree

104. Maximum Depth of Binary Tree
110. Balanced Binary Tree
543. Diameter of Binary Tree
226. Invert Binary Tree
617. Merge Two Binary Trees
112. Path Sum
437. Path Sum III
572. Subtree of Another Tree
101. Symmetric Tree
111. Minimum Depth of Binary Tree
404. Sum of Left Leaves
687. Longest Univalue Path
337. House Robber III
671. Second Minimum Node In a Binary Tree
637. Average of Levels in Binary Tree
513. Find Bottom Left Tree Value
144. Binary Tree Preorder Traversal
145. Binary Tree Postorder Traversal
94. Binary Tree Inorder Traversal

'''

######################### Recursion ######
'''
104. Maximum Depth of Binary Tree
Easy

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.
'''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Recursion
# Time: O(N)  Space: O(logN)
class Solution:
    def maxDepth(self, root: TreeNode):
        if root is None:
            return 0
        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)
        return max(leftDepth, rightDepth) + 1


'''
110. Balanced Binary Tree
Easy

Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

Example 1:

Given the following tree [3,9,20,null,null,15,7]:

    3
   / \
  9  20
    /  \
   15   7
Return true.

Example 2:

Given the following tree [1,2,2,3,3,null,null,4,4]:

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
Return false.
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# This is the regualr recursion method

# Time: O(NlogN)  Space: O(N)
class Solution2:
    def isBalanced(self, root: TreeNode) -> bool:
        # helper function to calculate the height of a tree
        def height(root):
            if not root:
                return 0
            return 1 + max(height(root.left), height(root.right))
        
        # curr is True if both left and right are true, and height diff is <= 1
        if not root:
            return True
        return self.isBalanced(root.left) and self.isBalanced(root.right) \
            and -1 <= height(root.left) - height(root.right) <= 1

########### Better solution: 
# This method eliminates the extra time and space required by calculating height and isBalanced separately. 

# Time: O(N)  Space: O(logN)
class Solution3:
    def isBalanced(self, root: TreeNode) -> bool:
        # helper function to calculate the height of a tree, and set height to -1 if it is not balanced
        def height(root):
            if not root:
                return 0
            l = height(root.left)
            r = height(root.right)
            # set the height to -1 if current tree is not balanced
            if l == -1 or r == -1 or abs(l - r) > 1:
                return -1
            # return normal height if it is balanced
            return 1 + max(l, r)
        
        # we only need to check the height of root if it is -1 to determine if it is balanced
        return height(root) != -1


'''
543. Diameter of Binary Tree
Easy

Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is represented by the number of edges between them.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# recursion.
# Time: O(N)  Space: O(N)
class Solution4:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # the helper function to calculate the max height from the curr point, as well as the 
        #   max diameter length with the curr point as root
        def helper(root): # returns a tuple of (maxHeight, maxDiameter)
            if not root:
                return 0, 0
            lHeight, lDiameter = helper(root.left)
            rHeight, rDiameter = helper(root.right)
            return max(lHeight, rHeight) + 1, max(lDiameter, rDiameter, (lHeight + rHeight))
        maxHeight, maxDiameter = helper(root)
        return maxDiameter
            

'''
226. Invert Binary Tree
Easy

Invert a binary tree.

Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
Trivia:
This problem was inspired by this original tweet by Max Howell:

Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# This one is really easy
class Solution5:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        root.right, root.left = self.invertTree(root.left), self.invertTree(root.right)
        return root


'''
617. Merge Two Binary Trees
Easy

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:

Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
 

Note: The merging process must start from the root nodes of both trees.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Pretty easy
# Time: O(N)  Space: O(logN)
class Solution6:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1:
            return t2
        if not t2:
            return t1
        l = self.mergeTrees(t1.left, t2.left)
        r = self.mergeTrees(t1.right, t2.right)
        t1.val += t2.val
        t1.left = l
        t1.right = r
        return t1


'''
112. Path Sum
Easy

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

Note: A leaf is a node with no children.

Example:

Given the below binary tree and sum = 22,

      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Recursion
# Time: O(N)  Space: O(logN)
class Solution7:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        sum -= root.val
        # If we've reached a leaf
        if not root.left and not root.right:
            return sum == 0
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)


'''
437. Path Sum III
Easy

You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# This is a DFS question

class Solution8:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        if not root:
            return 0
        # return the sum of solutions starting from this root and starting from other roots
        return self.pathSumFrom(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)
    
    # dfs helper
    def pathSumFrom(self, root: TreeNode, sum: int):
        if not root:
            return 0
        sum -= root.val
        return (1 if sum == 0 else 0) + \
            self.pathSumFrom(root.left, sum) + self.pathSumFrom(root.right, sum)
        

'''
572. Subtree of Another Tree
Easy

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# dfs
# Time: O(M*N)  Space: O(N)
class Solution9:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s and not t:
            return True
        if not s or not t:
            return False
        return self.traverse(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    # dfs helper
    def traverse(self, s: TreeNode, t: TreeNode):
        if not s and not t:
            return True
        if not s or not t:
            return False
        return s.val == t.val and self.traverse(s.left, t.left) and self.traverse(s.right, t.right)
    

'''
101. Symmetric Tree
Easy

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
 

But the following [1,2,2,null,3,null,3] is not:

    1
   / \
  2   2
   \   \
   3    3
 

Follow up: Solve it both recursively and iteratively.
'''

### Recursive way:
class Solution10:
    def isSymmetric(self, root: TreeNode) -> bool:

        def mirror(n1, n2):
            if not n1 and not n2:
                return True
            if not n1 or not n2: 
                return False
            return n1.val == n2.val and mirror(n1.left, n2.right) and mirror(n1.right, n2.left)
        
        if not root:
            return True
        return mirror(root.left, root.right)

### I will not write the iterative way here. It is pretty easy too.
# To implement the iterative way:
# Create a list and push the items into the list in the order of n1.left, n2.right, n1.right, n2.left
# Then each time we pop two items from the list and compare their values. If not the same, return False
# (when initializing the list, we need to push the value of root into it twice)


'''
111. Minimum Depth of Binary Tree
Easy

Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its minimum depth = 2.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N)  Space: O(N)
class Solution11:
    def minDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        # Store the count
        queue = [(root, 1)]
        
        # loop through the queue until nothing is inside
        for item, count in queue:
            if not item.left and not item.right: # reach the end, return
                return count
            if item.left:
                queue.append((item.left, count + 1))
            if item.right:
                queue.append((item.right, count + 1))
        

'''
404. Sum of Left Leaves
Easy

Find the sum of all left leaves in a given binary tree.

Example:

    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# BFS Iteration. Use a queue to store the current nodes in queue.  
#  Add to result if we find a left leaf.

# Time: O(N)  Space: O(N)
class Solution12:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if not root:
            return 0
        result = 0
        
        # queue stores a tuple of (node, isLeft)
        queue = [(root, 0)]
        for item, isLeft in queue:
            if not item.left and not item.right and isLeft:
                result += item.val 
            if item.left:
                queue.append((item.left, 1))
            if item.right:
                queue.append((item.right, 0))
        return result


'''
687. Longest Univalue Path
Easy

Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may or may not pass through the root.

The length of path between two nodes is represented by the number of edges between them.

Example 1:
Input:
              5
             / \
            4   5
           / \   \
          1   1   5
Output: 2

Example 2:
Input:
              1
             / \
            4   5
           / \   \
          4   4   5
Output: 2

Note: The given binary tree has not more than 10000 nodes. The height of the tree is not more than 1000.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Recursion
# Time: O(N) where N is number of nodes, as we need to go through all nodes
# Space: O(logN) as there are max of logN layers, where we have logN length of recursive call stack
class Solution13:
    def longestUnivaluePath(self, root: TreeNode) -> int:
        # An arrow is a one way path from root shooting to a leaf
        # We will keep track of the current arrow length and update the result everytime too
        self.result = 0
        
        # helper to calculate the max depth with root as n, and to update the result accordingly
        # everytime a new path length is generated
        def depthFromN(n: TreeNode):
            if not n:
                return 0
            # lDepth and rDepth for n.left and n.right
            lDepth = depthFromN(n.left)
            rDepth = depthFromN(n.right)
            # arrow length is depending on if left.val or right.val is equal to n.val
            lArrow = lDepth + 1 if n.left and n.left.val == n.val else 0
            rArrow = rDepth + 1 if n.right and n.right.val == n.val else 0
            # update result with the current arrow lengths
            self.result = max(self.result, lArrow + rArrow)
            # arrow length will become the new depth as it works with the current n.val
            return max(lArrow, rArrow)
        
        depthFromN(root)
        return self.result
            

'''
337. House Robber III
Medium

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:
Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1
Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

Example 2:
Input: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Recursion
# Time: O(N)  Space: O(logN)
class Solution14:
    def rob(self, root: TreeNode) -> int:
        
        # returns a tuple (doIt, notDoIt). doIt means the max money gain if we rob the current node n
        #   notDoIt means the max money gain if we skip the current node n
        def robTil(n: TreeNode):
            if not n:
                return 0, 0
            left = robTil(n.left)
            right = robTil(n.right)
            
            doIt = n.val + left[1] + right[1]
            notDoIt = max(left) + max(right)
            
            return doIt, notDoIt
        
        return max(robTil(root))


'''
671. Second Minimum Node In a Binary Tree
Easy

Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property root.val = min(root.left.val, root.right.val) always holds.

Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.

Example 1:
Input: 
    2
   / \
  2   5
     / \
    5   7
Output: 5
Explanation: The smallest value is 2, the second smallest value is 5.

Example 2:
Input: 
    2
   / \
  2   2
Output: -1
Explanation: The smallest value is 2, but there isn't any second smallest value.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Stupid question. Simple solution
class Solution15:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        if not root:
            return -1
        smallest = root.val 
        
        def recursionFindMin(n: TreeNode):
            if not n:
                return float('inf')
            if n.val > smallest:
                return n.val
            return min(recursionFindMin(n.left), recursionFindMin(n.right))
                      
        result = recursionFindMin(root)
        return result if result != float('inf') else -1



##################################### BFS ############################
'''
637. Average of Levels in Binary Tree
Easy

Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.
Example 1:
Input:
    3
   / \
  9  20
    /  \
   15   7
Output: [3, 14.5, 11]
Explanation:
The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on level 2 is 11. Hence return [3, 14.5, 11].
Note:
The range of node's value is in the range of 32-bit signed integer.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N) n is num of nodes
# Space: O(P + Q) where p is max depth of tree, and q is max width of the tree
class Solution16:
    def averageOfLevels(self, root: TreeNode):
        # BFS Solution
        if not root:
            return []
        # store a tuple (s, c) representing the sum and count for layer i
        sumAndCount = []
        # (root, i) is the current layer of root
        from collections import deque
        queue = deque([(root, 0)])
        while queue:
            curr, layer = queue.popleft()
            if len(sumAndCount) > layer:
                sumAndCount[layer] = (sumAndCount[layer][0] + curr.val, sumAndCount[layer][1] + 1)
            else:
                sumAndCount.append((curr.val, 1))
            if curr.left:
                queue.append((curr.left, layer + 1))
            if curr.right:
                queue.append((curr.right, layer + 1))
        result = []
        for s, c in sumAndCount:
            result.append(s / c)
        return result


'''
513. Find Bottom Left Tree Value
Medium

Given a binary tree, find the leftmost value in the last row of the tree.

Example 1:
Input:

    2
   / \
  1   3

Output:
1
Example 2:
Input:

        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

Output:
7
Note: You may assume the tree (i.e., the given root node) is not NULL.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# BFS
# Time: O(N)  Space: O(N)
class Solution17:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        currLeftMost = None
        # Use a queue to store nodes and their layer
        queue = [(root, 0)]
        prevLayer = -1
        
        # for each item in the queue, update result if it is the first item on a row
        for curr, currLayer in queue:
            if currLayer != prevLayer:
                currLeftMost = curr.val
                prevLayer = currLayer
            if curr.left:
                queue.append((curr.left, currLayer + 1))
            if curr.right:
                queue.append((curr.right, currLayer + 1))
        
        return currLeftMost


############################################# Tree Traversal ###########

'''
144. Binary Tree Preorder Traversal
Medium

Given a binary tree, return the preorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
Follow up: Recursive solution is trivial, could you do it iteratively?
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Iteration solution (recursion one should be easy)

# Use a stack as the holder for call stack
# Time: O(N)  Space: O(N)
class Solution18:
    def preorderTraversal(self, root: TreeNode):
        if not root:
            return []
        result = []
        # Use a stack to iteratively traverse the tree. We want to add the one that we want to execute
        #   later to the stack first, as stack is first in last out
        # Use a list to implement a stack
        stack = [root]
        while stack:
            curr = stack.pop()
            if not curr:
                continue
            result.append(curr.val)
            # append right first, as we want to pop left first
            stack.append(curr.right)
            stack.append(curr.left)
        return result


'''
145. Binary Tree Postorder Traversal
Hard

Given a binary tree, return the postorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [3,2,1]
Follow up: Recursive solution is trivial, could you do it iteratively?
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Iterative solution

# I did the same thing as I traverse a tree pre-orderly, except for putting left into the stack first
#   In this way, I get a totally reversed order of what we should have in post-order traversal
#   Therefore, I reverse the list and get the desired result.
#   Preorder traversal: https://leetcode.com/problems/binary-tree-preorder-traversal/description/

# Time and Space: O(N)
class Solution19:
    def postorderTraversal(self, root: TreeNode):
        if not root:
            return []
        result = []
        # Use a stack to iteratively traverse the tree. We want to add the one that we want to execute
        #   later to the stack first, as stack is first in last out
        # Use a list to implement a stack
        stack = [root]
        while stack:
            curr = stack.pop()
            if not curr:
                continue
            result.append(curr.val)
            # put left in first as we want to pop right first this time
            stack.append(curr.left)
            stack.append(curr.right)
        return result[::-1]


'''
94. Binary Tree Inorder Traversal
Medium

Given a binary tree, return the inorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
Follow up: Recursive solution is trivial, could you do it iteratively?
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Iteration solution

# Use a stack to store the nodes that has been went through. Keep going left in the traversal, and
#   once reach the end of left, pop from stack and set curr to it's right value, then again, keep
#   going left

# Time: O(N)  Space: O(N)
class Solution20:
    def inorderTraversal(self, root: TreeNode):
        if not root: 
            return []
        curr = root
        stack = []
        result = []
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left 
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result
        

