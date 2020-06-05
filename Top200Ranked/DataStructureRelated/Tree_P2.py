'''
##################################### Tree Part II ###############################
From Binary Search Tree to the end

669. Trim a Binary Search Tree
230. Kth Smallest Element in a BST
538. Convert BST to Greater Tree
235. Lowest Common Ancestor of a Binary Search Tree
236. Lowest Common Ancestor of a Binary Tree
108. Convert Sorted Array to Binary Search Tree
109. Convert Sorted List to Binary Search Tree
653. Two Sum IV - Input is a BST
530. Minimum Absolute Difference in BST
501. Find Mode in Binary Search Tree
208. Implement Trie (Prefix Tree)
677. Map Sum Pairs

'''

'''
669. Trim a Binary Search Tree
Easy

Given a binary search tree and the lowest and highest boundaries as L and R, trim the tree so that all its elements lies in [L, R] (R >= L). You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

Example 1:
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
Example 2:
Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1
 '''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Recursive solution

# Time Space: O(N)
class Solution1:
    def trimBST(self, root: TreeNode, L: int, R: int):
        # recursive helper. Returns the trimed version of the tree at root n
        def trimFromNode(n):
            if not n:
                return None
            # if curr value is greater than the right boundary, trimed tree should be n.left
            elif n.val > R:
                return trimFromNode(n.left)
            # if curr value is smaller than the left boundary, trimed tree should be n.right 
            elif n.val < L:
                return trimFromNode(n.right)
            # if curr value lies in the range, left and right should both be trimed
            else:
                n.left = trimFromNode(n.left)
                n.right = trimFromNode(n.right)
                return n
                
        return trimFromNode(root)


'''
230. Kth Smallest Element in a BST
Medium

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note:
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:

Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution2:
    def kthSmallest(self, root: TreeNode, k: int):
        # Use a stack, push nodes into it from root to the left most. Once we reach the left most leaf,
        #   we pop and set the curr to curr.right, then keep going left. 
        # k -= 1 for each pop. If after we pop, k == 0, we return the poped value
        stack = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val 
            curr = curr.right 
        return -1


'''
538. Convert BST to Greater Tree
Easy

Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

Example:

Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
Note: This question is the same as 1038: https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N)  Space: O(N)
class Solution3:
    total = 0
    def convertBST(self, root: TreeNode) -> TreeNode:
        # The idea is to traverse the BST in reverse in-order manner. Then we note down the sum
        #   of all values traversed and set the curr to it as the new value
        
        # I'll use recursion for this one. If you want iterative solution, use stack for the traversal.
        if not root:
            return None
        self.convertBST(root.right)
        self.total += root.val
        root.val = self.total 
        self.convertBST(root.left)
        return root


'''
235. Lowest Common Ancestor of a Binary Search Tree
Easy

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given binary search tree:  root = [6,2,8,0,4,7,9,null,null,3,5]

(view image at https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)


Example 1:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Time: O(logN)  (worst case will be O(N). If the tree is very unbalanced, we may need to go through
#   the whole tree)
# Space: O(logN)
class Solution4:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # The idea is that when we try to search a node in a tree, we use binary search. We can use
        #   a similar procedure to search for the lowest common ancestors. The reason is that because
        #   it is a BST, the LCA must be the first node we find that is <= q, and >= p
        smaller = min(p.val, q.val)
        larger = max(p.val, q.val)
        if smaller <= root.val <= larger:
            return root
        elif root.val < smaller:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return self.lowestCommonAncestor(root.left, p, q)


'''
236. Lowest Common Ancestor of a Binary Tree
Medium

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

(To view image, go to https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
 

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
Example 2:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
 

Note:

All of the nodes' values will be unique.
p and q are different and both values will exist in the binary tree.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Time: O(logN) but O(N) for worst case  
# Space: O(logN) but O(N) for worst case
class Solution5:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        # we will always get to the basic case where root equals to p or q and we find the LCA
        if root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # if p and q are on different side, then curr node is the LCA
        if left and right:
            return root
        # if p and q are on same side, simply return the sub problem depend on which side they are on
        else:
            return left if left else right


'''
108. Convert Sorted Array to Binary Search Tree
Easy

Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Used divide and conquer. Get the middle and set it as root, then for left and right, get the sub BST
#   and set them to left and right

# Time: O(N)  Space: O(logN)
class Solution6:
    def sortedArrayToBST(self, nums: [int]):
        if len(nums) == 0:
            return None
        mid = len(nums) // 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1:])
        return node

solution6 = Solution6()
print(solution6.sortedArrayToBST(nums = [-10,-3,0,5,9]))


'''
109. Convert Sorted List to Binary Search Tree
Medium

Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted linked list: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
'''

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Inorder Simulation
# This is a pretty hard to understand recursion

# Time: O(N) as we traversed each item exactly once
# Space: O(logN) as the call stack size is the height of tree
class Solution7:
    def sortedListToBST(self, head: ListNode):
        
        # helper function to traverse list from the l th element to the r th element
        def traverse(l, r):
            nonlocal head
            
            # return None if we've reached leaf
            if l > r:
                return None
            m = l + (r - l) // 2
            # traverse left first
            left = traverse(l, m - 1)
            # head will be automatically increased to the middle, as when we traverse left, we keep
            #   increasing it's value
            currNode = TreeNode(head.val)
            # increase index of head
            head = head.next
            # traverse right after set the root
            right = traverse(m + 1, r)
            currNode.left = left
            currNode.right = right
            return currNode
        
        # helper to count length of linked list
        def countLen():
            counter = 0
            p = head
            while p:
                counter += 1
                p = p.next
            return counter
        
        return traverse(0, countLen() - 1)


'''
653. Two Sum IV - Input is a BST
Easy

Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

Example 1:

Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True
 

Example 2:

Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 28

Output: False
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N)  Space: O(N)
class Solution8:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        # There are multiple ways to solve it:
        #   1. Traverse the tree in-order to a list, then use two pointers to find a match
        #   2. Use BFS to traverse through the whole tree layer by layer and use a set to note down all 
        #       traversed values. Compare each value to see if (k - i) is contained in the set
        #   3. Use recursion to traverse through the whole tree and after that same as no.2 
        
        # Solution using BFS:
        if not root:
            return False
        queue = [root]
        visited = set()
        for curr in queue:
            if (k - curr.val) in visited:
                return True
            visited.add(curr.val)
            if curr.left: 
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return False


'''
530. Minimum Absolute Difference in BST
Easy

Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.

Example:

Input:

   1
    \
     3
    /
   2

Output:
1

Explanation:
The minimum absolute difference is 1, which is the difference between 2 and 1 (or between 2 and 3).
 

Note:

There are at least two nodes in this BST.
This question is the same as 783: https://leetcode.com/problems/minimum-distance-between-bst-nodes/
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N)  Space: O(H) H is height of tree
class Solution9:
    def getMinimumDifference(self, root: TreeNode) -> int:
        # We can do in-order traversal
        stack = []
        curr = root
        prev = float('-inf')
        result = float('inf')
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result = min(result, curr.val - prev)
            prev = curr.val
            curr = curr.right
        return result


'''
501. Find Mode in Binary Search Tree
Easy

Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) in the given BST.

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than or equal to the node's key.
The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
Both the left and right subtrees must also be binary search trees.
 

For example:
Given BST [1,null,2,2],

   1
    \
     2
    /
   2
 

return [2].

Note: If a tree has more than one mode, you can return them in any order.

Follow up: Could you do that without using any extra space? (Assume that the implicit stack space incurred due to recursion does not count).
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Time: O(N)  Space: O(N)
class Solution10:
    def findMode(self, root: TreeNode) -> [int]:
        # This is NOT a O(1) space complexity solution. O(1) space is too complex but worth review 
        #   if I have time
        # use a dictionary to store the count of each value
        nodeCount = {}
        maxCount = 0
        
        def dfs(node):
            nonlocal maxCount
            if node:
                nodeCount[node.val] = nodeCount.get(node.val, 0) + 1
                maxCount = max(maxCount, nodeCount[node.val])
                dfs(node.left)
                dfs(node.right)
                
        dfs(root)
        result = []

        for k in nodeCount.keys():
            if nodeCount[k] == maxCount:
                result.append(k)
        return result


############################################ Trie #########################

'''
208. Implement Trie (Prefix Tree)
Medium

Implement a trie with insert, search, and startsWith methods.

Example:

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true
Note:

You may assume that all inputs are consist of lowercase letters a-z.
All inputs are guaranteed to be non-empty strings.
'''

class TrieNode:
    # We need a list of TrieNodes that links the curr head to the next layer of nodes
    def __init__(self):
        self.isEnd = False          # whether the current node is a ending of a word
        self.links = [None] * 26    # link to the next layer
        
    def print(self):
        line = []
        for i in range(len(self.links)):
            if i:
                line.append(1)
            else:
                line.append(0)
        print(line)


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # The root is gonna contain a TrieNode
        self.root = TrieNode()
        

    # Time: O(N)  N is length of input word
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        curr = self.root
        for c in word:
            index = ord(c) - ord('a')
            if not curr.links[index]:
                curr.links[index] = TrieNode()
            curr = curr.links[index]
        curr.isEnd = True
            
    
    # a Helper function that returns a tuple (startsWith, reachedEnd)
    # which represents whether the word is a prefix, and by reaching end of the word, if we've reached
    #   the end of theh whole word
    # We always return false for 'reachedEnd' if 'startsWith' returns False as it doesn't matter
    def helper(self, word: str) -> (bool, bool):
        curr = self.root
        for c in word:
            index = ord(c) - ord('a')
            if not curr.links[index]:
                return (False, False)
            curr = curr.links[index]
        return (True, curr.isEnd)
        

    # Time: O(N)  N is length of input word
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        return self.helper(word)[1]
    

    # Time: O(N)  N is length of input word
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        return self.helper(prefix)[0]



# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

obj = Trie()
obj.insert('hello')
obj.root.print()
obj.root.links[7].print()
print(obj.startsWith('hel'))
print(obj.startsWith('heo'))
print(obj.search('hello'))
print(obj.search('hel'))


'''
677. Map Sum Pairs
Medium

Implement a MapSum class with insert, and sum methods.

For the method insert, you'll be given a pair of (string, integer). The string represents the key and the integer represents the value. If the key already existed, then the original key-value pair will be overridden to the new one.

For the method sum, you'll be given a string representing the prefix, and you need to return the sum of all the pairs' value whose key starts with the prefix.

Example 1:
Input: insert("apple", 3), Output: Null
Input: sum("ap"), Output: 3
Input: insert("app", 2), Output: Null
Input: sum("ap"), Output: 5
'''

# class TrieNode:
#     # We need a list of TrieNodes that links the curr head to the next layer of nodes
#     def __init__(self):
#         self.val = 0                # The end value that carries with the node
#         self.links = [None] * 26    # link to the next layer


# This question is very similar to 208. Implement Trie (Prefix Tree)
# Carry a sum val with each TrieNode to count the sum
# Time complexity of each operation is O(N) where N is length of word
class MapSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    # Time: O(N)
    def insert(self, key: str, val: int) -> None:
        curr = self.root
        for c in key:
            index = ord(c) - ord('a')
            if not curr.links[index]:
                curr.links[index] = TrieNode()
            curr = curr.links[index]
        curr.val = val
        

    # Time: O(N)  Space: O(H) H is the height of the Trie
    def sum(self, prefix: str) -> int:
        curr = self.root
        for c in prefix:
            index = ord(c) - ord('a')
            if not curr.links[index]:
                return 0
            curr = curr.links[index]
        sum = 0
        
        # helper function to count the sum of values beginning with the TrieNode node
        def countSum(node):
            nonlocal sum
            sum += node.val
            for n in node.links:
                if n:
                    countSum(n)
        
        countSum(curr)
        return sum


# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)