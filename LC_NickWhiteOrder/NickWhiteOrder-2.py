'''
Leetcode problems following the order of Nick White's playlist - II

515. Find Largest Value in Each Tree Row
814. Binary Tree Pruning
846. Hand of Straights
86. Partition List
938. Range Sum of BST
965. Univalued Binary Tree
979. Distribute Coins in Binary Tree
958. Check Completeness of a Binary Tree
662. Maximum Width of Binary Tree
114. Flatten Binary Tree to Linked List

'''


'''
515. Find Largest Value in Each Tree Row
Medium

Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).

 
Example 1: (Picture at https://leetcode.com/problems/find-largest-value-in-each-tree-row/)

Input: root = [1,3,2,5,3,null,9]
Output: [1,3,9]

Example 2:

Input: root = [1,2,3]
Output: [1,3]

Example 3:

Input: root = [1]
Output: [1]

Example 4:

Input: root = [1,null,2]
Output: [1,2]

Example 5:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree will be in the range [0, 104].
-231 <= Node.val <= 231 - 1

'''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution1:
    def largestValues(self, root: TreeNode) -> [int]:
        # This is a BFS question
        # Use a queue to do the BFS
        
        # Time: O(N)  Space: O(logN)
        
        if not root:
            return []
        q = [root]
        res = []
        
        while q:
            res.append(max(node.val for node in q))
            new_q = []
            for n in q:
                if n.left:
                    new_q.append(n.left)
                if n.right:
                    new_q.append(n.right)
            q = new_q
        
        return res
            

'''
814. Binary Tree Pruning
Medium

We are given the head node root of a binary tree, where additionally every node's value is either a 0 or a 1.

Return the same tree where every subtree (of the given tree) not containing a 1 has been removed.

(Recall that the subtree of a node X is X, plus every node that is a descendant of X.)

Example 1: (Images on https://leetcode.com/problems/binary-tree-pruning/)

Input: [1,null,0,0,1]
Output: [1,null,0,null,1]
 
Explanation: 
Only the red nodes satisfy the property "every subtree not containing a 1".
The diagram on the right represents the answer.

Example 2:

Input: [1,0,1,0,0,0,1]
Output: [1,null,1,null,1]

Example 3:

Input: [1,1,0,1,1,0,1,0]
Output: [1,1,0,1,1,null,1]


Note:

The binary tree will have at most 200 nodes.
The value of each node will only be 0 or 1.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution2:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        # This is a recursion problem
        
        # Time: O(N)  We need to go through all nodes
        # Space: O(H)  which is the height of the tree
        
        # Base case:
        if not root:
            return None
        
        # Recursive case:
        else:
            root.left = self.pruneTree(root.left)
            root.right = self.pruneTree(root.right)
            return None if (not root.left and not root.right and root.val == 0) else root 
        

'''
846. Hand of Straights
Medium

Alice has a hand of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

Return true if and only if she can.

 
Example 1:

Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8].

Example 2:

Input: hand = [1,2,3,4,5], W = 4
Output: false
Explanation: Alice's hand can't be rearranged into groups of 4.
 

Constraints:

1 <= hand.length <= 10000
0 <= hand[i] <= 10^9
1 <= W <= hand.length
Note: This question is the same as 1296: https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/

'''

class Solution3:
    def isNStraightHand(self, hand: [int], W: int) -> bool:
        # Method 1:
        # First sort the list of cards, then group them. When grouping, take notes of the next starting index
        #   so that we can know from which card, the next group should continue with.
        #
        # Method 2: (this solution)
        # Use something like a TreeMap in Java so we have an ordered HashMap. Then, we retrieve values from 
        #   the hashmap while decreasing the value of count. In python, we have collections.OrderedDict
        
        # Time: O(N)  Space: O(N)
        
        if len(hand) % W != 0:
            return False
        
        from collections import OrderedDict
        
        od = OrderedDict()
        hand.sort()     # Sort list of cards first so our dictionary will be sorted by key values
        for card in hand:
            od[card] = 1 if card not in od else od[card] + 1    # build dictionary
                    
        while len(od) > 0:
            first_key = next(iter(od.items()))[0]   # Use iterator to get first key in a fast way
            for n in range(first_key, first_key + W):   # We need W cards in each sequence
                if n not in od:     # Return false if card is not here
                    return False
                else:
                    if od[n] > 1:   # Decrease count by 1 each time we meet a card
                        od[n] -= 1
                    else:
                        od.pop(n)
        
        return True


'''
86. Partition List
Medium

Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

Example:

Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5

'''

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution4:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # Create two separate lists, small and large - small for all numbers smaller than x, large for
        #   all numbers equal or greater than x. Then, combine them to form the result
        # Time: O(N)  Space: O(1)
        small = p_small = ListNode()
        large = p_large = ListNode()
        curr = head
        while curr:
            if curr.val < x:
                p_small.next = curr
                p_small = p_small.next
            else:
                p_large.next = curr
                p_large = p_large.next
            curr = curr.next
        
        p_large.next = None
        p_small.next = large.next
        
        return small.next
        
    
'''
938. Range Sum of BST
Easy

Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.

 
Example 1:

Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32

Example 2:

Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
 

Note:

The number of nodes in the tree is at most 10000.
The final answer is guaranteed to be less than 2^31.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution5:
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        # We can solve this problem in recursive way, or iterative way
        
        # Recursive:
        # Time: O(N)  Space: O(H) where H is the height of the tree
        
        def dfs(node):
            if not node:
                return 0
            else:
                if L <= node.val <= R:
                    return node.val + dfs(node.left) + dfs(node.right)
                elif node.val < L:
                    return dfs(node.right)
                else:
                    return dfs(node.left)
        
        return dfs(root)
    
        # Iterative way is also easy, and similar. Just use a stack to store the nodes, and use similar
        #   conditions to find proper children nodes and push back to the stack


'''
965. Univalued Binary Tree
Easy

A binary tree is univalued if every node in the tree has the same value.

Return true if and only if the given tree is univalued.

 

Example 1: (Images at https://leetcode.com/problems/univalued-binary-tree/)

Input: [1,1,1,1,1,null,1]
Output: true

Example 2:

Input: [2,2,2,5,2]
Output: false
 

Note:

The number of nodes in the given tree will be in the range [1, 100].
Each node's value will be an integer in the range [0, 99].

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution6:
    def isUnivalTree(self, root: TreeNode) -> bool:
        # recursion:
        head_val = root.val
        
        def dfs(node):
            if not node:
                return True
            else:
                return node.val == head_val and dfs(node.left) and dfs(node.right)
        
        return dfs(root)


'''
979. Distribute Coins in Binary Tree
Medium

Given the root of a binary tree with N nodes, each node in the tree has node.val coins, and there are N coins total.

In one move, we may choose two adjacent nodes and move one coin from one node to another.  (The move may be from parent to child, or from child to parent.)

Return the number of moves required to make every node have exactly one coin.

 
Example 1: (check pictures here https://leetcode.com/problems/distribute-coins-in-binary-tree/)

Input: [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.

Example 2:

Input: [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves].  Then, we move one coin from the root of the tree to the right child.

Example 3:

Input: [1,0,2]
Output: 2

Example 4:

Input: [1,0,0,null,3]
Output: 4
 
Note:

1<= N <= 100
0 <= node.val <= N

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution7:
    def distributeCoins(self, root: TreeNode) -> int:
        # We can solve this question recursively, or say, DFS
        # For each node with value v, we have too make |v - 1| moves to make it have exactly 1 coin
        
        # Time: O(N)  Space: O(H) where H is height of tree
        self.ans = 0
        
        def dfs(node):
            if not node:
                return 0
            else:
                l = dfs(node.left)
                r = dfs(node.right)
                self.ans += abs(l) + abs(r)
                return node.val + l + r - 1
        
        dfs(root)
        return self.ans
            

'''
958. Check Completeness of a Binary Tree
Medium

Given a binary tree, determine if it is a complete binary tree.

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.


Example 1:

Input: [1,2,3,4,5,6]
Output: true
Explanation: Every level before the last is full (ie. levels with node-values {1} and {2, 3}), and all nodes in the last level ({4, 5, 6}) are as far left as possible.

Example 2:

Input: [1,2,3,4,5,null,7]
Output: false
Explanation: The node with value 7 isn't as far left as possible.
 
Note:

The tree will have between 1 and 100 nodes.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution8:
    def isCompleteTree(self, root: TreeNode) -> bool:
        # Basically, we want to do a level order traversal (visit row by row), and the first None value
        #   we see should mark the end of tree. In other words, we shouldn't see any not-None node after
        #   any None node. If we find there's any node after a None value, we simply return false. 
        # Time: O(N)  Space: O(N) (Using queue it should be O(W) where W is the width of the tree)
        
        # We have a bool to store if we've seen a None
        seen = False
        
        # BFS
        bfs = [root]
        for n in bfs:
            if not n:
                seen = True
            else:
                if seen:
                    return False
                bfs.append(n.left)
                bfs.append(n.right)
                
        return True


'''
662. Maximum Width of Binary Tree
Medium

Given a binary tree, write a function to get the maximum width of the given tree. The maximum width of a tree is the maximum width among all levels.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

It is guaranteed that the answer will in the range of 32-bit signed integer.

Example 1:

Input: 

           1
         /   \
        3     2
       / \     \  
      5   3     9 

Output: 4
Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9).

Example 2:

Input: 

          1
         /  
        3    
       / \       
      5   3     

Output: 2
Explanation: The maximum width existing in the third level with the length 2 (5,3).

Example 3:

Input: 

          1
         / \
        3   2 
       /        
      5      

Output: 2
Explanation: The maximum width existing in the second level with the length 2 (3,2).

Example 4:

Input: 

          1
         / \
        3   2
       /     \  
      5       9 
     /         \
    6           7
Output: 8
Explanation:The maximum width existing in the fourth level with the length 8 (6,null,null,null,null,null,null,7).
 

Constraints:

The given binary tree will have between 1 and 3000 nodes.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution9:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        # The key concept required for solving this problem is indexing a Binary Tree
        # The rule is: For a parent node of index i, its left node will have index i * 2, and its right
        #   node will always have index i * 2 + 1
        # For example, below is a binary tree, with its indexes as node value:
        #           1
        #         /   \
        #        2     3
        #       / \   / \
        #      4   5 6   7
        # And width of a row can be calculated by index of right most node - index of left most node + 1
        #   ie: 3rd row: 7 - 4 + 1 = 4
        
        # We can use either BFS or DFS to traverse the tree. The BFS solution is more intuitive because 
        #   it is traversing row by row, but DFS is also pretty easy. The only thing we need to do with 
        #   DFS is to mark the first node in a row. If we use pre-order traversal, the sequence of visiting
        #   a node is always equal to its row number if it is the first node.
        
        # BFS Solution:
        # Time: O(N)  Space: O(N)
        from collections import deque
        queue = deque()
        queue.append((root, 1))
        res = 0
        
        while queue:
            _, left_most_index = queue[0]
            l = len(queue)
            for _ in range(l):
                curr_node, curr_index = queue.popleft()
                if curr_node.left:
                    queue.append((curr_node.left, curr_index * 2))
                if curr_node.right:
                    queue.append((curr_node.right, curr_index * 2 + 1))
            res = max(res, curr_index - left_most_index + 1)
            
        return res
            

'''
114. Flatten Binary Tree to Linked List
Medium

Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:

    1
   / \
  2   5
 / \   \
3   4   6
The flattened tree should look like:

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution10:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
#         # DFS, recursion
#         # Time: O(N)  Space: O(N)
        
#         # recursive method returns the tail of a flattened tree
#         def flatten_recursion(node: TreeNode) -> TreeNode:
#             if not node:        # solve None case
#                 return None
#             if not node.left and not node.right:    # solve leaf case
#                 return node
            
#             left_tail = flatten_recursion(node.left)    # get left and right tails
#             right_tail = flatten_recursion(node.right)
            
#             if left_tail:       # flat the tree if there's a left division
#                 left_tail.right, node.right, node.left = node.right, node.left, None
                
#             return right_tail if right_tail else left_tail  # return the right most tail
        
#         flatten_recursion(root)


        # DFS, Iteration
        # Time: O(N)  Space: O(N)
        
        # Iteration is a superior solution since if the tree is large, using recursion and relying on 
        #   system stack is not efficient
        # Use a stack to solve it iteratively. In the stack, store tuples of (node, state), state is 
        #   to determine if the node's children has been processed or not. With State of 1, it means
        #   the node is not processed yet, with 0 it means that the node is processed and the subtree
        #   with it as root has been flatten
        # For full explaination please see the Solution
        if not root:
            return None
        
        stack = [(root, 1)]
        while stack:
            node, state = stack.pop()

            if not node.left and not node.right:
                tail = node
            
            if state:
                if node.left:
                    stack.append((node, 0))
                    stack.append((node.left, 1))
                elif node.right:
                    stack.append((node.right, 1))
            else:
                if tail:
                    right_node = node.right
                    tail.right = node.right
                    node.right = node.left
                    node.left = None
                if right_node:
                    stack.append((right_node, 1))
                    

