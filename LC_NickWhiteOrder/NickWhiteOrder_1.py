'''
Leetcode problems following the order of Nick White's playlist

590. N-ary Tree Postorder Traversal
589. N-ary Tree Preorder Traversal
709. To Lower Case
844. Backspace String Compare
344. Reverse String
876. Middle of the Linked List
657. Robot Return to Origin
841. Keys and Rooms
977. Squares of a Sorted Array
16. 3Sum Closest

'''

'''
590. N-ary Tree Postorder Traversal
Easy

Given an n-ary tree, return the postorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).

 
Follow up:

Recursive solution is trivial, could you do it iteratively?


Example 1:

Input: root = [1,null,3,2,4,null,5,6]
Output: [5,6,3,2,4,1]

Example 2:

Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [2,6,14,11,7,3,12,8,4,13,9,10,5,1]
 

Constraints:

The height of the n-ary tree is less than or equal to 1000
The total number of nodes is between [0, 10^4]

'''

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution1:
    def postorder(self, root) -> [int]:
        if not root:
            return []
        # We need to use a stack to solve this problem iteratively
        stack = [root]
        result = []
        while stack:
            curr = stack.pop()
            result.append(curr.val)
            # Simply loop throught all the children left to right as we are using a stack, and they will
            #   be popped out in right to left order
            for child in curr.children:
                stack.append(child)
        # Reverse the order as our result is reversed in order
        return result[::-1]


'''
589. N-ary Tree Preorder Traversal
Easy

Given an n-ary tree, return the preorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).

 
Follow up:

Recursive solution is trivial, could you do it iteratively?

 
Example 1:

Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]

Example 2:

Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [1,2,3,6,7,11,14,4,8,12,5,9,13,10]
 

Constraints:

The height of the n-ary tree is less than or equal to 1000
The total number of nodes is between [0, 10^4]

'''

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution2:
    def preorder(self, root) -> [int]:
        # For preorder, we need to use DFS, with a stack for the call back
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            curr = stack.pop()
            result.append(curr.val)
            # We need to reverse the children, as we want to push left in last, so it can be popped first
            for c in curr.children[::-1]:
                stack.append(c)
        return result


'''
709. To Lower Case
Easy

Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.

 
Example 1:

Input: "Hello"
Output: "hello"

Example 2:

Input: "here"
Output: "here"

Example 3:

Input: "LOVELY"
Output: "lovely"

'''

class Solution3:
    # Time: O(N)  Space: O(1)
    def toLowerCase(self, str: str) -> str:
        # use a dictionary to pair each upper case letter with lower case letter
        upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"
        h = dict(zip(upper, lower))
        
        '''
        res = []
        for c in str:
            if c in h:
                res.append(h[c])
            else:
                res.append(c)
        
        return ''.join(res)
        '''

        # One line solution:
        return ''.join([h[c] if c in h else c for c in str])


'''
844. Backspace String Compare
Easy

Given two strings S and T, return if they are equal when both are typed into empty text editors. # means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

Example 1:

Input: S = "ab#c", T = "ad#c"
Output: true
Explanation: Both S and T become "ac".
Example 2:

Input: S = "ab##", T = "c#d#"
Output: true
Explanation: Both S and T become "".
Example 3:

Input: S = "a##c", T = "#a#c"
Output: true
Explanation: Both S and T become "c".
Example 4:

Input: S = "a#c", T = "b"
Output: false
Explanation: S becomes "c" while T becomes "b".
Note:

1 <= S.length <= 200
1 <= T.length <= 200
S and T only contain lowercase letters and '#' characters.
Follow up:

Can you solve it in O(N) time and O(1) space?

'''

class Solution4:
    def backspaceCompare(self, S: str, T: str) -> bool:
        # First solution: We can use two stacks and pop from it once we reach a '#'.
        # This way, both Time and Space complexity is O(N)
        
        # Second solution is better: Use two pointers from right to left on both strings, take note when we 
        #   reach a '#' and skip certain amount of chars afterwards. We will create a method that acts as
        #   a generator, to transform the strings to a form for comparison
        #   Time: O(N)  Space: O(1)
        if not S and not T:
            return True
        
        # Function that is basically a generator, which returns a character once at a time
        # Then, it will stay at the state, until it is called again
        # In this case, if input s is "ab#c", calling it first will yield 'c', second time it will yield '#'
        def transform(s: str):
            skip_count = 0
            for c in reversed(s):
                if c == '#':
                    skip_count += 1
                elif skip_count:
                    skip_count -= 1
                else:
                    yield(c)
        
        # Here is an example of how this generator works:
        # s_gen = transform(S)
        # t_gen = transform(T)
        # print(next(s_gen), next(s_gen), next(t_gen))  -->  c a c
        
        # We will use Python's Itertool module, and Itertools.zip_longest() function. 
        # zip_longest(a, b, fillvalue) returns pairs of values from inputs a and b. When one is shorter, 
        #   fillvalue is used to fill in the remaining pairs
        return all([a == b for a, b in itertools.zip_longest(transform(S), transform(T))])


'''
344. Reverse String
Easy

Write a function that reverses a string. The input string is given as an array of characters char[].

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

You may assume all the characters consist of printable ascii characters.


Example 1:

Input: ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Example 2:

Input: ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]

'''

class Solution5:
    def reverseString(self, s: [str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        # Use two pointers
        p1 = 0
        p2 = len(s) - 1
        while p1 < p2:
            s[p1], s[p2] = s[p2], s[p1]
            p1 += 1
            p2 -= 1

        # Actually, this can be solved with only 1 pointer, as we really only need to track 1, the other
        #   can be achieved by len(s) - p1. 
        # It's also pretty easy so I won't code it.


'''
876. Middle of the Linked List
Easy

Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.


Example 1:

Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).
Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.

Example 2:

Input: [1,2,3,4,5,6]
Output: Node 4 from this list (Serialization: [4,5,6])
Since the list has two middle nodes with values 3 and 4, we return the second one.
 

Note:

The number of nodes in the given list will be between 1 and 100.

'''

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution6:
    def middleNode(self, head: ListNode) -> ListNode:
        # Use two pointers, one Fast pointer, one Slow pointer
        # Time: O(N)  Space: O(1)
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            
        return slow


'''
657. Robot Return to Origin
Easy

There is a robot starting at position (0, 0), the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves.

The move sequence is represented by a string, and the character moves[i] represents its ith move. Valid moves are R (right), L (left), U (up), and D (down). If the robot returns to the origin after it finishes all of its moves, return true. Otherwise, return false.

Note: The way that the robot is "facing" is irrelevant. "R" will always make the robot move to the right once, "L" will always make it move left, etc. Also, assume that the magnitude of the robot's movement is the same for each move.


Example 1:

Input: moves = "UD"
Output: true
Explanation: The robot moves up once, and then down once. All moves have the same magnitude, so it ended up at the origin where it started. Therefore, we return true.

Example 2:

Input: moves = "LL"
Output: false
Explanation: The robot moves left twice. It ends up two "moves" to the left of the origin. We return false because it is not at the origin at the end of its moves.

Example 3:

Input: moves = "RRDD"
Output: false

Example 4:

Input: moves = "LDRRLRUULR"
Output: false
 

Constraints:

1 <= moves.length <= 2 * 104
moves only contains the characters 'U', 'D', 'L' and 'R'.

'''

class Solution7:
    def judgeCircle(self, moves: str) -> bool:
        # Too easy. Just simulate
        # Time: O(N) N is length of moves   Space: O(1)
        x = y = 0
        for c in moves: 
            if c == 'U': y += 1
            elif c == 'D': y -= 1
            elif c == 'R': x += 1
            else: x -= 1
        
        return x == y == 0


'''
841. Keys and Rooms
Medium

There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, and each room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.

Example 1:

Input: [[1],[2],[3],[]]
Output: true
Explanation:  
We start in room 0, and pick up key 1.
We then go to room 1, and pick up key 2.
We then go to room 2, and pick up key 3.
We then go to room 3.  Since we were able to go to every room, we return true.

Example 2:

Input: [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: We can't enter the room with number 2.
Note:

1 <= rooms.length <= 1000
0 <= rooms[i].length <= 1000
The number of keys in all rooms combined is at most 3000.

'''

class Solution8:
    def canVisitAllRooms(self, rooms: [[int]]) -> bool:
        # Use DFS
        # We use a stack to implement the DFS. Starting from room 0, we put all rooms on the key list 
        #   of it into the stack, then moving forward, we pop from the stack and again, push all rooms
        #   on curr's key list onto the stack
        
        # Time: O(N+K)   Space: O(N) 
        #   where N is the total number of rooms, K is the total number of keys
        
        stack = [0]
        visited = set()     # Use a set to store visited rooms
        remaining_count = len(rooms)
        
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                remaining_count -= 1
                for r in rooms[curr]:
                    stack.append(r)
        
        return remaining_count == 0
         
        
'''
977. Squares of a Sorted Array
Easy

Given an array of integers A sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.

 

Example 1:

Input: [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Example 2:

Input: [-7,-3,2,3,11]
Output: [4,9,9,49,121]
 

Note:

1 <= A.length <= 10000
-10000 <= A[i] <= 10000
A is sorted in non-decreasing order.

'''
        
class Solution9:
    def sortedSquares(self, A: [int]) -> [int]:
        # First solution is to do square first, then sort, takes NlogN time
        # Better solution is to use two pointers, one left to right, one right to left
        
        # Time: O(N)  Space: O(1)
        l = 0
        r = len(A) - 1
        res = []
        
        while l <= r:
            if A[l] * A[l] > A[r] * A[r]:
                res.append(A[l] * A[l])
                l += 1
            else: 
                res.append(A[r] * A[r])
                r -= 1
        
        return res[::-1]


'''
16. 3Sum Closest
Medium

Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

 

Example 1:

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
 

Constraints:

3 <= nums.length <= 10^3
-10^3 <= nums[i] <= 10^3
-10^4 <= target <= 10^4

'''

class Solution10:
    def threeSumClosest(self, nums: [int], target: int) -> int:
        # 1. First, we sort the list
        # 2. Then, we fix one number
        # 3. Use two pointers on all numbers to the right of the fixed number
        # 4. Calculate and update the minimum absolute difference, while moving the two pointers
        # 5. If sum is too small, move left pointer to the right, vise versa
        
        # Time: O(N*N) since sort takes NlogN but it won't count as we only count the more complex one
        # Space: O(N) for the merge (quick) sort that is built in
        
        nums.sort()
        res = 4000
        lowest_diff = 14000        # A random large initial number that is larger than the largest 
                                   #    possible difference
        for i in range(len(nums)):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == target:
                    return target
                else:
                    if abs(s - target) < lowest_diff:
                        lowest_diff = abs(s - target)
                        res = s
                    if s < target:
                        l += 1
                    else:
                        r -= 1
        
        return res
