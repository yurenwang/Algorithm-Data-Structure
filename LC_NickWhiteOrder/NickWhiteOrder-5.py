'''
Leetcode problems following the order of Nick White's playlist - V

463. Island Perimeter
1160. Find Words That Can Be Formed by Characters
796. Rotate String
193. Valid Phone Numbers
482. License Key Formatting
904. Fruit Into Baskets
807. Max Increase to Keep City Skyline
1119. Remove Vowels from a String
143. Reorder List
92. Reverse Linked List II

'''

'''
463. Island Perimeter
Easy

You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

 
Example 1:

Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.

Example 2:

Input: grid = [[1]]
Output: 4

Example 3:

Input: grid = [[1,0]]
Output: 4
 

Constraints:

row == grid.length
col == grid[i].length
1 <= row, col <= 100
grid[i][j] is 0 or 1.

'''

class Solution1:
    def islandPerimeter(self, grid: [[int]]) -> int:
        # Let's take a look at the island from 4 different views. When viewing from up up to down, we can see all parts
        #   of its perimeter that is on top, then view from left to right, we can see all parts of its perimeters on its
        #   left side, and so on. This way, we can have 4 loops, each with a view angle and add the length of perimeters 
        #   together to become the whole length
        
        # Time: O(WH) where w and h are width and height of the grid
        # Space: O(1)
        
        h = len(grid)
        if h == 0: return 0
        w = len(grid[0])
        res = 0
        
        for i in range(h):
            for j in range(w):
                if (i == 0 and grid[i][j]) or grid[i][j] and not grid[i - 1][j]:        # Top view
                    res += 1
                if (i == h - 1 and grid[i][j]) or grid[i][j] and not grid[i + 1][j]:    # Bottom view
                    res += 1
                if (j == 0 and grid[i][j]) or grid[i][j] and not grid[i][j - 1]:        # Left view
                    res += 1
                if (j == w - 1 and grid[i][j]) or grid[i][j] and not grid[i][j + 1]:    # Right view
                    res += 1
        
        return res
                    

'''
1160. Find Words That Can Be Formed by Characters
Easy

You are given an array of strings words and a string chars.

A string is good if it can be formed by characters from chars (each character can only be used once).

Return the sum of lengths of all good strings in words.


Example 1:

Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: 
The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.

Example 2:

Input: words = ["hello","world","leetcode"], chars = "welldonehoneyr"
Output: 10
Explanation: 
The strings that can be formed are "hello" and "world" so the answer is 5 + 5 = 10.
 

Note:

1 <= words.length <= 1000
1 <= words[i].length, chars.length <= 100
All strings contain lowercase English letters only.

'''

class Solution2:
    def countCharacters(self, words: [str], chars: str) -> int:
#         # Similar question to 916. Word Subsets 
#         # Time: O(M + N)  Space: O(M + N)
#         # This is method using zip()
#         def countWord(word):
#             res = [0] * 26
#             for c in word:
#                 res[ord(c) - ord('a')] += 1
#             return res
        
#         checker = countWord(chars)
#         good_words = [w for w in words if all([x <= y for x, y in zip(countWord(w), checker)])]
#         result = 0
#         for w in good_words:
#             result += len(w)
            
#         return result

        # This is using only collections.Counter()
        # Time: O(M + N)  Space: O(M + N)
        res = 0
        checker = collections.Counter(chars)
        
        for w in words:
            w_count = collections.Counter(w)
            for c in w_count:           # For Else loop, else block is executed after for is done
                if w_count[c] > checker[c]:
                    break
            else:
                res += len(w)
            
        return res


'''
796. Rotate String
Easy

We are given two strings, A and B.

A shift on A consists of taking string A and moving the leftmost character to the rightmost position. For example, if A = 'abcde', then it will be 'bcdea' after one shift on A. Return True if and only if A can become B after some number of shifts on A.

Example 1:
Input: A = 'abcde', B = 'cdeab'
Output: true

Example 2:
Input: A = 'abcde', B = 'abced'
Output: false
Note:

A and B will have length at most 100.

'''

class Solution3:
    def rotateString(self, A: str, B: str) -> bool:
        # Just one line solution
        # Time: O(M*M) where M is the length of A or B
        # Space: O(M)
        return len(A) == len(B) and B in A + A
        
        
        # # This is also a possible solution but it basically does the same thing
        # if len(A) != len(B):
        #     return False
        # if not A:
        #     return True
        # lb = len(B)
        # A = A + A
        # for i in range(lb):
        #     if A[i:i+lb] == B:
        #         return True
        # return False


'''
193. Valid Phone Numbers
Easy

Given a text file file.txt that contains list of phone numbers (one per line), write a one liner bash script to print all valid phone numbers.

You may assume that a valid phone number must appear in one of the following two formats: (xxx) xxx-xxxx or xxx-xxx-xxxx. (x means a digit)

You may also assume each line in the text file must not contain leading or trailing white spaces.

Example:

Assume that file.txt has the following content:

987-123-4567
123 456 7890
(123) 456-7890
Your script should output the following valid phone numbers:

987-123-4567
(123) 456-7890

'''

# Solution4
# Read from the file file.txt and output all valid phone numbers to stdout.

# Use one of the command line tools (ex: grep) and regular expression to do the pattern matching
grep -P '^(\d{3}-|\(\d{3}\) )\d{3}-\d{4}$' file.txt


'''
Problem #975 was skipped because it was too hard.

'''


'''
482. License Key Formatting
Easy

You are given a license key represented as a string S which consists only alphanumeric character and dashes. The string is separated into N+1 groups by N dashes.

Given a number K, we would want to reformat the strings such that each group contains exactly K characters, except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.

Given a non-empty string S and a number K, format the string according to the rules described above.

Example 1:
Input: S = "5F3Z-2e-9-w", K = 4

Output: "5F3Z-2E9W"

Explanation: The string S has been split into two parts, each part has 4 characters.
Note that the two extra dashes are not needed and can be removed.
Example 2:
Input: S = "2-5g-3-J", K = 2

Output: "2-5G-3J"

Explanation: The string S has been split into three parts, each part has 2 characters except the first part as it could be shorter as mentioned above.
Note:
The length of string S will not exceed 12,000, and K is a positive integer.
String S consists only of alphanumerical characters (a-z and/or A-Z and/or 0-9) and dashes(-).
String S is non-empty.

'''

class Solution5:
    def licenseKeyFormatting(self, S: str, K: int) -> str:
        # split the string into a list of characters, then reverse order and remove '-', then rejoin into a string
        #   according to K, connecting with '-', and reverse the order again
        # Time: O(N)  Space: O(N)
        s_reverse = S.replace('-', '').upper()[::-1]
        return '-'.join([s_reverse[i:i+K] for i in range(0, len(s_reverse), K)])[::-1]


'''
904. Fruit Into Baskets
Medium

In a row of trees, the i-th tree produces fruit with type tree[i].

You start at any tree of your choice, then repeatedly perform the following steps:

Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, then step 2, then back to step 1, then step 2, and so on until you stop.

You have two baskets, and each basket can carry any quantity of fruit, but you want each basket to only carry one type of fruit each.

What is the total amount of fruit you can collect with this procedure?

 

Example 1:

Input: [1,2,1]
Output: 3
Explanation: We can collect [1,2,1].
Example 2:

Input: [0,1,2,2]
Output: 3
Explanation: We can collect [1,2,2].
If we started at the first tree, we would only collect [0, 1].
Example 3:

Input: [1,2,3,2,2]
Output: 4
Explanation: We can collect [2,3,2,2].
If we started at the first tree, we would only collect [1, 2].
Example 4:

Input: [3,3,3,1,2,1,1,2,3,3,4]
Output: 5
Explanation: We can collect [1,2,1,1,2].
If we started at the first tree or the eighth tree, we would only collect 4 fruits.
 

Note:

1 <= tree.length <= 40000
0 <= tree[i] < tree.length

'''

class Solution6:
    def totalFruit(self, tree: List[int]) -> int:
        # We need to scan through the list, since in this question, all we need to know is equivalent to 
        #   the longest sub list of the original list where numbers in it consist of only 2 different ones.
        #   because we can only carry two different kinds of fruits
        # Therefore, in this question, we need to scan through the list and keep track of the first occurance
        #   of the current 2 numbers, and update it if we see a new number. We also need last seen index to 
        #   update the other one once needed, since that will show the last point the other number strike got
        #   broken by the current number
        
        # Time: O(N)  Space: O(N)
        if len(tree) <= 2:
            return len(tree)
        
        # get the first 2 different fruits and its indexes
        f1, i1_first, i1_last = tree[0], 0, 0
        f2, i2_first, i2_last = -1, -1, -1
        for i in range(1, len(tree)):
            if tree[i] != f1:
                f2, i2_first, i2_last = tree[i], i, i
                i1_last = i - 1
                break
        
        if f2 == -1:
            return len(tree)
        
        start = i2_last
        res = i2_last - i1_first + 1
        
        for i in range(start + 1, len(tree)):
            if tree[i] == f1:
                i1_last = i
            elif tree[i] == f2:
                i2_last = i
            else:
                if tree[i-1] == f1:
                    # then fruit f2 is out of scope, update f2 and its first occurance index
                    f2 = tree[i]
                    i2_first = i
                    i1_first = i2_last + 1
                    i2_last = i
                else:
                    # then f1
                    f1 = tree[i]
                    i1_first = i
                    i2_first = i1_last + 1
                    i1_last = i
            res = max(res, i - min(i1_first, i2_first) + 1)
        
        return res


'''
807. Max Increase to Keep City Skyline
Medium

In a 2 dimensional array grid, each value grid[i][j] represents the height of a building located there. We are allowed to increase the height of any number of buildings, by any amount (the amounts can be different for different buildings). Height 0 is considered to be a building as well. 

At the end, the "skyline" when viewed from all four directions of the grid, i.e. top, bottom, left, and right, must be the same as the skyline of the original grid. A city's skyline is the outer contour of the rectangles formed by all the buildings when viewed from a distance. See the following example.

What is the maximum total sum that the height of the buildings can be increased?

Example:
Input: grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
Output: 35
Explanation: 
The grid is:
[ [3, 0, 8, 4], 
  [2, 4, 5, 7],
  [9, 2, 6, 3],
  [0, 3, 1, 0] ]

The skyline viewed from top or bottom is: [9, 4, 8, 7]
The skyline viewed from left or right is: [8, 7, 9, 3]

The grid after increasing the height of buildings without affecting skylines is:

gridNew = [ [8, 4, 8, 7],
            [7, 4, 7, 7],
            [9, 4, 8, 7],
            [3, 3, 3, 3] ]

Notes:

1 < grid.length = grid[0].length <= 50.
All heights grid[i][j] are in the range [0, 100].
All buildings in grid[i][j] occupy the entire grid cell: that is, they are a 1 x 1 x grid[i][j] rectangular prism.

'''

class Solution7:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        # for each building b at grid[i'][j'], it must make sure b < top[j] and b < left[i]
        
        # Time: O(N*N)  Space: O(N)
        ans = 0
        
        h = len(grid)
        w = len(grid[0])
        
        # Get the left/right and top/bottom view of skyline
        left = [max(row) for row in grid]
        top = [max([grid[i][j] for i in range(h)]) for j in range(w)]
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                ans += min(top[j], left[i]) - grid[i][j]
                
        return ans


'''
1119. Remove Vowels from a String
Easy

Given a string S, remove the vowels 'a', 'e', 'i', 'o', and 'u' from it, and return the new string.

 

Example 1:

Input: "leetcodeisacommunityforcoders"
Output: "ltcdscmmntyfrcdrs"
Example 2:

Input: "aeiou"
Output: ""
 

Note:

S consists of lowercase English letters only.
1 <= S.length <= 1000

'''

class Solution8:
    def removeVowels(self, S: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return ''.join([c for c in S if c not in vowels])


'''
143. Reorder List
Medium

Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:

Given 1->2->3->4, reorder it to 1->4->2->3.
Example 2:

Given 1->2->3->4->5, reorder it to 1->5->2->4->3.

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution9:
    # Time: O(N)  Space: O(1)
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # We can split the list into two smaller lists, then reverse the 2nd list, then merge two
        #   lists together
        
        if not head:
            return
        
        # Split:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # then the middle point is where slow is
        
        # Reverse 2nd list:
        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        
        # Merge 2 lists
        # prev is the node head of 2nd list
        p1, p2 = head, prev
        while p2.next:  # check p2.next because list 2 is always shorter or equal to the length of l1
            p1.next, p1 = p2, p1.next
            p2.next, p2 = p1, p2.next

            
'''
92. Reverse Linked List II
Medium

Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.

Example:

Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution10:
    # Time: O(N)  Space: O(1)
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        # first find the beginning point
        curr, prev = head, None
        
        while m > 1:
            curr, prev = curr.next, curr
            m, n = m - 1, n - 1
        
        # Save the two connection points for future use
        p1_end = prev
        p2_end = curr
        
        # Do the reverse
        while n:
            curr.next, prev, curr = prev, curr, curr.next
            n -= 1
            
        # Connect together
        if p1_end:
            p1_end.next = prev
        else:
            head = prev
        p2_end.next = curr
        
        return head

        
        