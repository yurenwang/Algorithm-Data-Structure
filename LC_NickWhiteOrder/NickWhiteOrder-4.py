'''
Leetcode problems following the order of Nick White's playlist - IV

561. Array Partition I
1002. Find Common Characters
933. Number of Recent Calls
985. Sum of Even Numbers After Queries
922. Sort Array By Parity II
867. Transpose Matrix
209. Minimum Size Subarray Sum
916. Word Subsets
74. Search a 2D Matrix
402. Remove K Digits

'''

'''
561. Array Partition I
Easy

Given an array of 2n integers, your task is to group these integers into n pairs of integer, say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

Example 1:
Input: [1,4,3,2]

Output: 4
Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).
Note:
n is a positive integer, which is in the range of [1, 10000].
All the integers in the array will be in the range of [-10000, 10000].

'''

class Solution1:
    def arrayPairSum(self, nums: [int]) -> int:
        # Simply sort the list and get the result, super easy
        # Time: O(NlogN)  Space: O(1)
        nums.sort()
        s = 0
        for i in range(len(nums) // 2):
            s += nums[2 * i]
        return s


'''
1002. Find Common Characters
Easy

Given an array A of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).  For example, if a character occurs 3 times in all strings but not 4 times, you need to include that character three times in the final answer.

You may return the answer in any order.

 
Example 1:

Input: ["bella","label","roller"]
Output: ["e","l","l"]

Example 2:

Input: ["cool","lock","cook"]
Output: ["c","o"]
 

Note:

1 <= A.length <= 100
1 <= A[i].length <= 100
A[i][j] is a lowercase letter

'''

class Solution2:
    def commonChars(self, A: [str]) -> [str]:
        check = list(A[0])              # make the first string as our checking target
        for s in A[1:]:                 # loop through the rest of strings
            new_check = []
            for c in s:                 # loop through each char of a string
                if c in check:          
                    new_check.append(c) # add a char to a new check if it exists in the old check
                    check.remove(c)     # remove from old check since we want to know how many times it occurs
            check = new_check           # new check is the same chars between first two strings
        return check


'''
933. Number of Recent Calls
Easy

Write a class RecentCounter to count recent requests.

It has only one method: ping(int t), where t represents some time in milliseconds.

Return the number of pings that have been made from 3000 milliseconds ago until now.

Any ping with time in [t - 3000, t] will count, including the current ping.

It is guaranteed that every call to ping uses a strictly larger value of t than before.

 

Example 1:

Input: inputs = ["RecentCounter","ping","ping","ping","ping"], inputs = [[],[1],[100],[3001],[3002]]
Output: [null,1,2,3,3]
 

Note:

Each test case will have at most 10000 calls to ping.
Each test case will call ping with strictly increasing values of t.
Each call to ping will have 1 <= t <= 10^9.

'''

# Solution3
class RecentCounter:
    # Since time t is strictly increasing, we can use a queue and pop from begin to remove the earlier pings  
    # Time: O(N) where N is the number of queries made
    # Space: O(1) since 3000 is the max
    
    from collections import deque

    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)

# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)  


'''
985. Sum of Even Numbers After Queries
Easy

We have an array A of integers, and an array queries of queries.

For the i-th query val = queries[i][0], index = queries[i][1], we add val to A[index].  Then, the answer to the i-th query is the sum of the even values of A.

(Here, the given index = queries[i][1] is a 0-based index, and each query permanently modifies the array A.)

Return the answer to all queries.  Your answer array should have answer[i] as the answer to the i-th query.

 

Example 1:

Input: A = [1,2,3,4], queries = [[1,0],[-3,1],[-4,0],[2,3]]
Output: [8,6,2,4]
Explanation: 
At the beginning, the array is [1,2,3,4].
After adding 1 to A[0], the array is [2,2,3,4], and the sum of even values is 2 + 2 + 4 = 8.
After adding -3 to A[1], the array is [2,-1,3,4], and the sum of even values is 2 + 4 = 6.
After adding -4 to A[0], the array is [-2,-1,3,4], and the sum of even values is -2 + 4 = 2.
After adding 2 to A[3], the array is [-2,-1,3,6], and the sum of even values is -2 + 6 = 4.
 

Note:

1 <= A.length <= 10000
-10000 <= A[i] <= 10000
1 <= queries.length <= 10000
-10000 <= queries[i][0] <= 10000
0 <= queries[i][1] < A.length

'''

class Solution4:
    def sumEvenAfterQueries(self, A: [int], queries: [[int]]) -> [int]:
        # Brute force will take O(M*N) time, where M is len of A, and N is len of queries
        
        # We will use a trick to modify the sum directly. 
        # Each time we execute a query, we check the A[index] to see if that is an even number or not
        # If it is even, we subtract it from the sum
        # Then we check A[index] + new value and see if it is an even, if it is even, we add it to sum
        # Time: O(M+N)  Space: O(1)
        
        s = sum([x for x in A if x % 2 == 0])
        res = []
        for q in queries:
            old_val = A[q[1]]
            new_val = A[q[1]] + q[0]
            if old_val % 2 == 0:
                s -= old_val
            if new_val % 2 == 0:
                s += new_val
            A[q[1]] = new_val
            res.append(s)
        return res


'''
922. Sort Array By Parity II
Easy

Given an array A of non-negative integers, half of the integers in A are odd, and half of the integers are even.

Sort the array so that whenever A[i] is odd, i is odd; and whenever A[i] is even, i is even.

You may return any answer array that satisfies this condition.

 
Example 1:

Input: [4,2,5,7]
Output: [4,5,2,7]
Explanation: [4,7,2,5], [2,5,4,7], [2,7,4,5] would also have been accepted.
 

Note:

2 <= A.length <= 20000
A.length % 2 == 0
0 <= A[i] <= 1000

'''

class Solution5:
    def sortArrayByParityII(self, A: [int]) -> [int]:
        # Two pointers, in place
        # Time: O(N)  Space: O(1)
        even, odd = 0, 1
        while even < len(A) and odd < len(A):
            if A[even] % 2 == 1 and A[odd] % 2 == 0:    # when even pointer points to odd, odd pointer to even
                A[even], A[odd] = A[odd], A[even]
                even += 2
                odd += 2
            # when both point to odd, then odd pointer is fine and can move to next odd index
            elif A[even] % 2 == 1 and A[odd] % 2 == 1:   
                odd += 2
            elif A[even] % 2 == 0 and A[odd] % 2 == 0:  # vise versa
                even += 2
            else:                   # this is when both are fine, we move both pointer
                even += 2
                odd += 2
        return A


'''
867. Transpose Matrix
Easy

Given a matrix A, return the transpose of A.

The transpose of a matrix is the matrix flipped over it's main diagonal, switching the row and column indices of the matrix.

(For images go to https://leetcode.com/problems/transpose-matrix/)


Example 1:

Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]

Example 2:

Input: [[1,2,3],[4,5,6]]
Output: [[1,4],[2,5],[3,6]]
 

Note:

1 <= A.length <= 1000
1 <= A[0].length <= 1000

'''

class Solution6:
    def transpose(self, A: [[int]]) -> [[int]]:
        # create a new matrix M, and for each cell A[i][j] in matrix A, put it 
        #   at place M[j][i], and return M
        # Time: O(MN)  Space: O(1)
        
        H = len(A)
        W = len(A[0])
        M = [[0] * H for _ in range(W)]
        for i in range(H):
            for j in range(W):
                M[j][i] = A[i][j]
        return M


'''
209. Minimum Size Subarray Sum
Medium

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example: 

Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n). 

'''

class Solution7:
    def minSubArrayLen(self, s: int, nums: [int]) -> int:
        # Two pointers and modify curr_sum dynamically to avoid consuming unnecessary time
        # Time: O(N)  Space: O(1)
        if not nums:
            return 0
        l = r = 0
        res = len(nums) + 1
        curr_sum = nums[0]
        while l < len(nums) and r < len(nums):
            if curr_sum >= s:
                res = min(res, r - l + 1)
                curr_sum -= nums[l]
                l += 1
            else:
                r += 1
                if r < len(nums): 
                    curr_sum += nums[r]
        return res if res <= len(nums) else 0


'''
916. Word Subsets
Medium

We are given two arrays A and B of words.  Each word is a string of lowercase letters.

Now, say that word b is a subset of word a if every letter in b occurs in a, including multiplicity.  For example, "wrr" is a subset of "warrior", but is not a subset of "world".

Now say a word a from A is universal if for every b in B, b is a subset of a. 

Return a list of all universal words in A.  You can return the words in any order.

 
Example 1:

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","o"]
Output: ["facebook","google","leetcode"]

Example 2:

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["l","e"]
Output: ["apple","google","leetcode"]

Example 3:

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","oo"]
Output: ["facebook","google"]

Example 4:

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["lo","eo"]
Output: ["google","leetcode"]

Example 5:

Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["ec","oc","ceo"]
Output: ["facebook","leetcode"]
 

Note:

1 <= A.length, B.length <= 10000
1 <= A[i].length, B[i].length <= 10
A[i] and B[i] consist only of lowercase letters.
All words in A[i] are unique: there isn't i != j with A[i] == A[j].

'''

class Solution8:
    def wordSubsets(self, A: [str], B: [str]) -> [str]:
        # Since a universal word needs to be superset for all words in B, we can combine all words in B
        #   to become a new word, and words in A only need to have this one single word as its subset
        # Time: O(M + N)  Space: O(1)
        
        # get a list of ints of length 26 to represent how many of each letter is in a word
        def countLetter(s):
            res = [0] * 26
            for c in s:
                res[ord(c) - ord('a')] += 1
            return res
        
        # get how many of each letter is in the combined letter of B, note that we only need the larger
        #   number if one char appears in two different words. Ex: 'lo' and 'loo' will become 'loo'
        b_count = [0] * 26
        for b in B:
            for i, c in enumerate(countLetter(b)):
                b_count[i] = max(b_count[i], c)
        
        # get all words x with count(x) greater than b_count in every character
        return [x for x in A if all([x >= y for x, y in zip(countLetter(x), b_count)])]


'''
74. Search a 2D Matrix
Medium

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
Example 1:

Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
Example 2:

Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
Output: false

'''

class Solution9:
    def searchMatrix(self, matrix: [[int]], target: int) -> bool:
        # This whole matrix can be stretched to a sorted list, so we can use binary search for finding the target
        # Time: O(logMN)  Space: O(1)
        
        if not matrix:
            return False
        row, col = len(matrix), len(matrix[0])
        l, r = 0, row * col - 1
        
        while l <= r:
            m = (l + r) // 2
            m_val = matrix[m // col][m % col]
            if m_val == target:
                return True
            elif m_val > target:
                r = m - 1
            else:
                l = m + 1
                
        return False


'''
402. Remove K Digits
Medium

Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.

Note:
The length of num is less than 10002 and will be ≥ k.
The given num does not contain any leading zero.
Example 1:

Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
Example 2:

Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
Example 3:

Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.

'''

class Solution10:
    def removeKdigits(self, num: str, k: int) -> str:
        # Give two integers of same length, A = 1axxx, B = 1bxxx, it is the left most difference that decide which one
        #   is bigger. Ex: we don't care about the xxx in the end, as long as a > b, A must be bigger than B
        # Therefore, we can scan from left to right on the string and remove any digit that is bigger than the next 
        #   digit and use a stack to remove all of them
        # Time: O(N)  Space: O(N)
        
        stack = []
        for c in num:
            while k > 0 and stack and stack[-1] > c:
                stack.pop()
                k -= 1
            stack.append(c)
        
        # if there's k left, simply trunk the k last digits, as they have to be larger than any digits in front of them
        res = stack[:-k] if k else stack
        
        return ''.join(res).lstrip('0') or '0'