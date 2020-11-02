'''
Leetcode problems following the order of Nick White's playlist - VII

860. Lemonade Change
509. Fibonacci Number
746. Min Cost Climbing Stairs
908. Smallest Range I
752. Open the Lock
1004. Max Consecutive Ones III
27. Remove Element
424. Longest Repeating Character Replacement
724. Find Pivot Index
832. Flipping an Image

'''


'''
860. Lemonade Change
Easy

At a lemonade stand, each lemonade costs $5. 

Customers are standing in a queue to buy from you, and order one at a time (in the order specified by bills).

Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill.  You must provide the correct change to each customer, so that the net transaction is that the customer pays $5.

Note that you don't have any change in hand at first.

Return true if and only if you can provide every customer with correct change.

 

Example 1:

Input: [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we give a $10 bill and a $5 bill.
Since all customers got correct change, we output true.
Example 2:

Input: [5,5,10]
Output: true
Example 3:

Input: [10,10]
Output: false
Example 4:

Input: [5,5,10,10,20]
Output: false
Explanation: 
From the first two customers in order, we collect two $5 bills.
For the next two customers in order, we collect a $10 bill and give back a $5 bill.
For the last customer, we can't give change of $15 back because we only have two $10 bills.
Since not every customer received correct change, the answer is false.
 

Note:

0 <= bills.length <= 10000
bills[i] will be either 5, 10, or 20.

'''

class Solution1:
    def lemonadeChange(self, bills: [int]) -> bool:
        # Time: O(N)  Space: O(1)

        fives = tens = 0
        for b in bills:
            if b == 5:
                fives += 1
            elif b == 10:
                if not fives:
                    return False
                fives -= 1
                tens += 1
            else:
                if tens and fives:  # give tens changes first
                    tens -= 1
                    fives -= 1
                elif fives >= 3:
                    fives -= 3
                else:
                    return False
        return True


'''
509. Fibonacci Number
Easy

The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), for N > 1.
Given N, calculate F(N).

 

Example 1:

Input: 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.
Example 2:

Input: 3
Output: 2
Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.
Example 3:

Input: 4
Output: 3
Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.
 

Note:

0 ≤ N ≤ 30.

'''

class Solution2:
    def fib(self, N: int) -> int:
        # iteration bottom up
        # Time: O(N)  Space: O(1)
        if N <= 1:
            return N
        m = 0
        n = 1
        for i in range(2, N + 1):
            m, n = n, m + n
        return n


'''
746. Min Cost Climbing Stairs
Easy

On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

Example 1:
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
Example 2:
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
Note:
cost will have a length in the range [2, 1000].
Every cost[i] will be an integer in the range [0, 999].

'''

class Solution3:
    def minCostClimbingStairs(self, cost: [int]) -> int:
        # dynamic programming
        # Time: O(N)  Space: O(1
        if len(cost) <= 2:
            return min(cost)
        m, n = cost[0], cost[1]
        for i in range(2, len(cost)):
            m, n = n, min(m, n) + cost[i]
        return min(m, n)


'''
908. Smallest Range I
Easy

Given an array A of integers, for each integer A[i] we may choose any x with -K <= x <= K, and add x to A[i].

After this process, we have some array B.

Return the smallest possible difference between the maximum value of B and the minimum value of B.

 

Example 1:

Input: A = [1], K = 0
Output: 0
Explanation: B = [1]
Example 2:

Input: A = [0,10], K = 2
Output: 6
Explanation: B = [2,8]
Example 3:

Input: A = [1,3,6], K = 3
Output: 0
Explanation: B = [3,3,3] or B = [4,4,4]
 

Note:

1 <= A.length <= 10000
0 <= A[i] <= 10000
0 <= K <= 10000

'''

class Solution4:
    def smallestRangeI(self, A: [int], K: int) -> int:
        # Meaningless question, just math
        return max(0, max(A) - min(A) - 2 * K)


'''
752. Open the Lock
Medium

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.

 
Example 1:

Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".

Example 2:

Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation:
We can turn the last wheel in reverse to move from "0000" -> "0009".

Example 3:

Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
Explanation:
We can't reach the target without getting stuck.

Example 4:

Input: deadends = ["0000"], target = "8888"
Output: -1
 

Constraints:

1 <= deadends.length <= 500
deadends[i].length == 4
target.length == 4
target will not be in the list deadends.
target and deadends[i] consist of digits only.

'''

class Solution5:
    def openLock(self, deadends: [str], target: str) -> int:
        # BFS
        # We need a helper method to find all states that the lock can be rotated to, from the curr state
        # We use 'yield' to return a list of all neighbors:
        def neighbors(curr: str) -> [str]:
            for i in range(len(curr)):
                yield(curr[:i] + str((int(curr[i]) + 1) % 10) + curr[i+1:])
                yield(curr[:i] + str((int(curr[i]) - 1) % 10) + curr[i+1:])
        
        # We need a set of seen states, we can put deadends in it too, since they both mean we won't go there
        #   any more
        seen = set(deadends)
        if '0000' in seen:
            return -1
        
        # For BFS, we need a queue to represent the current layer
        q = collections.deque()
        q.append(('0000', 0))
        
        while q:
            curr, step = q.popleft()
            if curr == target:
                return step
            for n in neighbors(curr):
                if n not in seen:
                    q.append((n, step + 1))
                    seen.add(n)
                    
        return -1
        

'''
1004. Max Consecutive Ones III
Medium

Given an array A of 0s and 1s, we may change up to K values from 0 to 1.

Return the length of the longest (contiguous) subarray that contains only 1s. 


Example 1:

Input: A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
Output: 6
Explanation: 
[1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.

Example 2:

Input: A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
Output: 10
Explanation: 
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.
 

Note:

1 <= A.length <= 20000
0 <= K <= A.length
A[i] is 0 or 1 

'''

class Solution6:
    def longestOnes(self, A: [int], K: int) -> int:
        # Use a sliding window, use two pointers for the left and right most index of the window
        # Move right to the right until the window is not valid, then move left to the right
        #   until the window is back to valid, and keep tracking the length of the window
        
        # Time: O(N)  Space: O(1)
        left = right = 0
        res = 0
        zero_count = 0
        while right < len(A):
            if A[right] == 0:
                zero_count += 1
            while zero_count > K:
                if A[left] == 0:
                    zero_count -= 1
                left += 1
            res = max(res, right - left + 1)
            right += 1
        return res
            

'''
27. Remove Element
Easy

Given an array nums and a value val, remove all instances of that value in-place and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Clarification:

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by reference, which means a modification to the input array will be known to the caller as well.

Internally you can think of this:

// nums is passed in by reference. (i.e., without making a copy)
int len = removeElement(nums, val);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
 

Example 1:

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2]
Explanation: Your function should return length = 2, with the first two elements of nums being 2.
It doesn't matter what you leave beyond the returned length. For example if you return 2 with nums = [2,2,3,3] or nums = [2,3,0,0], your answer will be accepted.
Example 2:

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3]
Explanation: Your function should return length = 5, with the first five elements of nums containing 0, 1, 3, 0, and 4. Note that the order of those five elements can be arbitrary. It doesn't matter what values are set beyond the returned length.
 

Constraints:

0 <= nums.length <= 100
0 <= nums[i] <= 50
0 <= val <= 100

'''

class Solution7:
    def removeElement(self, nums: [int], val: int) -> int:
        # Use a pointer starting at index 0, then move to the right, and swap with the last
        #   element of the array every time we encounter the val, and remove the last element
        
        # Time: O(N)  Space: O(1)
        p = 0
        l = len(nums)
        while p < l:
            if nums[p] == val:
                nums[p] = nums[l - 1]
                l -= 1
            else:
                p += 1
        return l


'''
424. Longest Repeating Character Replacement
Medium

Given a string s that consists of only uppercase English letters, you can perform at most k operations on that string.

In one operation, you can choose any character of the string and change it to any other uppercase English character.

Find the length of the longest sub-string containing all repeating letters you can get after performing the above operations.

Note:
Both the string's length and k will not exceed 104.

Example 1:

Input:
s = "ABAB", k = 2

Output:
4

Explanation:
Replace the two 'A's with two 'B's or vice versa.
 

Example 2:

Input:
s = "AABABBA", k = 1

Output:
4

Explanation:
Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.

'''

class Solution8:
    def characterReplacement(self, s: str, k: int) -> int:
        # This is to find the max count of letters within a specific range from start to end, 
        #   then calculate the total number of different letters and move start to right if 
        #   it exceeds k
        
        # Time: O(N)  Space: O(1) since there are only 24 different letters
        max_count = start = end = 0
        counter = {}
        res = 0
        while end < len(s):
            counter[s[end]] = counter.get(s[end], 0) + 1
            max_count = max(max_count, counter[s[end]])
            if end - start + 1 - max_count > k:
                counter[s[start]] -= 1
                start += 1
            res = max(res, end - start + 1)
            end += 1
        return res
        

'''
724. Find Pivot Index
Easy

Given an array of integers nums, write a method that returns the "pivot" index of this array.

We define the pivot index as the index where the sum of all the numbers to the left of the index is equal to the sum of all the numbers to the right of the index.

If no such index exists, we should return -1. If there are multiple pivot indexes, you should return the left-most pivot index.

 

Example 1:

Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The sum of the numbers to the left of index 3 (nums[3] = 6) is equal to the sum of numbers to the right of index 3.
Also, 3 is the first index where this occurs.
Example 2:

Input: nums = [1,2,3]
Output: -1
Explanation:
There is no index that satisfies the conditions in the problem statement.
 

Constraints:

The length of nums will be in the range [0, 10000].
Each element nums[i] will be an integer in the range [-1000, 1000].


'''

class Solution9:
    def pivotIndex(self, nums: [int]) -> int:
        # find the total sum, and check prefix sum until number at index i
        # Time: O(N)  Space: O(1)
        # sum
        s = sum(nums)
        cur_sum = 0
        for i in range(len(nums)):
            if cur_sum == (s - nums[i]) / 2:
                return i
            cur_sum += nums[i]
        return -1


'''
832. Flipping an Image
Easy

Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting image.

To flip an image horizontally means that each row of the image is reversed.  For example, flipping [1, 1, 0] horizontally results in [0, 1, 1].

To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0. For example, inverting [0, 1, 1] results in [1, 0, 0].

Example 1:

Input: [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]
Example 2:

Input: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
Output: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
Explanation: First reverse each row: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]].
Then invert the image: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
Notes:

1 <= A.length = A[0].length <= 20
0 <= A[i][j] <= 1

'''

class Solution10:
    def flipAndInvertImage(self, A: [[int]]) -> [[int]]:
        # Just simply follow the logic, modify the matrix in place
        # Time: O(N) where N is number of elements in A
        # Space: O(1)
        h = len(A)
        w = len(A[0])
        for i in range(h):
            for j in range((w + 1) // 2):
                A[i][j], A[i][w - 1 - j] = 1 - A[i][w - 1 - j], 1- A[i][j]
        return A