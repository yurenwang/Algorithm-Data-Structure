'''
########################### Dynamic Programming - Part I ############################
### This is part 1 of DP problems, which covers Fibonacci to Common SubSequence

70. Climbing Stairs
198. House Robber
213. House Robber II
4. 信件错排
5. 母牛生产
64. Minimum Path Sum
62. Unique Paths
303. Range Sum Query - Immutable
413. Arithmetic Slices
343. Integer Break
279. Perfect Squares
91. Decode Ways
300. Longest Increasing Subsequence
646. Maximum Length of Pair Chain
376. Wiggle Subsequence
1143. Longest Common Subsequence

'''

##################################### 2D Fibonacci like problems #####

'''
70. Climbing Stairs
Easy

You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:

Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
'''

# Dynamic Programming (actually this is same as Fibonacci)
def climbStairs(n: int):
	ways = [0, 1, 2]
	if n <= 2:
		return n
	for i in range(3, n + 1):
		ways.append(sum(ways[-2:]))
	
	return ways.pop()

print(climbStairs(3))
print(climbStairs(5))


'''
198. House Robber
Easy

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.
'''
# Dynamic programming
def rob(nums: [int]):
	max2IndexAgo = 0
	max1IndexAgo = 0
	for i in nums:
		currMax = max(max2IndexAgo + i, max1IndexAgo)
		max2IndexAgo = max1IndexAgo
		max1IndexAgo = currMax
	return max1IndexAgo # returned max1IndexAgo instead of currMax so that the algo will work with empty list as input

print(rob([2,7,9,3,1]))


'''
213. House Robber II
Medium

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
             because they are adjacent houses.
Example 2:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
'''

# Dynamic Programming

# In this question, we want to make use of the previous rob1 solution of robbing a single line of houses
# However, since we are not having a circle, we can assume one house that is not robbed, and break the circle 
# For example, for houses 1 -> 2 -> 3 -> 4 -> 1, we can break down to 2 -> 3 -> 4, if we assume 1 is not robbed
# Similarly, we can assume the another house next to 1 is not robbed for another try, and get the max of these 2

# Time: O(n)  Space: O(1)
def rob2(nums: [int]):
	if len(nums) == 1:
		return nums[0]

	# this is the helper function to rob a single line of houses, starting with first, ending with last
	def robHelper(first, last):
		max2IndexAgo = 0
		max1IndexAgo = 0
		for i in nums[first : last + 1]:
			currMax = max(max2IndexAgo + i, max1IndexAgo)
			max2IndexAgo = max1IndexAgo
			max1IndexAgo = currMax
		return max1IndexAgo
	
	return max(robHelper(0, len(nums) - 2), robHelper(1, len(nums) - 1))

print(rob2([2,3,2]))
print(rob2([1,2,3,1]))


'''
4. 信件错排
题目描述：有 N 个 信 和 信封，它们被打乱，求错误装信方式的数量（所有信封都没有装各自的信）。

定义一个数组 dp 存储错误方式数量，dp[i] 表示前 i 个信和信封的错误方式数量。假设第 i 个信装到第 j 个信封里面，而第 j 个信装到第 k 个信封里面。根据 i 和 k 是否相等，有两种情况：

i==k，交换 i 和 j 的信后，它们的信和信封在正确的位置，但是其余 i-2 封信有 dp[i-2] 种错误装信的方式。由于 j 有 i-1 种取值，因此共有 (i-1)*dp[i-2] 种错误装信方式。
i != k，交换 i 和 j 的信后，第 i 个信和信封在正确的位置，其余 i-1 封信有 dp[i-1] 种错误装信方式。由于 j 有 i-1 种取值，因此共有 (i-1)*dp[i-1] 种错误装信方式。
综上所述，错误装信数量方式数量为：

'''


'''
5. 母牛生产
程序员代码面试指南-P181

题目描述：假设农场中成熟的母牛每年都会生 1 头小母牛，并且永远不会死。第一年有 1 只小母牛，从第二年开始，母牛开始生小母牛。每只小母牛 3 年之后成熟又可以生小母牛。给定整数 N，求 N 年后牛的数量。

第 i 年成熟的牛的数量为：

'''


##################################### Matrix #####

'''
64. Minimum Path Sum
Medium

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
'''

# Normally, this question should be solved using a 2D DP, which uses O(mn) time and space. 
# However, I want to reduce the usage of space to O(m), where m is the length of each row.  
# The reason behind is that we only move down or right so we only need to know the minPathSum of the cell on top of curr. 

# Dynamic Programming
# Time: O(M*N)  Space: O(N) where M is num of rows, N is num of cols
def minPathSum(grid: [[int]]):
	if len(grid) == 0 or len(grid[0]) == 0:
		return 0
	# Initialize a list to store the min steps needed for the current row.
	currMin = [None] * len(grid[0])

	for i in range(len(grid)):
		for j in range(len(grid[i])):
			curr = grid[i][j]
			# update the min steps of current row based on the cell left of it and top of it
			if i == 0:
				currMin[j] = curr if j == 0 else currMin[j - 1] + curr
			else:
				currMin[j] = currMin[j] + curr if j == 0 else min(currMin[j - 1], currMin[j]) + curr
	return currMin.pop()

print(minPathSum([[1,3,1],[1,5,1],[4,2,1]])) # expect 7, which is 1->3->1->1->1
			

'''
62. Unique Paths
Medium

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

Above is a 7 x 3 grid. How many possible unique paths are there?
(For the graph, see the original question at https://leetcode.com/problems/unique-paths/)

Example 1:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
Example 2:

Input: m = 7, n = 3
Output: 28
 

Constraints:

1 <= m, n <= 100
It's guaranteed that the answer will be less than or equal to 2 * 10 ^ 9.
'''
# Dynamic Programming
def uniquePaths( m: int, n: int) -> int:
	numOfPaths = [[1] * n for _ in range(m)]
	for i in range(m):
		for j in range(n):
			if i == 0 or j == 0:
				numOfPaths[i][j] = 1
			else:
				numOfPaths[i][j] = numOfPaths[i - 1][j] + numOfPaths[i][j - 1]
	return numOfPaths[m - 1][n - 1]


################################### Range Sum Query #####

'''
303. Range Sum Query - Immutable
Easy

Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
Note:
You may assume that the array does not change.
There are many calls to sumRange function.
'''
# Time: O(n) for preprocess, and O(1) when calling. Space: O(n)
class NumArray:
    sumTil = []
    def __init__(self, nums: [int]):
        # preprocess the list to stored the sum from index 0 to k for each item
        self.sumTil = [sum(nums[0:i+1]) for i in range(len(nums))]
        
    # since we have preprocessed the list, we can just subtract sumTil(i-1) from sumTil(j) for the result
    def sumRange(self, i: int, j: int) -> int:
        return self.sumTil[j] - self.sumTil[i-1] if i != 0 else self.sumTil[j]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)


'''
413. Arithmetic Slices
Medium

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

For example, these are arithmetic sequence:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
The following sequence is not arithmetic.

1, 1, 2, 5, 7

A zero-indexed array A consisting of N numbers is given. A slice of that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular, this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.


Example:

A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
'''
# A common solution is to use 2d array dynamic programming. However, as we only use the last item
# for the comparison, we can actually use a constance space dynamic programming. 

# Time: O(n)  Space: O(1)
def numberOfArithmeticSlices(self, A: [int]):
	if len(A) < 3:
		return 0
	result = 0
	curr = 0
	for i in range(2, len(A)):
		if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
			curr += 1
			result += curr
		else: 
			curr = 0
	return result


###################################### Integer Break #####

'''
343. Integer Break
Medium

Given a positive integer n, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get.

Example 1:

Input: 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
Example 2:

Input: 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
Note: You may assume that n is not less than 2 and not larger than 58.
'''

# Below is the logic:
'''
We can find a pattern in num of 2's and 3's in the split:
n	Maximum Product			# of 2's	# of 3's
2	1 x 1 = 1				0			0
3	1 x 2 = 2				1			0
4	2 x 2 = 4				2			0
5	2 x 3 = 6				1			1
6	3 x 3 = 9				0			2
7	2 x 2 x 3 = 12			2			1
8	2 x 3 x 3 = 18			1			2
9	3 x 3 x 3 = 27			0			3
10	2 x 2 x 3 x 3 = 36		2			2
11	2 x 3 x 3 x 3 = 54		1			3
12	3 x 3 x 3 x 3 = 81		0			4
13	2 x 2 x 3 x 3 x 3 = 108	2			3
14	2 x 3 x 3 x 3 x 3 = 162	1			4
15	3 x 3 x 3 x 3 x 3 = 243	0			5

*** For DP solution, we can find out that for n >= 7, dp[n] == 3*dp[n-3]
*** For math solution, we can find out that there is a pattern of num of 2's and 3's depends on n % 3
'''

# Dynamic Programming solution:
# Time: O(n)  Space: O(n)
def integerBreak(n: int):
	# first define the result if n < 7
	results = [0, 0, 1, 2, 4, 6, 9]
	if n < 7: return results[n] 

	# for n >= 7, the result is always 3 times the result of n - 3
	for i in range(7, n + 1):
		results.append(3 * results[i - 3])
	
	return results[-1]

print(integerBreak(10))


'''
279. Perfect Squares

This Question was previously solved using BFS.

We can also use dynamic programming to solve this problem. The algorithm behind it is:
- As for almost all DP solutions, we first create an array dp of one or multiple dimensions to hold the values of intermediate sub-solutions, as well as the final solution which is usually the last element in the array. Note that, we create a fictional element dp[0]=0 to simplify the logic, which helps in the case that the remainder (n-k) happens to be a square number.
- As an additional preparation step, we pre-calculate a list of square numbers (i.e. square_nums) that is less than the given number n.
- As the main step, we then loop from the number 1 to n, to calculate the solution for each number i (i.e. numSquares(i)). At each iteration, we keep the result of numSquares(i) in dp[i], while resuing the previous results stored in the array.
- At the end of the loop, we then return the last element in the array as the result of the solution.

In the main step, dp[i] = min(dp[i-1], dp[i-4], dp[i-9], ...) + 1

'''


'''
91. Decode Ways
Medium

A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example 1:

Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

'''
# Easy DP
def numDecodings(s: str):
	results = [0, 1]
	for i in range(0, len(s)):
		if i == 0:
			if s[i] == '0': return 0
			results.append(1)
			continue  
		if s[i] == '0':
			if s[i - 1] in ['1', '2']:
				results.append(results[i]) # did not put i - 2 here as results are shifted 2 digits to the right
			else:
				return 0
		else:
			if int(s[i - 1 : i + 1]) <= 9:
				results.append(results[i + 1]) # same here
			elif int(s[i - 1 : i + 1]) <= 26:
				results.append(results[i + 1] + results[i]) # same here
			else:
				results.append(results[i + 1]) # same here
	return results[-1]

print(numDecodings("226"))
print(numDecodings("236"))


######################################### Longest Increasing Sub-List #####

'''
300. Longest Increasing Subsequence
Medium

Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity?
'''
# DP
# Time: O(n^2) Space: O(n)
def lengthOfLIS(nums: [int]):
	if len(nums) == 0: return 0 
	dp = [1] * len(nums)
	result = 1
	for i in range(1, len(nums)):
		maxBefore = 0
		for j in range(0, i):
			if nums[i] > nums[j]:
				maxBefore = max(maxBefore, dp[j])
		dp[i] = maxBefore + 1
		result = max(result, dp[i])
	return result


'''
646. Maximum Length of Pair Chain
Medium

You are given n pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair (c, d) can follow another pair (a, b) if and only if b < c. Chain of pairs can be formed in this fashion.

Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.

Example 1:
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]
Note:
The number of given pairs will be in the range [1, 1000].
'''

# The normal way of solving this question is to use dynamic programming, which takes O(N^2) time.  

# However, there is a more clever way to solve it: Sort it depends on the 2nd item in the pairs, then solve it
#	with the greedy manner. Sort with 2nd item is to make sure in our greedy model, we have the current item as small
#	as possible. Below is the solution:

# Time: O(N log(N)) Space: O(1)
import operator

def findLongestChain(pairs: [[int]]):
	pairs.sort(key=operator.itemgetter(1)) # or you can use lambda x: x[1] for the key (which is slightly slower)
	curr = float("-inf")
	result = 0
	for x, y in pairs:
		if x > curr:
			curr = y
			result += 1
	return result

print(findLongestChain([[1,2], [2,3], [3,4]]))


'''
376. Wiggle Subsequence
Medium

A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with fewer than two elements is trivially a wiggle sequence.

For example, [1,7,4,9,2,5] is a wiggle sequence because the differences (6,-3,5,-7,3) are alternately positive and negative. In contrast, [1,4,7,2,5] and [1,7,4,5,5] are not wiggle sequences, the first because its first two differences are positive and the second because its last difference is zero.

Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence. A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence, leaving the remaining elements in their original order.

Example 1:

Input: [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence.
Example 2:

Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
Explanation: There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].
Example 3:

Input: [1,2,3,4,5,6,7,8,9]
Output: 2
'''
# A common way to solve this would be to use a regular dynamic programming. Which takes O(n^2) time

# We will use a 1 pass dynamic programming, and use two lists, up, and down, to note down the max length we currently
# 	have ending with a rising end, or a decreasing one 

# For the 1-pass linear dp, complexity is Time: O(n)  Space: O(n)
# We can actually optimize the space usage as we only use the last item of up and down array. We can store them as 
#	two variables instead. Please see below solution 2 for it.
def wiggleMaxLength(nums: [int]):
	if len(nums) == 0: return 0
	up = [1] * len(nums)  
	down = [1] * len(nums)  
	for i in range(1, len(nums)):
		if nums[i] < nums[i - 1]:
			down[i] = up[i - 1] + 1
			up[i] = up[i - 1]
		elif nums[i] > nums[i - 1]:
			down[i] = down[i - 1]
			up[i] = down[i - 1] + 1
		else:
			down[i] = down[i - 1]
			up[i] = up[i - 1] 
	return max(up[-1], down[-1])

print(wiggleMaxLength([1,17,5,10,13,15,10,5,16,8]))

# Optimized space usage
# Time: O(n)  Space: O(1)
def wiggleMaxLength2(nums: [int]):
	if len(nums) == 0: return 0
	up = 1 
	down = 1
	for i in range(1, len(nums)):
		if nums[i] < nums[i - 1]:
			down = up + 1
		elif nums[i] > nums[i - 1]:
			up = down + 1
	return max(up, down)

print(wiggleMaxLength2([1,17,5,10,13,15,10,5,16,8]))


############################################## Longest public sub-sequence #####

'''
1143. Longest Common Subsequence
Medium

Given two strings text1 and text2, return the length of their longest common subsequence.

A subsequence of a string is a new string generated from the original string with some characters(can be none) deleted without changing the relative order of the remaining characters. (eg, "ace" is a subsequence of "abcde" while "aec" is not). A common subsequence of two strings is a subsequence that is common to both strings.

If there is no common subsequence, return 0.


Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
 
Constraints:

1 <= text1.length <= 1000
1 <= text2.length <= 1000
The input strings consist of lowercase English characters only.

'''

# We want to create a 2-D list for our DP. DP[i][j] means the longest common subsequence until text1[i] and text2[j]
# if text1[i] and text2[j] is the same, that means we can continue our subsequence. Otherwise, we get the larger one 
# possible and put it at DP[i][j]

# Time: O(mn)  Space: O(mn)
def longestCommonSubsequence(text1: str, text2: str):
	if len(text1) == 0 or len(text2) == 0: 
		return 0
	# expanded the dp list by 1 cell because we want to use it to calculate the values when i or j equals 0
	dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
	for i in range(len(text1)):
		for j in range(len(text2)):
			if text1[i] == text2[j]:
				# if we find a match, we can add 1 to previous result
				dp[i + 1][j + 1] = dp[i][j] + 1
			else:
				# if no match, curr result would be the larger of the cell above it or on the left of it 
				dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
	return dp[-1][-1]

print(longestCommonSubsequence(text1 = "abcde", text2 = "ace"))


