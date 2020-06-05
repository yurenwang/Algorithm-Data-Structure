'''
########################################## Backtracking ########################
17. Letter Combinations of a Phone Number
93. Restore IP Addresses
79. Word Search
257. Binary Tree Paths
46. Permutations
47. Permutations II
77. Combinations
39. Combination Sum
40. Combination Sum II
216. Combination Sum III
78. Subsets
90. Subsets II
131. Palindrome Partitioning
37. Sudoku Solver (TODO)
51. N-Queens (TODO)

'''

'''
17. Letter Combinations of a Phone Number
Medium

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
Note:

Although the above answer is in lexicographical order, your answer could be in any order you want.

'''
# I used itertools.product to find all combinations and used join to join them together then return
# I did not use backtracking for this specific question

# Time: O(3^m * 4^n) where m is the number of 2,3,4,5,6,8 in the digits, and n is num of 7 or 9
def letterCombinations(digits: str):
	phone = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'],
			'5': ['j', 'k', 'l'], '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'],
			'8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}	
	from itertools import product
	return [] if len(digits) == 0 else [''.join(i) for i in list(product(*[l for l in [phone[d] for d in digits]]))]

print(letterCombinations("23"))


'''
93. Restore IP Addresses
Medium

Given a string containing only digits, restore it by returning all possible valid IP address combinations.

Example:

Input: "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]

'''
# This question is very complex

# Use backtrack to recursively go into the string, put '.' in if it can form a valid interval, then continue
#	Once hit an end point (either success, in which case we append to the result, or fail, in which case we do nothing), 
#   We remove the last positioned '.' and position it at the next place to check. We do this until we finish all.

# Time: O(27), or O(1). We have at most 27 different options to check
# Space: Also constant as we have at most 19 result
def restoreIpAddresses(s: str):

	# This is the recursive call to backtrack starting at the current positions list and the current remaining num of '.'
	def backtrack(positions, remainder):
		prev = positions[len(positions) - 1]
		for curr in range(prev + 1, min(len(s), prev + 4)):
			if isValid(prev, curr):
				# Append curr to positions if it forms a valid interval
				positions.append(curr)
				if remainder == 1:
					# Check the remaining part of the string to see if everything is correct, update result if yes
					if isValid(curr, len(s)):
						updateResult(positions)
				else: 
					# if curr forms a valid interval but we haven't reach the end, simply continue
					backtrack(positions, remainder - 1)
				# We pop the last '.' out in order to backtrack, and use another position for '.'
				positions.pop()

	# Helper function to check if the current interval is valid
	def isValid(prev, curr):
		if curr - prev == 1:
			return True
		elif curr - prev == 2 and 10 <= int(s[prev:curr]) <= 99:
			return True
		elif curr - prev == 3 and 100 <= int(s[prev:curr]) <= 255:
			return True
		else:
			return False

	# Helper function to update the result
	def updateResult(positions):
		parts = [s[positions[i]:positions[i+1]] for i in range(3)]
		parts.append(s[positions[3]:])
		result.append('.'.join(parts))

	result = []
	positions = [0]

	# Starting at initial positions, and 3 remaining dots
	backtrack(positions, 3)

	return result

print(restoreIpAddresses('25525511135'))


'''
79. Word Search
Medium

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
 

Constraints:

board and word consists only of lowercase and uppercase English letters.
1 <= board.length <= 200
1 <= board[i].length <= 200
1 <= word.length <= 10^3

'''

# Backtracking

# iterate through the board to find a start point, then for each start point, backtrack until find a match

# I believe my solution should work, but it exceeds the time limit on Leetcode

# Time: O(N * 4^L) where N is the number of items in the board, and L is the length of the target word 
# Space: O(N) because the max call stack length would be N
def exist(board: [[str]], word: str):
	result = []
	# interative method to backtrack, beginning with the trace that's already done, and the remainder of the word
	def backtrack(i, j, remainder, visited = set()):
		if len(remainder) == 0:
			result.append(1)
		# check each cell next to current one
		for r, c in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
			# if a cell is valid, put it in the trace, then continue backtracking the next element, then pop out the next item
			if 0 <= r < len(board) and 0 <= c < len(board[0]) and (r, c) not in visited and len(remainder) > 0 and board[r][c] == remainder[0]:
				visited.add((r, c))
				# return True if this is already the last item to check
				if len(remainder) == 1:
					result.append(1)
				backtrack(r, c, remainder[1:], visited)
				visited.remove((r, c))
	
	# go through the whole board for the starting point
	for i in range(len(board)):
		for j in range(len(board[0])):
			if board[i][j] == word[0]:
				backtrack(i, j, word[1:], {(i, j)})

	return len(result) > 0

print(exist([
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
], 'ABCCED'))
print(exist([["a"]], "a"))
print(exist([["C","A","A"],["A","A","A"],["B","C","D"]], "AAB"))


'''
257. Binary Tree Paths
Easy

Given a binary tree, return all root-to-leaf paths.

Note: A leaf is a node with no children.

Example:

Input:

   1
 /   \
2     3
 \
  5

Output: ["1->2->5", "1->3"]

Explanation: All root-to-leaf paths are: 1->2->5, 1->3

'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def binaryTreePaths(root: TreeNode):
	if root == None:
		return []
	result = []

	def backtrack(curr, tmpResult = []):
		tmpResult.append(str(curr.val))
		if curr.left == None and curr.right == None:
			result.append('->'.join(tmpResult))
		if curr.left != None:
			backtrack(curr.left, tmpResult)
		if curr.right != None:
			backtrack(curr.right, tmpResult)
		tmpResult.pop()
	
	backtrack(root)

	return result


'''
46. Permutations
Medium

Given a collection of distinct integers, return all possible permutations.

Example:

Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

'''

# Classic Backtracking solution
def permute(nums: []):
	result = []

	def backtrack(tmpResult):
		for i in nums:
			if i not in tmpResult:
				tmpResult.append(i)
				if len(tmpResult) == len(nums):
					result.append(tmpResult[:])
				else: 
					backtrack(tmpResult)
				tmpResult.pop()
	
	backtrack([])
	return result

print(permute([1,2,3]))


'''
47. Permutations II
Medium

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Example:

Input: [1,1,2]
Output:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

'''
# Backtracking

# Sort the list first to find out all duplicates 
# Then, do backtracking, for each item, if it is the same as the prev one, we know that it is a duplicate, then it can 
#	only participate in the backtracking after the prev one is added. In this way we ensure we don't add them in twice.

# Time: O(N^2) Space: O(N)
def permuteUnique(nums: []):
	result = []
	nums.sort()

	def backtrack(tmpResult):
		for i in range(len(nums)):
			if (i == 0 or nums[i] != nums[i-1] or (nums[i] == nums[i-1] and i-1 in tmpResult)) and i not in tmpResult:
				tmpResult.append(i)
				if len(tmpResult) == len(nums):
					result.append([nums[x] for x in tmpResult])
				else: 
					backtrack(tmpResult)
				tmpResult.pop()

	backtrack([])
	return result

print(permuteUnique([1,1,2]))


'''
77. Combinations
Medium

Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

Example:

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

'''
# Backtracking

def combine(n: int, k: int):
	result = []

	# for the recursion part, we used i+1 for the next first, so that our recursion will always go towards the 
	#	end of the int list, so that we won't have any duplications
	def backtrack(first, tmpResult):
		if len(tmpResult) == k:
			result.append(tmpResult[:])
		for i in range(first, n + 1):
			if i not in tmpResult:
				tmpResult.append(i)
				backtrack(i + 1, tmpResult)
				tmpResult.pop()
	
	backtrack(1, [])
	return result

print(combine(4, 2))


'''
39. Combination Sum
Medium

Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
Example 2:

Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
'''
# Simple backtracking
def combinationSum(candidates: [int], target: int):
	result = []
	l = len(candidates)

	def backtrack(first, tmpResult):
		if sum(tmpResult) == target:
			result.append(tmpResult[:])
		if sum(tmpResult) < target:
			for i in range(first, l):
				tmpResult.append(candidates[i])
				backtrack(i, tmpResult)
				tmpResult.pop()
	
	backtrack(0, [])
	return result

print(combinationSum([2,3,6,7], 7))


'''
40. Combination Sum II
Medium

Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

Each number in candidates may only be used once in the combination.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5,
A solution set is:
[
  [1,2,2],
  [5]
]
'''
# Backtracking 
def combinationSum2(candidates: [int], target: int): 
	result = []
	candidates.sort()

	def backtrack(first, tmpIndex, tmpValue):
		if sum(tmpValue) == target: 
			result.append(tmpValue[:])
		if sum(tmpValue) < target:
			for i in range(first, len(candidates)):
				if i == 0 or candidates[i] != candidates[i - 1] or (candidates[i] == candidates[i - 1] and i - 1 in tmpIndex):
					tmpIndex.append(i)
					tmpValue.append(candidates[i])
					backtrack(i + 1, tmpIndex, tmpValue)
					tmpIndex.pop()
					tmpValue.pop()
		
	backtrack(0, [], [])
	return result 

# print(combinationSum2([14,6,25,9,30,20,33,34,28,30,16,12,31,9,9,12,34,16,25,32,8,7,30,12,33,20,21,29,24,17,27,34,11,17,30,6,32,21,27,17,16,8,24,12,12,28,11,33,10,32,22,13,34,18,12],
# 27))


'''
216. Combination Sum III
Medium

Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Note:

All numbers will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: k = 3, n = 7
Output: [[1,2,4]]
Example 2:

Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
'''
# Backtracking
def combinationSum3(k: int, n: int):
	result = []

	def backtrack(first, tmpResult):
		if len(tmpResult) == k and sum(tmpResult) == n:
			result.append(tmpResult[:])
		if len(tmpResult) > k or sum(tmpResult) > n:
			return
		for i in range(first, 10):
			tmpResult.append(i)
			backtrack(i + 1, tmpResult)
			tmpResult.pop()

	backtrack(1, [])
	return result

print(combinationSum3(3, 7))


'''
78. Subsets
Medium

Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:

Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
'''
# Backtracking
def subsets(nums: [int]):
	result = [[]]

	def backtrack(first, end, tmpResult):
		if first > end:
			result.append(tmpResult[:])
		for i in range(first, end + 1):
			# tmpResult.append(nums[i])
			backtrack(i + 1, end, tmpResult + [nums[i]]) # We can eliminate the use of append and pop by concatenating the tmpResult like this
			# tmpResult.pop()

	# We need to do backtracing for different endings in order to get all results
	for i in range(len(nums)):
		backtrack(0, i, [])
	return result

print(subsets([1,2,3]))


'''
90. Subsets II
Medium

Given a collection of integers that might contain duplicates, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:

Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
'''

# Backtracking
def subsetsWithDup(nums: [int]):
	result = [[]]
	nums.sort()

	def backtrack(first, end, tmpResult, tmpIndex):
		if first > end:
			result.append(tmpResult[:])
		for i in range(first, end + 1):
			if i == 0 or nums[i] != nums[i - 1] or nums[i] == nums[i - 1] and i - 1 in tmpIndex:
				# tmpResult.append(nums[i])
				# tmpIndex.append(i)
				backtrack(i + 1, end, tmpResult + [nums[i]], tmpIndex + [i]) # concatenate in the call works the same as using append and pop methods above and below
				# tmpResult.pop()
				# tmpIndex.pop()

	# We need to do backtracing for different endings in order to get all results
	for i in range(len(nums)):
		backtrack(0, i, [], [])
	return result

print('90. Subsets II')
print(subsetsWithDup([1,2,2]))
print(subsetsWithDup([5,5,5,5,5]))


'''
131. Palindrome Partitioning
Medium

Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:

Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
'''
# Backtracking
def partition(s: str):
	result = []

	def isPal(str): # check if a given string is palindrome
		return str == str[::-1]
	
	def backtrack(first, tmpResult):
		if first >= len(s):
			result.append(tmpResult)
		for i in range(first, len(s)):
			if isPal(s[first:i+1]):
				backtrack(i + 1, tmpResult + [s[first:i+1]])

	backtrack(0, [])
	return result 

print(partition('aab'))


'''
37. Sudoku Solver
Hard

Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
Empty cells are indicated by the character '.'.
(Check the original problem for the example sudoku puzzle and solution
	at https://leetcode.com/problems/sudoku-solver/)

A sudoku puzzle...


...and its solution numbers marked in red.

Note:

The given board contain only digits 1-9 and the character '.'.
You may assume that the given Sudoku puzzle will have a single unique solution.
The given board size is always 9x9.
'''
def solveSudoku(board: [[str]]):
	# Do not return anything, modify board in-place instead.

	# Save this question for later. 
	return


'''
51. N-Queens
Hard

1574

65

Add to List

Share
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.



Given an integer n, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.

Example:

Input: 4
Output: [
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above.
'''

def solveNQueens(n: int):
	# Save this one for later. 
	return 


