'''
############################################### Divide and Conquer ######################
241. Different Ways to Add Parentheses
95. Unique Binary Search Trees II

'''

'''
241. Different Ways to Add Parentheses
Medium

Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and *.

Example 1:

Input: "2-1-1"
Output: [0, 2]
Explanation: 
((2-1)-1) = 0 
(2-(1-1)) = 2
Example 2:

Input: "2*3-4*5"
Output: [-34, -14, -10, -10, 10]
Explanation: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
'''

# Divide and Conquer
# put each possible solutions of sub-questions into arrays, then use nested for loop to find all
#   possible solutions
# Time: O(2^n)??
calculated = {}
def diffWaysToCompute(input: str):
	if input.isdigit():
		return [int(input)]
	if input in calculated:
		return calculated[input]
	result = []
	for i in range(len(input)):
		if input[i] in '+-*':
			res1 = diffWaysToCompute(input[:i])
			res2 = diffWaysToCompute(input[i+1:])
			for r1 in res1:
				for r2 in res2:
					result.append(compute(r1, r2, input[i]))
	calculated[input] = result
	return result

# Helper function to do the math about first and second number, based on the operation
def compute(first, second, operation):
	if operation == '+':
		return first + second
	elif operation == '-':
		return first - second
	else:
		return first * second

print(diffWaysToCompute("2*3-4*5"))
print(diffWaysToCompute("2-1-1"))


'''
95. Unique Binary Search Trees II
Medium

Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1 ... n.

Example:

Input: 3
Output:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
Explanation:
The above output corresponds to the 5 unique BST's shown below:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# Divide and Conquer

def generateTrees(n: int):
	return generateTreesInRange(1, n) if n else []

# helper function to generate a tree in a range recursively
def generateTreesInRange(lo, hi, calculated = {}):
	if makeKey(lo, hi) in calculated:
		return calculated[makeKey(lo, hi)]
	if lo > hi:
		return [None]
	if lo == hi:
		return [TreeNode(lo)]
	result = []
	for i in range(lo, hi + 1):
		# all possibilities for left of the tree, and for right of the tree
		left = generateTreesInRange(lo, i - 1)
		right = generateTreesInRange(i + 1, hi)
		# loop through all options to build and append the tree
		for l in left:
			for r in right:
				currNode = TreeNode(i)
				currNode.left = l
				currNode.right = r
				result.append(currNode)
	calculated[makeKey(lo, hi)] = result
	return result

# helper function to calculate the key of a TreeNode to check if it is calculated
def makeKey(lo, hi):
	return str(lo) + '-' + str(hi)

print(generateTrees(3))
