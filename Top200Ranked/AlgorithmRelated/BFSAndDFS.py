'''
########################################## BFS and DFS ###############
###### BFS:
1091. Shortest Path in Binary Matrix
279. Perfect Squares
127. Word Ladder

###### DFS:
695. Max Area of Island
200. Number of Islands
547. Friend Circles
130. Surrounded Regions
417. Pacific Atlantic Water Flow

'''

########################################## Breadth First Search ########################

'''
1091. Shortest Path in Binary Matrix
Medium

In an N by N square grid, each cell is either empty (0) or blocked (1).

A clear path from top-left to bottom-right has length k if and only if it is composed of cells C_1, C_2, ..., C_k such that:

Adjacent cells C_i and C_{i+1} are connected 8-directionally (ie., they are different and share an edge or corner)
C_1 is at location (0, 0) (ie. has value grid[0][0])
C_k is at location (N-1, N-1) (ie. has value grid[N-1][N-1])
If C_i is located at (r, c), then grid[r][c] is empty (ie. grid[r][c] == 0).
Return the length of the shortest such clear path from top-left to bottom-right.  If such a path does not exist, return -1.

Example 1:

Input: [[0,1],[1,0]]
Output: 2

Example 2:

Input: [[0,0,0],[1,1,0],[1,1,0]]
Output: 4

Note:
1 <= grid.length == grid[0].length <= 100
grid[r][c] is 0 or 1
'''

# BFS

# Time: O(n^2)  Space: O(n^2)??
def shortestPathBinaryMatrix(grid: [[int]]):
	l = len(grid) - 1
	if grid[0][0] or grid[l][l]:
		return -1
	queue = [(0, 0, 1)]
	grid[0][0] = 1

	for i, j, step in queue:
		if i == l and j == l:
			return step
		for x, y in [(i, j+1), (i, j-1), (i-1, j), (i+1, j), (i-1, j+1), (i+1, j+1), (i-1, j-1), (i+1, j-1)]:
			if 0 <= x <= l and 0 <= y <= l and not grid[x][y]:
				queue.append((x, y, step + 1))
				grid[x][y] = 1
	return -1

print(shortestPathBinaryMatrix([[0,0,0],[1,1,0],[1,1,0]]))
print(shortestPathBinaryMatrix([[0,1],[1,0]]))


'''
279. Perfect Squares
Medium

Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

Example 1:

Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
Example 2:

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
Accepted
256,623
Submissions
570,335

'''
import math

# BFS

# Used BFS with Greedy to minimize the Time
# Store the remainder after subtracting a square in the tree, then use BFS to traverse it  
def numSquares(n: int):
	if n == 1:
		return 1
	squares = [x * x for x in range(1, int(math.sqrt(n)) + 1)]
	# Store the count in the queue
	queue = [(n, 1)]

	for item, count in queue:
		for s in squares:
			if item == s:
				return count
			elif item < s:
				break
			else:
				queue.append((item - s, count + 1))

print(numSquares(12))
print(numSquares(13))


'''
127. Word Ladder
Medium

Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.

'''
from collections import deque
# Time: O(m^2 n) where m is number of strings in the wordlist, and n is the length of each string 
# Space: O(mn)

# This version can pass all the tests, but will run for too long time because of using the helper function 'connectedNodes()' for some reason
def ladderLength(beginWord: str, endWord: str, wordList: [str]):
	if endWord not in wordList:
		return 0
	queue = deque([(beginWord, 1)])
	# mark as visited to avoid visiting repeatly
	visited = {beginWord: True}
	while queue:
		word, count = queue.popleft()
		if word == endWord:
			return count
		# checking the next word takes O(mn), which is slow
		for candidate in wordList:
			if candidate not in visited:
				if connectedNodes(word, candidate):
					queue.append((candidate, count + 1))
					visited[candidate] = True
	return 0
		
# this helper method will take time O(n) to run, n is length of a word
def connectedNodes(word1: str, word2: str):
	diffCount = 0
	if word1 == word2:
		return False
	for i in range(0, len(word1)):
		if word1[i] != word2[i]:
			if diffCount == 1:
				return False
			else:
				diffCount = 1
	return True

print(ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]))
print(ladderLength("hit", "cog", ["hot","dot","dog","lot","log"]))
print(ladderLength("hot", "dog", ["hot","dog"]))


# this version creates the next word based on the 26 letters
# Time: O(mn) Space: O(mn)

# this version will be faster
def ladderLengthModified(beginWord: str, endWord: str, wordList: [str]):
	# get a set of all potential chars for swapping later
	potentialSwapChar = {w for word in wordList for w in word}

	if endWord not in wordList:
		return 0
	queue = deque([(beginWord, 1)])
	ogList = set(wordList)
	# mark as visited to avoid visiting repeatly
	visited = set(beginWord)

	while queue:
		word, count = queue.popleft()
		if word == endWord:
			return count
		# checking the next word takes O(26n) = O(n) time, which is faster than O(mn)
		for i in range(len(word)):
			for c in potentialSwapChar:
				nextWord = word[:i] + c + word[i+1:]
				if nextWord in ogList and nextWord not in visited:
					queue.append((nextWord, count + 1))
					visited.add(nextWord)
	return 0

print(ladderLengthModified("hit", "cog", ["hot","dot","dog","lot","log","cog"]))
print(ladderLengthModified("hit", "cog", ["hot","dot","dog","lot","log"]))
print(ladderLengthModified("hot", "dog", ["hot","dog"]))


########################################## Depth First Search #########################

'''
695. Max Area of Island
Medium

Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

Example 1:

[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
Given the above grid, return 6. Note the answer is not 11, because the island must be connected 4-directionally.
Example 2:

[[0,0,0,0,0,0,0,0]]
Given the above grid, return 0.
Note: The length of each dimension in the given grid does not exceed 50.

'''
# DFS
# Time: O(mn)  m is height, n is width  
# Space: O(mn) since we need to remember visited cells
def maxAreaOfIsland(grid: [[int]]):
	visited = set()

	def area(i, j): # calculate the max area at a specific cell
		if not (0 <= i < len(grid) and 0 <= j < len(grid[0]) and (i, j) not in visited and grid[i][j]):
			return 0 # return 0 if cell is not valid or visited
		visited.add((i, j))
		return 1 + area(i-1, j) + area(i+1, j) + area(i, j-1) + area(i, j+1)

	return max(area(r, c) for r in range(len(grid)) for c in range(len(grid[0])))

print(maxAreaOfIsland([[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]))
print(maxAreaOfIsland([[0,0,0,0,0,0,0,0]]))


'''
200. Number of Islands
Medium

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1

Example 2:

Input:
11000
11000
00100
00011

Output: 3

'''

# DFS with Recursion

# Time: O(mn)  Space: worst case: O(mn) in case when map is filled with island, and 
# 	dfs goes by m*n deep
def numIslands(grid: [[str]]):
	def traverseIsland(i, j):
		# if this is still a valid part of island 
		if not (0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1'):
			return
		grid[i][j] = '0'
		traverseIsland(i+1, j)
		traverseIsland(i-1, j)
		traverseIsland(i, j+1)
		traverseIsland(i, j-1)

	count = 0
	for x in range(len(grid)):
		for y in range(len(grid[0])):
			if grid[x][y] == '1':
				count += 1
				traverseIsland(x, y)
	return count

print("200. Number of Islands")
print(numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]))
print(numIslands([["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]))


'''
547. Friend Circles
Medium

There are N students in a class. Some of them are friends, while some are not. Their friendship is transitive in nature. For example, if A is a direct friend of B, and B is a direct friend of C, then A is an indirect friend of C. And we defined a friend circle is a group of students who are direct or indirect friends.

Given a N*N matrix M representing the friend relationship between students in the class. If M[i][j] = 1, then the ith and jth students are direct friends with each other, otherwise not. And you have to output the total number of friend circles among all the students.

Example 1:
Input: 
[[1,1,0],
 [1,1,0],
 [0,0,1]]
Output: 2
Explanation:The 0th and 1st students are direct friends, so they are in a friend circle. 
The 2nd student himself is in a friend circle. So return 2.
Example 2:
Input: 
[[1,1,0],
 [1,1,1],
 [0,1,1]]
Output: 1
Explanation:The 0th and 1st students are direct friends, the 1st and 2nd students are direct friends, 
so the 0th and 2nd students are indirect friends. All of them are in the same friend circle, so return 1.
Note:
N is in range [1,200].
M[i][i] = 1 for all students.
If M[i][j] = 1, then M[j][i] = 1.

'''
# DFS. Set cell to be 0 when traversing
# Time: O(mn)  Space: O(m+n)?
def findCircleNum(M: [[int]]):
	def traverseFriend(i, j):
		if not (0 <= i < len(M) and 0 <= j < len(M[0]) and M[i][j]):
			return
		M[i][j] = 0
		for r in range(len(M)):
			traverseFriend(r, j)
		for c in range(len(M[0])):
			traverseFriend(i, c)

	count = 0
	for x in range(len(M)):
		for y in range(len(M[0])):
			if M[x][y]:
				count += 1
				traverseFriend(x, y)
	return count

print('547. Friend Circles')
print(findCircleNum([[1,1,0],[1,1,0],[0,0,1]]))
print(findCircleNum([[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]))


'''
130. Surrounded Regions
Medium

Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:

X X X X
X O O X
X X O X
X O X X
After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X
Explanation:

Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.

'''

# We go through the whole most outside layer of the matrix, and for each 'O' and its connected 'O's we met, 
# 	we mark it as something else, like 'M'. Then we change all remaining 'O's to 'X' and change 'M's back to 'O'
# DFS
# Time: O(m*n)  Space: O(m*n)
def solve(board: [[str]]):
	"""
	Do not return anything, modify board in-place instead.
	"""
	if len(board) <= 2 or len(board[0]) <= 2:
		return

	row = len(board)
	col = len(board[0])
	
	# Helper function to traverse all connected cells starting [r][c]
	def traverseFrom(r, c):
		if not (0 <= r < row and 0 <= c < col and board[r][c] == 'O'):
			return
		board[r][c] = 'M'
		traverseFrom(r+1, c)
		traverseFrom(r, c+1) 
		traverseFrom(r-1, c) 
		traverseFrom(r, c-1)

	# get the border
	border = [(i, j) for i in [0, row-1] for j in range(col)]
	border.extend([(i, j) for i in range(row) for j in [0, col-1]])

	# Traverse all boarder 'O's and their connecting 'O's
	for (i, j) in border:
		if board[i][j] == 'O':
			traverseFrom(i, j)

	# intertools.product(l1, l2) returns all combinations of the two lists
	from itertools import product
	for i, j in product(range(row), range(col)): 
		# rewrite everything to what we desire
		if board[i][j] == 'O':
			board[i][j] = 'X'
		elif board[i][j] == 'M':
			board[i][j] = 'O'

matrix = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
solve(matrix)
print(matrix)


'''
417. Pacific Atlantic Water Flow
Medium

Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

Note:

The order of returned grid coordinates does not matter.
Both m and n are less than 150.
 

Example:

Given the following 5x5 matrix:

  Pacific ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic

Return:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).

'''

# DFS
# Time: O(m*n)  Space: O(m*n)
def pacificAtlantic(matrix: [[int]]):
	if len(matrix) == 0 or len(matrix[0]) == 0:
		return []
	row = len(matrix)
	col = len(matrix[0])
	flowPacific = set() # set of cells where can flow to Pacific ocean
	flowAtlantic = set() # set of cells where can flow to Atlantic ocean

	# traverse from [i][j]
	def dfs(i, j, prev, visited):
		if not (0 <= i < row and 0 <= j < col and (i, j) not in visited and matrix[i][j] >= prev):
			return
		visited.add((i, j))
		curr = matrix[i][j]
		dfs(i+1, j, curr, visited)
		dfs(i-1, j, curr, visited)
		dfs(i, j+1, curr, visited)
		dfs(i, j-1, curr, visited)

	# create list for both ocean border
	pacific = [(0, i) for i in range(col)]
	pacific.extend([(j, 0) for j in range(1, row)])
	atlantic = [(row-1, i) for i in range(col)]
	atlantic.extend([(j, col-1) for j in range(row-1)])

	# put all flow-able area into both sets, and get the intersection of the two sets
	for (r, c) in pacific: dfs(r, c, 0, flowPacific)
	for (r, c) in atlantic: dfs(r, c, 0, flowAtlantic)
	return [[i, j] for i, j in flowPacific.intersection(flowAtlantic)]

print(pacificAtlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]))

