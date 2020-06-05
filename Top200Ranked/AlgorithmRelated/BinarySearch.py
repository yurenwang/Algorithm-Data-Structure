'''
######################################## Binary Search ##################
69. Sqrt(x)
744. Find Smallest Letter Greater Than Target
540. Single Element in a Sorted Array
278. First Bad Version
153. Find Minimum in Rotated Sorted Array
34. Find First and Last Position of Element in Sorted Array

'''


'''
69. Sqrt(x)
Easy

1126

1749

Add to List

Share
Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:

Input: 4
Output: 2
Example 2:

Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
'''
# Time: O(log(n))  Space: O(1)
def mySqrt(x: int):
	l = 0
	h = x
	while l <= h:
		m = l + (h - l) // 2
		if m * m == x:
			return m
		elif m * m < x:
			if (m + 1) * (m + 1) > x:
				return m
			l = m + 1
		else:
			h = m

print(mySqrt(8))
print(mySqrt(16))
print(mySqrt(1))
print(mySqrt(4))


'''
744. Find Smallest Letter Greater Than Target
Easy

Given a list of sorted characters letters containing only lowercase letters, and given a target letter target, find the smallest element in the list that is larger than the given target.

Letters also wrap around. For example, if the target is target = 'z' and letters = ['a', 'b'], the answer is 'a'.

Examples:
Input:
letters = ["c", "f", "j"]
target = "a"
Output: "c"

Input:
letters = ["c", "f", "j"]
target = "c"
Output: "f"

Input:
letters = ["c", "f", "j"]
target = "d"
Output: "f"

Input:
letters = ["c", "f", "j"]
target = "g"
Output: "j"

Input:
letters = ["c", "f", "j"]
target = "j"
Output: "c"

Input:
letters = ["c", "f", "j"]
target = "k"
Output: "c"
Note:
letters has a length in range [2, 10000].
letters consists of lowercase letters, and contains at least 2 unique letters.
target is a lowercase letter.
'''

# We can use Linear Scan, which takes O(n) time. Just scan through the input list and return when
# the current char is greater than the target

# However, we can use binary search to improve the time to O(log(n)) because the input list is sorted

# Time: O(log(n))  Space: O(1)
def nextGreatestLetter(letters: [str], target: str):
	l = 0
	r = len(letters) - 1
	while l < r:
		m = l + (r - l) // 2
		if letters[m] > target:
			r = m
		else:
			l = m + 1
	if letters[l] > target:
		return letters[l]
	else:
		return letters[0]

print(nextGreatestLetter(["c", "f", "j"], "c"))
print(nextGreatestLetter(["c", "f", "j"], "k"))


'''
540. Single Element in a Sorted Array
Medium

You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.

Example 1:

Input: [1,1,2,3,3,4,4,8,8]
Output: 2
Example 2:

Input: [3,3,7,7,10,11,11]
Output: 10
 

Note: Your solution should run in O(log n) time and O(1) space.
'''

# We can use linary scan to solve this with O(n) time 
# However, we can use binary search as below, to improve it with O(logn) time 

# You need to draw out all the scenarios to analyze how the while loop should go 

# Time: O(log(n))  Space: O(1)
def singleNonDuplicate(nums: [int]):
	if len(nums) == 1:
		return nums[0]
	l = 0
	r = len(nums) - 1
	while l < r:
		m = l + (r - l) // 2
		remainder = (r - l + 1) % 4
		if nums[m] == nums[m+1]:
			if remainder == 1:
				l = m + 2
			else:
				r = m - 1
		elif nums[m] == nums[m-1]:
			if remainder == 1:
				r = m - 2
			else:
				l = m + 1
		else:
			return nums[m]
	return nums[l]

print('540. Single Element in a Sorted Array')
print(singleNonDuplicate([1,1,2,2,3]))
print(singleNonDuplicate([3,3,7,7,10,11,11]))
print(singleNonDuplicate([1,1,2,3,3,4,4,8,8]))


'''
278. First Bad Version
Easy

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example:

Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version. 
'''

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

# Time: O(log(n))  Space: O(1)
def isBadVersion(m):
	# I'm just faking this so we don't have error
	return True

def firstBadVersion(self, n):
	"""
	:type n: int
	:rtype: int
	"""
	lo = 1
	hi = n
	while lo < hi:
		m = lo + (hi - lo) // 2
		if isBadVersion(m):
			hi = m
		else:
			lo = m + 1
	return lo


'''
153. Find Minimum in Rotated Sorted Array
Medium

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:

Input: [3,4,5,1,2] 
Output: 1
Example 2:

Input: [4,5,6,7,0,1,2]
Output: 0
'''

# Binary search
# Time: O(log(n))  Space: O(1)
def findMin(nums: [int]):
	lo = 0
	hi = len(nums) - 1
	while lo < hi:
		m = lo + (hi - lo) // 2
		if nums[m] > nums[hi]:
			lo = m + 1
		else:
			hi = m
	return nums[lo]

print(findMin([4,5,6,7,0,1,2]))


'''
34. Find First and Last Position of Element in Sorted Array
Medium

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
'''

# For this question, we need to find the first and last target item separately

# Time: O(log(n))  Space: O(1)
def searchRange(nums: [int], target: int) -> [int]:
	return [searchRangeLeft(nums, target, True), searchRangeLeft(nums, target, False)]

# helper function to return the index of either left or right most target
def searchRangeLeft(nums: [int], target: int, left: bool) -> int:
	if len(nums) == 0:
		return -1
	lo = 0
	hi = len(nums) - 1
	while lo < hi:
		m = lo + (hi - lo) // 2
		if left:
			if nums[m] >= target:
				hi = m
			else:
				lo = m + 1
		else:
			if nums[m] <= target:
				lo = m + 1 
			else:
				hi = m
	if left and nums[lo] == target or not left and nums[lo] == target:
		return lo
	elif not left and nums[lo-1] == target:
		return lo - 1
	else:
		return -1

print(searchRange([5,7,7,8,8,10], 8))
print(searchRange([5,7,7,8,8,10], 6))

