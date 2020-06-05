'''
################################################ Array and Matrix ##################

283. Move Zeroes
566. Reshape the Matrix
485. Max Consecutive Ones
240. Search a 2D Matrix II
378. Kth Smallest Element in a Sorted Matrix
645. Set Mismatch
287. Find the Duplicate Number
667. Beautiful Arrangement II
697. Degree of an Array
766. Toeplitz Matrix
565. Array Nesting
769. Max Chunks To Make Sorted

'''

'''
283. Move Zeroes
Easy

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Example:

Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
Note:

You must do this in-place without making a copy of the array.
Minimize the total number of operations.
'''

class Solution:
    # Time: O(N)  Space: O(1)
    def moveZeroes(self, nums: [int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # Two pointers
        # p2 is the fast pointer finding all non-0's. Swap p2 and p1 value if non-0 is found
        # Then move p1 forward
        p1 = p2 = 0
        while p2 < len(nums):
            if nums[p2] != 0:
                nums[p1], nums[p2] = nums[p2], nums[p1]
                p1 += 1
            p2 += 1

	
'''
566. Reshape the Matrix
Easy

In MATLAB, there is a very useful function called 'reshape', which can reshape a matrix into a new one with different size but keep its original data.

You're given a matrix represented by a two-dimensional array, and two positive integers r and c representing the row number and column number of the wanted reshaped matrix, respectively.

The reshaped matrix need to be filled with all the elements of the original matrix in the same row-traversing order as they were.

If the 'reshape' operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.

Example 1:
Input: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
Output: 
[[1,2,3,4]]
Explanation:
The row-traversing of nums is [1,2,3,4]. The new reshaped matrix is a 1 * 4 matrix, fill it row by row by using the previous list.
Example 2:
Input: 
nums = 
[[1,2],
 [3,4]]
r = 2, c = 4
Output: 
[[1,2],
 [3,4]]
Explanation:
There is no way to reshape a 2 * 2 matrix to a 2 * 4 matrix. So output the original matrix.
Note:
The height and width of the given matrix is in range [1, 100].
The given r and c are all positive.
'''

class Solution2:
    # Time: O(h * w)  Space: No extra space is used. Used O(r * c) space for the result
    def matrixReshape(self, nums: [[int]], r: int, c: int) -> [[int]]:
        h = len(nums)
        w = len(nums[0])
        if r * c != h * w:
            return nums
        # First build a result matrix of c * r
        result = [[None] * c for _ in range(r)]
        
        # x, and y to represent the current x and y index of the result list
        x = y = 0
        
        # loop through nums row by row, and fill in the result. Move x, y to next line when reach to end
        #   of a row
        for i in range(h):
            for j in range(w):
                result[x][y] = nums[i][j]
                if y == c - 1:
                    x, y = x + 1, 0
                else:
                    y += 1
                    
        return result


'''
485. Max Consecutive Ones
Easy

Given a binary array, find the maximum number of consecutive 1s in this array.

Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.
Note:

The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000
'''

class Solution3:
    # Time: O(N)  Space: O(1)
    def findMaxConsecutiveOnes(self, nums: [int]) -> int:
        # One scan:
        p = curr = result = 0
        while p < len(nums):
            if nums[p]:
                curr += 1
            else:
                result = max(result, curr)
                curr = 0
            p += 1
        result = max(result, curr) # Deal with the last group of 1's
        return result


'''
240. Search a 2D Matrix II
Medium

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.
'''

class Solution4:
    # Time: O(h + w)  Space: O(1)
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # We can start our pointer from bottom left, or from top right, as starting at those places, we
        #   know which direction to go if the curr value is larger than or smaller than the target
        if not matrix or not matrix[0]:
            return False
        h = len(matrix)
        w = len(matrix[0])
        # initial search pointer, starting from bottom left
        i = h - 1
        j = 0
        while 0 <= i < h and 0 <= j < w:
            curr = matrix[i][j]
            if curr < target:
                j += 1
            elif curr > target:
                i -= 1
            else:
                return True
        return False


'''
378. Kth Smallest Element in a Sorted Matrix
Medium

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note:
You may assume k is always valid, 1 ≤ k ≤ n2.
'''

# I used heap to solve this problem. 
# It can also be solved using binary search. However, I think using heap makes more sense
class Solution5:
    # Time: O(min(h, k) + k * log(min(h, k)))   heap construction takes O(min(h, k)) time
    # Space: O(min(h, k))
    def kthSmallest(self, matrix: [[int]], k: int) -> int:
        # If we only have a 2xN matrix, which is 2 sorted arrays, it would be easier
        # We can just use two pointers starting at the beginning of the two arrays, and go right until 
        # find the kth smallest. 
        # However, since we are having M arrays to compare, we have to use a heap to store the pointers.
        # Using heap is ideal as for each time, we want to get the smallest value amoung the pointers, 
        #   and a min-heap is perfect for it with O(1) time of retrieval and O(logN) time of adding new
        #   elements.
        
        # importing "heapq" to implement heap queue 
        import heapq
        
        h = len(matrix)
        w = len(matrix[0])
        # Initialize the heap with the first values in each array
        # We need to keep track of the value, and the row and column number
        # Put the value at the first place in the tuple to let the heap compare based on the value
        # Set range to min(h, k) to eliminate unnecessary heap size if k < h
        ptrHeap = [(matrix[i][0], i, 0) for i in range(min(h, k))]
        heapq.heapify(ptrHeap)
        
        # Keep poping and pushing pointers accordingly to get the kth smallest
        while k:
            smallest, row, col = heapq.heappop(ptrHeap)
            if col < w - 1:
                heapq.heappush(ptrHeap, (matrix[row][col + 1], row, col + 1))
            k -= 1
        
        return smallest


'''
645. Set Mismatch
Easy

The set S originally contains numbers from 1 to n. But unfortunately, due to the data error, one of the numbers in the set got duplicated to another number in the set, which results in repetition of one number and loss of another number.

Given an array nums representing the data status of this set after the error. Your task is to firstly find the number occurs twice and then find the number that is missing. Return them in the form of an array.

Example 1:
Input: nums = [1,2,2,4]
Output: [2,3]
Note:
The given array size will in the range [2, 10000].
The given array's numbers won't have any order.
'''

class Solution6:
    # Time: O(N)  Space: O(1)
    def findErrorNums(self, nums: [int]) -> [int]:
        # Easy way to solve is to use an extra set, or list, and put numbers in it. It would take
        #   O(N) of time and O(N) of space
        
        # However, a better solution is to solve it without extra space, modifying the original list
        #   and invert it in the iteration
        dup = -1
        mis = -1
        for n in nums:
            if nums[abs(n) - 1] < 0:
                # being negative means the current nums[n-1] has been flipped, means n has been visited 
                # Therefore, duplication is n
                dup = abs(n)
            else:
                nums[abs(n) - 1] *= -1
        
        for i in range(0, len(nums)):
            # num i won't be flipped as it is not in nums, therefore it will still be possible
            if nums[i] > 0:
                mis = i + 1
        
        return (dup, mis)
        

'''
287. Find the Duplicate Number
Medium

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

Example 1:

Input: [1,3,4,2,2]
Output: 2
Example 2:

Input: [3,1,3,4,2]
Output: 3
Note:

You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.
'''

class Solution7:
    # Time: O(N)  Space: O(1)
    def findDuplicate(self, nums: [int]) -> int:
        # Three Approaches:
        #   1. Sort the list then find consecutive same value items. This will change the original list,
        #       or use N extra space
        #   2. Use a set, this will use N extra spaces
        #   3. Floyd's Tortoise and Hare (Cycle Detection). Will use this one as it takes O(N) time and
        #       O(1) space
        
        # Check the detailed explanation in the solution section. The idea is, for each number n, point
        #   the next node to be the number at nums[n]. Therefore we can know that a cycle occurs when 
        #   two numbers pointing to the same next at nums[n]
        
        # Set original value for slow and fast
        pSlow = pFast = nums[0]
        
        # First make fast 2x speed of the slow. They will encounter each other but that isn't necessarily
        #   the meet point
        while True:   
            pSlow = nums[pSlow]
            pFast = nums[nums[pFast]]
            if pSlow == pFast:
                break
        pFast = nums[0]
        
        # Then move fast to original point and set them to same speed. They will then meet at the meet
        #   point where duplicate exists. Proof of it is throughly explained in the solution section 3
        while pSlow != pFast:   
            pSlow = nums[pSlow]
            pFast = nums[pFast]
            
        return pSlow


'''
667. Beautiful Arrangement II
Medium

Given two integers n and k, you need to construct a list which contains n different positive integers ranging from 1 to n and obeys the following requirement:
Suppose this list is [a1, a2, a3, ... , an], then the list [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] has exactly k distinct integers.

If there are multiple answers, print any of them.

Example 1:
Input: n = 3, k = 1
Output: [1, 2, 3]
Explanation: The [1, 2, 3] has three different positive integers ranging from 1 to 3, and the [1, 1] has exactly 1 distinct integer: 1.
Example 2:
Input: n = 3, k = 2
Output: [1, 3, 2]
Explanation: The [1, 3, 2] has three different positive integers ranging from 1 to 3, and the [2, 1] has exactly 2 distinct integers: 1 and 2.
Note:
The n and k are in the range 1 <= k < n <= 104.
'''

class Solution8:
    # Time: O(N)  Space: O(1) no extra space is used except for the output list
    def constructArray(self, n: int, k: int) -> [int]:
        # When k = n - 1, result has to be [1, n, 2, n-1, 3, n-2, ...]
        # When k = 1, result has to be [1, 2, 3, 4, ...]
        # Therefore, we can patch them together
        
        # For example, when n = 6, k = 5, we need [1,  6, 2, 5, 3, 4]
        #              when n = 6, k = 4, we need [1, 2,  6, 3, 5, 4]
        #              when n = 6, k = 3, we need [1, 2, 3,  6, 4, 5]
        #              when n = 6, k = 2, we need [1, 2, 3, 4,  6, 5]
        #              when n = 6, k = 1, we need [1, 2, 3, 4, 5,  6]
        
        # With this in mind, we can construct our result:
        # First part:
        result = [i for i in range(1, n - k + 1)]
        
        # Second part:
        result.append(n)
        k -= 1
        prev = n
        while k != 0:
            # Modify the value appended on the result list in the loop
            result.append(prev - k)
            prev = prev - k
            k = -(k - 1) if k > 0 else -(k + 1)
        
        return result


'''
697. Degree of an Array
Easy

Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency of any one of its elements.

Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.

Example 1:
Input: [1, 2, 2, 3, 1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
Example 2:
Input: [1,2,2,3,1,4,2]
Output: 6
Note:

nums.length will be between 1 and 50,000.
nums[i] will be an integer between 0 and 49,999.
'''

class Solution9:
    # Time: O(N)  Space: O(N)
    def findShortestSubArray(self, nums: [int]) -> int:
        # The smallest subarry would be begin with the first index of the most recurring number, end with
        #   the last index of the most recurring number
        # Count the occurance of each n in nums, as well as their left most occurance index and right
        #   most occurance index. Then get the max of counts, and calculate result based on that.
        left, right, count = {}, {}, {}
        for i in range(len(nums)):
            n = nums[i]
            if n not in left:
                left[n] = i
            right[n] = i
            count[n] = count.get(n, 0) + 1
            
        # Find the most occuring one
        degree = max(count.values())
        
        # Calculate the result
        result = len(nums)
        for c in count.keys():
            if count[c] == degree:
                result = min(result, right[c] - left[c] + 1)
                
        return result


'''
766. Toeplitz Matrix
Easy

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.

Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
 

Example 1:

Input:
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
Output: True
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.
Example 2:

Input:
matrix = [
  [1,2],
  [2,2]
]
Output: False
Explanation:
The diagonal "[1, 2]" has different elements.

Note:

matrix will be a 2D array of integers.
matrix will have a number of rows and columns in range [1, 20].
matrix[i][j] will be integers in range [0, 99].

Follow up:

What if the matrix is stored on disk, and the memory is limited such that you can only load at most one row of the matrix into the memory at once?
What if the matrix is so large that you can only load up a partial row into the memory at once?
'''

class Solution10:
    # Time: O(W * H)  Space: O(1)
    
    # Follow up is not considered
    def isToeplitzMatrix(self, matrix: [[int]]) -> bool:
        # We can:
        #   1. Group the matrix into diagonals, and all nums on one diagonal should have same value.
        #       Use a hashMap to check if one item exists, if it is equal to the existing ones.
        #       Space: O(W + H)
        #   2. Compare every item to its upper left neighbor, and return False if they are not the same.
        #       Space: O(1)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i > 0 and j > 0 and matrix[i - 1][j - 1] != matrix[i][j]:
                    return False
        return True


'''
565. Array Nesting
Medium

A zero-indexed array A of length N contains all integers from 0 to N-1. Find and return the longest length of set S, where S[i] = {A[i], A[A[i]], A[A[A[i]]], ... } subjected to the rule below.

Suppose the first element in S starts with the selection of element A[i] of index = i, the next element in S should be A[A[i]], and then A[A[A[i]]]… By that analogy, we stop adding right before a duplicate element occurs in S.
 

Example 1:

Input: A = [5,4,0,3,1,6,2]
Output: 4
Explanation: 
A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.

One of the longest S[K]:
S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
 

Note:

N is an integer within the range [1, 20,000].
The elements of A are all distinct.
Each element of A is an integer within the range [0, N-1].
'''

class Solution11:
    # Time: O(N)  Space: O(1)
    # Time is O(N) because although we have nested loops, we only visit each number once.
    def arrayNesting(self, nums: [int]) -> int:
        # as 1 <= N <= 20000, we can add 20000 to each visited number to mark it as visited so that
        #   we only use O(1) space
        i = 0
        result = 0
        # We have to consider all indexes as the starting point
        for i in range(len(nums)):
            count = 0
            # Count until we reach value >= 20000, which means visited
            while nums[i] < 20000:
                count += 1
                tmp = i
                i = nums[i]
                nums[tmp] += 20000
            result = max(result, count)
        return result


'''
769. Max Chunks To Make Sorted
Medium

Given an array arr that is a permutation of [0, 1, ..., arr.length - 1], we split the array into some number of "chunks" (partitions), and individually sort each chunk.  After concatenating them, the result equals the sorted array.

What is the most number of chunks we could have made?

Example 1:

Input: arr = [4,3,2,1,0]
Output: 1
Explanation:
Splitting into two or more chunks will not return the required result.
For example, splitting into [4, 3], [2, 1, 0] will result in [3, 4, 0, 1, 2], which isn't sorted.
Example 2:

Input: arr = [1,0,2,3,4]
Output: 4
Explanation:
We can split into two chunks, such as [1, 0], [2, 3, 4].
However, splitting into [1, 0], [2], [3], [4] is the highest number of chunks possible.
Note:

arr will have length in range [1, 10].
arr[i] will be a permutation of [0, 1, ..., arr.length - 1].
'''

class Solution12:
    # Time: O(N)  Space: O(1)
    def maxChunksToSorted(self, arr: [int]) -> int:
        # The 'brute force' solution in the solution section is actually really smart
        # You have to draw out some examples to fully understand the solution
        result = curMax = 0
        # This is the same as using 'for i in range(len(arr)):' and use 'arr[i]'
        for i, val in enumerate(arr):
            curMax = max(curMax, val)
            if curMax == i: 
                result += 1
        return result
