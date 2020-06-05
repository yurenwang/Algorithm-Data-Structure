'''
######################################################### Ordering ###################
215. Kth Largest Element in an Array
347. Top K Frequent Elements
451. Sort Characters By Frequency
75. Sort Colors

'''


'''
215. Kth Largest Element in an Array
Medium

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5
Example 2:

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.

'''

'''
1. We can sort the list first using Merge sort or quick sort, then get the kth item, this will
take O(nlog(n)) time  

2. A better solution is to use a MinHeap with size k, then in the end we pop the top item,
which will be kth largest. In python, we have the heapq.nlargest() method which does the
same thing. This will take O(nlog(k)) time, which is improved

3 An even better solution is to use QuickSelect, which takes O(n)
	- Choose a random pivot.

	- Use a partition algorithm to place the pivot into its perfect position pos in the sorted array, move smaller elements to the left of pivot, and larger or equal ones - to the right.

	- Compare pos and N - k to choose the side of array to proceed recursively.
'''

# My solution uses heapq.nlargest, which is the same as using minHeap
# Time: O(nlog(k))  Space: O(k)
import heapq 

def findKthLargest(nums: [int], k: int):
	return heapq.nlargest(k, nums)[-1]

print(findKthLargest([3,2,1,5,6,4],2))


'''
347. Top K Frequent Elements
Medium

Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

'''

# I used dictionary to store number of existance for each item, 
#	then use minHeap to get the top k frequent ones

# Time: O(nlog(k))  Space: O(n)

# import heapq
def topKFrequent(nums: [int], k: int):
	dict = {}
	for num in nums:
		if num not in dict:
			dict[num] = 1
		else:
			dict[num] += 1

	# using heapq to create a minHeap
	return heapq.nlargest(k, dict.keys(), lambda x: dict[x])

print(topKFrequent([1,1,1,2,2,3], 3))
print(topKFrequent([1,2,2,3,5,5,5,60,80,100], 3))



'''
451. Sort Characters By Frequency
Medium

Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:
Input: "tree"		Output: "eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:
Input: "cccaaa"		Output: "cccaaa"

Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.

Example 3:
Input: "Aabb"		Output: "bbAa"

Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
'''

# import collections to use collections.Counter()
import collections

# I used Bucket Sort to reduce the time complexity of sorting the dictionary to O(n)
# Time: O(n)  Space: O(n)
def frequencySort(s: str):
	if len(s) == 0:
		return s

	# use Counter to get a dictionary of all item with their counts
	counter = collections.Counter(list(s))
	maxCount = max(counter.values())

	# use bucket sort to sort all items based on their count in O(n) time
	bucket = [[] for _ in range(maxCount + 1)]
	for item, count in counter.items():
		bucket[count].append(item)
	
	# build the result list and join to get the result
	resultList = []
	for count in range(len(bucket) - 1, 0, -1):
		for item in bucket[count]:
			resultList.append(item * count)

	return ''.join(resultList)

print(frequencySort("tree"))
	

'''
75. Sort Colors
Medium

Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Follow up:

A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
Could you come up with a one-pass algorithm using only constant space?

'''

# Time: O(n)  Space: O(1)
def sortColors(nums: [int]):
	# P1 stands for the right boundary of 0's
	p0 = 0
	# P2 stands for the left boundary of 2's
	p2 = len(nums) - 1
	# this pointer is for the current item
	curr = 0
	while curr <= p2:
		# Swap curr with p2 if curr is 2
		if nums[curr] == 2:
			nums[curr], nums[p2] = nums[p2], nums[curr]
			p2 -= 1
		# Swap curr with p1 if curr is 0
		elif nums[curr] == 0:
			nums[curr], nums[p0] = nums[p0], nums[curr]
			p0 += 1
			curr += 1
		# Move curr if it is 1
		else:
			curr += 1

# Test
myList1 = [2,1,2]
myList2 = [2,0,2,1,1,0]
sortColors(myList1)
sortColors(myList2)
print(myList1)
print(myList2)