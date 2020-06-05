'''
########################################### Greedy ######################
455. Assign Cookies
435. Non-overlapping Intervals
452. Minimum Number of Arrows to Burst Balloons
406. Queue Reconstruction by Height
121. Best Time to Buy and Sell Stock
122. Best Time to Buy and Sell Stock II
605. Can Place Flowers
392. Is Subsequence
665. Non-decreasing Array
53. Maximum Subarray
763. Partition Labels

'''


'''
455. Assign Cookies
Easy

Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie. Each child i has a greed factor gi, which is the minimum size of a cookie that the child will be content with; and each cookie j has a size sj. If sj >= gi, we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

Note:
You may assume the greed factor is always positive.
You cannot assign more than one cookie to one child.

Example 1:
Input: [1,2,3], [1,1]

Output: 1

Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.
Example 2:
Input: [1,2], [1,2,3]

Output: 2

Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
You have 3 cookies and their sizes are big enough to gratify all of the children, 
You need to output 2.

'''

# I will sort both list, and find the greedy solution starting from the Child with lowest requirement
def findContentChildren(g: [int], s: [int]):
	g.sort()
	s.sort()
	child = 0
	cookie = 0
	result = 0
	while child < len(g) and cookie < len(s):
		if g[child] <= s[cookie]:
			result += 1
			cookie += 1
			child += 1
		else: 
			cookie += 1
	return result

# Test
print(findContentChildren([1,2], [1,2,3]))


'''
435. Non-overlapping Intervals
Medium

Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Example 1:

Input: [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of intervals are non-overlapping.
Example 2:

Input: [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of intervals non-overlapping.
Example 3:

Input: [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
 
Note:

You may assume the interval's end point is always bigger than its start point.
Intervals like [1,2] and [2,3] have borders "touching" but they don't overlap each other.

'''

# I am using the greedy algorithm to solve this problem. 
# Dynamic programming can also be accepted, but will be a bit slower O(n^2)

# Time: O(nlog(n)) because of the sorting
# Space: O(1)
def eraseOverlapIntervals(intervals: [[int]]):
	if len(intervals) <= 1:
		return 0
	# sort the intervals based on their start point
	intervals.sort(key = lambda x: x[0])
	result = 0
	# get the first interval as the prev for comparison
	prev = intervals[0]

	for item in range(1, len(intervals)):
		curr = intervals[item]
		# if there is an overlap
		if prev[1] > curr[0]:
			# remove prev interval if the previous one is totally covering up the whole curr one, and update prev
			if prev[1] >= curr[1]:
				prev = curr
			# keep prev and remove curr if they are only partially overlapped
			result += 1
		else:
			# update prev if there is no overlap
			prev = curr

	return result

# Test
print(eraseOverlapIntervals([[0,2],[1,3],[2,4],[3,5],[4,6]]))


'''
452. Minimum Number of Arrows to Burst Balloons
Medium

There are a number of spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter and hence the x-coordinates of start and end of the diameter suffice. Start is always smaller than end. There will be at most 104 balloons.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps travelling up infinitely. The problem is to find the minimum number of arrows that must be shot to burst all balloons.

Example:

Input:
[[10,16], [2,8], [1,6], [7,12]]

Output:
2

Explanation:
One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).

'''

# This question is almost the same as last one, with different requirement of increasing result count  
# Time: O(nlogn)  Space: O(1)
def findMinArrowShots(points: [[int]]):
	if len(points) <= 1:
		return len(points)

	points.sort(key = lambda x: x[0])
	result = 1
	prev = points[0]

	for item in range(1, len(points)):
		curr = points[item]
		if prev[1] >= curr[0]:
			if prev[1] >= curr[1]:
				prev = curr
		else:
			prev = curr
			result += 1

	return result

print(findMinArrowShots([[10,16], [2,8], [1,6], [7,12]]))
print(findMinArrowShots([[1,2],[2,3],[3,4],[4,5]]))


'''
406. Queue Reconstruction by Height
Medium

Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue.

Note:
The number of people is less than 1,100.

 
Example

Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

'''

''' Algorithm:
1. Sort people:
	a. In the descending order by height.
	b. Among the guys of the same height, in the ascending order by k-values.
2. Take guys one by one, and place them in the output array at the indexes equal to their k-values.
3. Return output array.
'''

# Time: O(n^2) as insertion takes n time for each person, and we do n people
# Space: O(1) I believe it doesn't take extra space as we returned the result list as output
#	However, on LeetCode's solution, they say it takes O(n) space
def reconstructQueue(people: [[int]]):
	people.sort(key = lambda x: (-x[0], x[1]))
	result = []
	for person in people:
		result.insert(person[1], person)
	return result

print(reconstructQueue([[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]))


'''
121. Best Time to Buy and Sell Stock
Easy

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''
# this one is pretty easy

# Time: O(n)  Space: O(1)
def maxProfit(prices: [int]):
	if len(prices) == 0:
		return 0
	min = prices[0]
	maxProfit = 0
	for i in range(1, len(prices)):
		if prices[i] > min:
			maxProfit = max(maxProfit, prices[i] - min)
		elif prices[i] < min:
			min = prices[i]
	return maxProfit

print(maxProfit([7,1,5,3,6,4]))
print(maxProfit([7,6,4,3,1]))


'''
122. Best Time to Buy and Sell Stock II
Easy

Say you have an array prices for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Example 2:

Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
 

Constraints:

1 <= prices.length <= 3 * 10 ^ 4
0 <= prices[i] <= 10 ^ 4
'''
# this one is pretty easy too

# Time: O(n)  Space: O(1)
def maxProfit(prices: [int]):
	if len(prices) == 0:
		return 0
	prev = prices[0]
	result = 0
	for i in range(1, len(prices)):
		if prices[i] > prev:
			result += prices[i] - prev
		prev = prices[i]
	return result

print(maxProfit([7,1,5,3,6,4]))
print(maxProfit([1,2,3,4,5]))


'''
605. Can Place Flowers
Easy

Suppose you have a long flowerbed in which some of the plots are planted and some are not. However, flowers cannot be planted in adjacent plots - they would compete for water and both would die.

Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty), and a number n, return if n new flowers can be planted in it without violating the no-adjacent-flowers rule.

Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: True
Example 2:
Input: flowerbed = [1,0,0,0,1], n = 2
Output: False
Note:
The input array won't violate no-adjacent-flowers rule.
The input array size is in the range of [1, 20000].
n is a non-negative integer which won't exceed the input array size.
'''
# I used a single scan to find all the available spaces and minus 1 from n everytime I find a slot

# Time: O(n)  Space: O(1)
def canPlaceFlowers(flowerbed: [int], n: int):
	# insert two empty slots at beginning and end of array to consider edge cases
	flowerbed.insert(0, 0)
	flowerbed.append(0)
	space = 0
	for b in flowerbed:
		# reset the current slot if we encounter a 1
		if b == 1:
			space = 0
		# increase the current space size if we see 0
		if b == 0:
			space += 1
			# insert one plant once our space is big enough, and reset
			if space == 3:
				n -= 1
				space = 1
		# we succeed, if n is back to 0, means we have no more plant left
		if n == 0:
			return True
	return False

print(canPlaceFlowers([1,0,0,0,1], 1))
print(canPlaceFlowers([1,0,0,0,1], 2))
print(canPlaceFlowers([1,0,0,0,0,1], 2))
print(canPlaceFlowers([1,0,0,0,1,0,0], 2))


'''
392. Is Subsequence
Easy

Given a string s and a string t, check if s is subsequence of t.

You may assume that there is only lower case English letters in both s and t. t is potentially a very long (length ~= 500,000) string, and s is a short string (<=100).

A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ace" is a subsequence of "abcde" while "aec" is not).

Example 1:
s = "abc", t = "ahbgdc"

Return true.

Example 2:
s = "axc", t = "ahbgdc"

Return false.

Follow up:
If there are lots of incoming S, say S1, S2, ... , Sk where k >= 1B, and you want to check one by one to see if T has its subsequence. In this scenario, how would you change your code?
'''
# I used two pointers for simplicity. For the follow up, we can create a dictionary
#	with t, storing 26 chars as key, and their occurring indexes as the value. Then we 
#	loop through s, and find out the earliest occurance for each letter, after the occurence
#	of the previous one. If one char runs out of occurance, we return false. 
#	*** Doing this will keep the time of checking each element lower after the initial setup

# Time: O(n+k) where k is length of s. Space: O(1)
def isSubsequence(s: str, t: str):
	if len(s) == 0:
		return True
	if len(t) == 0:
		return False
	curr = 0
	for item in t:
		if s[curr] == item:
			curr += 1
		if curr == len(s):
			return True
	return False

print(isSubsequence('abc', 'ahbgdc'))
print(isSubsequence('axc', 'ahbgdc'))


'''
665. Non-decreasing Array
Easy

Given an array nums with n integers, your task is to check if it could become non-decreasing by modifying at most 1 element.

We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for every i (0-based) such that (0 <= i <= n - 2).

 

Example 1:

Input: nums = [4,2,3]
Output: true
Explanation: You could modify the first 4 to 1 to get a non-decreasing array.
Example 2:

Input: nums = [4,2,1]
Output: false
Explanation: You can't get a non-decreasing array by modify at most one element.
 

Constraints:

1 <= n <= 10 ^ 4
- 10 ^ 5 <= nums[i] <= 10 ^ 5
'''

# Time: O(n)  Space: O(1)
def checkPossibility(nums: [int]):
	if len(nums) <= 2:
		return True
	count = 0
	if nums[1] < nums[0]:
		count = 1
	# if curr is smaller than the prev, we should try to set prev = curr so that we don't 
	# affect anything after curr. However, if curr is even smaller than the 2nd previous 
	# one, we will have to set curr = prev, as otherwise it will still be smaller than
	# something in the front
	for i in range(2, len(nums)):
		if nums[i] < nums[i - 1]:
			if nums[i] < nums[i - 2]:
				nums[i] = nums[i - 1]
			else:
				nums[i - 1] = nums[i]
			count += 1
		if count >= 2:
			return False
	return True

print('665. Non-decreasing Array')
print(checkPossibility([4,2,3]))
print(checkPossibility([4,2,1]))


'''
53. Maximum Subarray
Easy

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Follow up:

If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
'''

# this one I used dynamic programming
# Time: O(n)  Space: O(1)
def maxSubArray(nums: [int]):
	if len(nums) == 0:
		return 0 
	maxSub = nums[0]
	maxTilCurr = nums[0]
	for i in range(1, len(nums)):
		maxTilCurr = max(maxTilCurr + nums[i], nums[i])
		maxSub = max(maxTilCurr, maxSub)
	return maxSub

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


'''
763. Partition Labels
Medium

A string S of lowercase letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

Example 1:
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
Note:

S will have length in range [1, 500].
S will consist of lowercase letters ('a' to 'z') only.
'''

# Time: O(n)  Space: O(1) as we at most have 26 different char 
def partitionLabels(S: str):
	# this automatically creates a dictionary for all char with their last occurance
	last = {c: i for i, c in enumerate(S)}
	p = 0
	# we set this anchor so we can calculate the size of partitions
	anchor = 0
	result = []
	# use enumerate to create all char with their indexes in S
	for i, c in enumerate(S):
		# p is the last occurance of anything that we've met in the current partition
		p = max(p, last[c])
		# if p == i, we've get to a point where we've included all occurance in the current partition
		if p == i:
			result.append(i - anchor + 1)
			anchor = i + 1
	return result

print(partitionLabels('ababcbacadefegdehijhklij'))