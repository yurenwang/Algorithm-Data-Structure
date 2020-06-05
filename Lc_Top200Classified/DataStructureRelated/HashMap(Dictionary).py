'''
###################################### Dictionary #######################

1. Two Sum
217. Contains Duplicate
594. Longest Harmonious Subsequence
128. Longest Consecutive Sequence

'''

'''
1. Two Sum
Easy

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
'''

class Solution:
    def twoSum(self, nums: [int], target: int) -> [int]:
        # Use dictionary to store complement of a value and its index
        d = {}
        for i in range(len(nums)):
            if nums[i] in d:
                return [d.get(nums[i]), i]
            d[target - nums[i]] = i


'''
217. Contains Duplicate
Easy

Given an array of integers, find if the array contains any duplicates.

Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

Example 1:

Input: [1,2,3,1]
Output: true
Example 2:

Input: [1,2,3,4]
Output: false
Example 3:

Input: [1,1,1,3,3,4,3,2,4,2]
Output: true
'''

class Solution2:
    def containsDuplicate(self, nums: [int]) -> bool:
        # Use Set
        s = set()
        for n in nums:
            if n in s:
                return True
            s.add(n)


'''
594. Longest Harmonious Subsequence
Easy

We define a harmounious array as an array where the difference between its maximum value and its minimum value is exactly 1.

Now, given an integer array, you need to find the length of its longest harmonious subsequence among all its possible subsequences.

Example 1:

Input: [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].
 

Note: The length of the input array will not exceed 20,000.
'''

class Solution3:
    # Time: O(N)  Space: O(N)
    def findLHS(self, nums: [int]) -> int:
        result = 0
        
        # Build the dict first. Dict contains the count of each number.
        d = {}
        for n in nums:
            d[n] = d.get(n, 0) + 1
        
        # Then loop through the dict and check each d[i] + d[i+1]
        for k in d.keys():
            if k + 1 in d.keys():
                result = max(result, d.get(k) + d.get(k + 1))
                
        return result


'''
128. Longest Consecutive Sequence
Hard

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

Example:

Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
'''

class Solution4:
    # Time: O(N)  Space: O(N)
    def longestConsecutive(self, nums: [int]) -> int:
        # Uses Set
        # First put everyting in the set for quick look up
        # Then loop through the set and iteratively find i + 1 to build sequences. However, we only want
        #   to loop through those with i - 1 not in the set so that we only start the loop when it can 
        #   form a brand new sequence. This way, we only loop through everything once, or O(N)
        numsSet = set(nums)
        result = 0
        for n in nums:
            if n - 1 not in numsSet:
                currStrike = 1
                curr = n
                while curr + 1 in numsSet:
                    currStrike += 1
                    curr += 1
                result = max(result, currStrike)
        return result

