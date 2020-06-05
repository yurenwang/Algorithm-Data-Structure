'''
############################## Two Pointers ##########################
LeetCode: 
167. Two Sum II - Input array is sorted
633. Sum of Square Numbers
345. Reverse Vowels of a String
680. Valid Palindrome II
88. Merge Sorted Array
141. Linked List Cycle
524. Longest Word in Dictionary through Deleting

'''


'''
167. Two Sum II - Input array is sorted
Easy

Given a asc sorted integer array, find two numbers that add up to a target number

Your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.


'''

# Time: O(n)	Space: O(1)
def twoSum(numbers, target):
    pointer1 = 0
    pointer2 = len(numbers) - 1

    while pointer1 < pointer2:
        sum = numbers[pointer1] + numbers[pointer2]

        if sum < target:
            pointer1 += 1
        elif sum > target:
            pointer2 -= 1
        else:
            return [pointer1 + 1, pointer2 + 1]

    return []

print('167. Two Sum II - Input array is sorted')
print(twoSum([5, 6, 12, 20, 31, 40], 32))


'''
633. Sum of Square Numbers
Easy

Given a non-negative integer c, your task is to decide whether there're two integers a and b such that a^2 + b^2 = c.

Example 1:

Input: 5
Output: True
Explanation: 1 * 1 + 2 * 2 = 5
 
Example 2:

Input: 3
Output: False

'''

# Time: O(sqrt(c)), where c is the input integer	Space: O(1)
import math
def judgeSquareSum(c):
	num1 = 0
	num2 = math.floor(math.sqrt(c))

	while num1 <= num2:
		squareSum = num1 * num1 + num2 * num2

		if squareSum < c:
			num1 += 1
		elif squareSum > c:
			num2 -= 1
		else:
			return True

	return False

print('633. Sum of Square Numbers')
print(judgeSquareSum(5))
print(judgeSquareSum(4))
print(judgeSquareSum(3))
print(judgeSquareSum(2))


'''
345. Reverse Vowels of a String
Easy

Write a function that takes a string as input and reverse only the vowels of a string.

Example 1:

Input: "hello"
Output: "holle"
Example 2:

Input: "leetcode"
Output: "leotcede"
Note:
The vowels does not include the letter "y".

'''

# Time: O(n)  Space: O(n)
def reverseVowels(s):
	if len(s) == 0:
		return s

	ourList = list(s)
	vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])

	p1 = 0
	p2 = len(ourList) - 1

	while p1 < p2:
		while p1 < p2 and ourList[p1] not in vowels:
			p1 += 1
		while p1 < p2 and ourList[p2] not in vowels:
			p2 -= 1
		ourList[p1], ourList[p2] = ourList[p2], ourList[p1]
		p1 += 1
		p2 -= 1
	
	return ''.join(ourList)

print('345. Reverse Vowels of a String')
print(reverseVowels("hello"))
print(reverseVowels("leetcode"))


'''
680. Valid Palindrome II
Easy

Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.

Example 1:
Input: "aba"
Output: True
Example 2:
Input: "abca"
Output: True
Explanation: You could delete the character 'c'.
Note:
The string will only contain lowercase characters a-z. The maximum length of the string is 50000.

'''

# Time: O(n)  Space: O(1)
def validPalindrome(s: str):
	deleted = False
	p1 = 0
	p2 = len(s) - 1
	while p1 < p2:
		if s[p1] == s[p2]:
			p1 += 1
			p2 -= 1
		else:
			if deleted:
				return False
			else:
				# remove one char and check if remaining is palindrome
				return s[p1:p2] == s[p1:p2][::-1] or s[p1+1:p2+1] == s[p1+1:p2+1][::-1]
	return True

print('680. Valid Palindrome II')
print(validPalindrome('aba'))
print(validPalindrome('abca'))
print(validPalindrome('heleoh'))
print(validPalindrome('abcad'))


'''
88. Merge Sorted Array
Easy

Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:

The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
Example:

Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]

'''

# Time: O(m + n)  Space: O(m) we can make space usage to be constant if we insert nums1 from the end
def merge(nums1: [int], m: int, nums2: [int], n: int) -> None:
	p1 = 0
	p2 = 0
	nums1Copy = nums1[:m]
	nums1[:] = []

	while p1 < m and p2 < n:
		nums1Curr = nums1Copy[p1]
		nums2Curr = nums2[p2]
		if nums1Curr <= nums2[p2]:
			nums1.append(nums1Curr)
			p1 += 1
		else:
			nums1.append(nums2Curr)
			p2 += 1
	
	if p1 < m:
		nums1[p1+p2:] = nums1Copy[p1:]
	if p2 < n:
		nums1[p1+p2:] = nums2[p2:]


print('88. Merge Sorted Array')
input1 = [1,2,3,0,0,0]
input2 = [2,5,6]
merge(input1, 3, input2, 3)
print(input1)



'''
141. Linked List Cycle
Easy

Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.

Example 2:

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the first node.

Example 3:

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.


'''

# Time: O(n)  Space: O(1)
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def hasCycle(head: ListNode):
	if head == None or head.next == None:
		return False
	p1 = head
	p2 = head.next
	while p1 != p2:
		if p2 == None or p2.next == None:
			return False
		p1 = p1.next
		p2 = p2.next.next
	return True

print('141. Linked List Cycle')


'''
524. Longest Word in Dictionary through Deleting
Medium

Given a string and a string dictionary, find the longest string in the dictionary that can be formed by deleting some characters of the given string. If there are more than one possible results, return the longest word with the smallest lexicographical order. If there is no possible result, return the empty string.

Example 1:
Input:
s = "abpcplea", d = ["ale","apple","monkey","plea"]

Output: 
"apple"
Example 2:
Input:
s = "abpcplea", d = ["a","b","c"]

Output: 
"a"
Note:
All the strings in the input will only contain lower-case letters.
The size of the dictionary won't exceed 1,000.
The length of all the strings in the input won't exceed 1,000.

'''

# Time: O(x*n), where n is the len of d, and x is the average length of strings
# Space: O(x), where x is the length of the longest string
def findLongestWord(s: str, d: [str]) -> str:
	result = ''
	for item in d:
		if isSubSequence(s, item):
			if len(item) > len(result) or (len(item) == len(result) and item < result):
				result = item
	return result

# this is the helper function for checking if inner is subsequence of outer
def isSubSequence(outer, inner):
	p1 = 0
	p2 = 0
	while p1 < len(outer):
		if p2 >= len(inner):
			return True
		elif outer[p1] == inner[p2]:
			p1 += 1
			p2 += 1
		else:
			p1 += 1
	if p2 >= len(inner):
		return True
	return False

# Test
print("524. Longest Word in Dictionary through Deleting")
print(findLongestWord("bab", ["ba","ab","a","b"]))
print(findLongestWord("abpcplea", ["ale","apple","monkey","plea"]))
print("ab"<"ba")