'''
#################################################### Math #######################
### Math problems are usually pretty easy. The only thing is to review the bit manipulation methods.

504. Base 7
405. Convert a Number to Hexadecimal
168. Excel Sheet Column Title
172. Factorial Trailing Zeroes
67. Add Binary  (use bit manipulation)
415. Add Strings
462. Minimum Moves to Equal Array Elements II
169. Majority Element
367. Valid Perfect Square
326. Power of Three
238. Product of Array Except Self
628. Maximum Product of Three Numbers

'''


'''
504. Base 7
Easy

Given an integer, return its base 7 string representation.

Example 1:
Input: 100
Output: "202"
Example 2:
Input: -7
Output: "-10"
Note: The input will be in range of [-1e7, 1e7].
'''

# Math problem
def convertToBase7(num: int):
	if num == 0:
		return "0"
	res = ''
	n = abs(num)
	while n > 0:
		res = str(n % 7) + res
		n //= 7
	return res if num >= 0 else '-' + res


'''
405. Convert a Number to Hexadecimal
Easy

Given an integer, write an algorithm to convert it to hexadecimal. For negative integer, two’s complement method is used.

Note:

All letters in hexadecimal (a-f) must be in lowercase.
The hexadecimal string must not contain extra leading 0s. If the number is zero, it is represented by a single zero character '0'; otherwise, the first character in the hexadecimal string will not be the zero character.
The given number is guaranteed to fit within the range of a 32-bit signed integer.
You must not use any method provided by the library which converts/formats the number to hex directly.
Example 1:

Input:
26

Output:
"1a"
Example 2:

Input:
-1

Output:
"ffffffff"
'''

# Math problem
# this is a smart solution found in discussion board on leetcode
def toHex(num: int) -> str:
	if num == 0: return '0'
	m = '0123456789abcdef'  # use this as a map
	
	ans = ''
	for _ in range(8):
		ans = m[num & 15] + ans  # 'n & 15' is equal to 'n % 16'
		num >>= 4  # bit manipulation. shift right for 4 digits. This is equal to num //= 16
	
	return ans.lstrip('0')  # remove leading 0's

print(toHex(26))
print(toHex(259))


'''
168. Excel Sheet Column Title
Easy

Given a positive integer, return its corresponding column title as appear in an Excel sheet.

For example:

    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
    ...
Example 1:

Input: 1
Output: "A"

Example 2:

Input: 28
Output: "AB"

Example 3:

Input: 701
Output: "ZY"
'''
# Math
def convertToTitle(n: int):
	# we don't need to store the letter map, as we can use ord(c) to find out the unicode presentation of
	# character c, and use chr(i) to get the char at unicode presentation of i 
	result = []
	while n > 0:
		result.append(chr(ord('A') + (n - 1) % 26))
		n = (n - 1) // 26
	result.reverse()
	return ''.join(result)

print(convertToTitle(28))
print(convertToTitle(701))


'''
172. Factorial Trailing Zeroes
Easy

Given an integer n, return the number of trailing zeroes in n!.

Example 1:

Input: 3
Output: 0
Explanation: 3! = 6, no trailing zero.
Example 2:

Input: 5
Output: 1
Explanation: 5! = 120, one trailing zero.
Note: Your solution should be in logarithmic time complexity.
'''

# This is a stupid math question. However, the solution of this question is very well explained.
# Read through the solution to review this question

# Solution:
# https://leetcode.com/problems/factorial-trailing-zeroes/solution/


'''
67. Add Binary
Easy

Given two binary strings, return their sum (also a binary string).

The input strings are both non-empty and contains only characters 1 or 0.

Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
 

Constraints:

Each string consists only of '0' or '1' characters.
1 <= a.length, b.length <= 10^4
Each string is either "0" or doesn't contain any leading zero.
'''
# this is a math problem
# I iterate the two strings from last index to the first to calculate the sum. Make sure to consider the
#  carried digit 

def addBinary(a: str, b: str):
	carry = 0
	resultList = []
	lenA = len(a) 
	lenB = len(b) 
	for i in range(1, max(lenA, lenB) + 1):
		if i > lenA:
			tmp = int(b[-i]) + carry
		elif i > lenB:
			tmp = int(a[-i]) + carry
		else:
			tmp = int(a[-i]) + int(b[-i]) + carry
		if tmp <= 1:
			resultList.append(str(tmp))
			carry = 0
		else:
			resultList.append(str(tmp % 2))
			carry = 1
	if carry == 1:
		resultList.append('1')
	resultList.reverse()
	result = ''.join(resultList)
	return result

print(addBinary('1010', '1011'))


################################
# Use bit manipulation to solve this question, in case we are not allowed to add 2 things numerically:

# without caring about the carry, sum of a and b can be represented as a ^ b (a xor b)  
#							a = 1 1 1 1
#							b = 0 0 1 0
# answer without carry  a ^ b = 1 1 0 1

# carry can be represented as a & b, then shift left 1 place
#							a = 1 1 1 1
#							b = 0 0 1 0
# carry  =     (a & b) << 1 = 0 0 1 0 0

# then this question is converted to: sum of the ans without carry, and the carry
# and we can use if carry == 0 to solve it
def addBinary2(a: str, b: str):
	x, y = int(a, 2), int(b, 2)
	while y:
		x, y = x ^ y, (x & y) << 1
	return bin(x)[2:]

print(addBinary('11', '1'))


'''
415. Add Strings
Easy

Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.

Note:

The length of both num1 and num2 is < 5100.
Both num1 and num2 contains only digits 0-9.
Both num1 and num2 does not contain any leading zero.
You must not use any built-in BigInteger library or convert the inputs to integer directly.
'''

# Math question 
# use integer presentation of chars to find out the value at each digit
def addStrings(num1: str, num2: str):
	l1, l2 = list(num1), list(num2)
	resList = []
	carry = 0
	while len(l1) > 0 or len(l2) > 0:
		a = ord(l1.pop()) - ord('0') if len(l1) > 0 else 0
		b = ord(l2.pop()) - ord('0') if len(l2) > 0 else 0
		tmpRes = a + b + carry 
		resList.append(tmpRes % 10)
		carry = tmpRes // 10
	if carry: resList.append(carry)
	return ''.join([str(x) for x in resList])[::-1]

print(addStrings('100', '111'))
print(addStrings('800', '211'))


################################################ 相遇问题 ####################

'''
462. Minimum Moves to Equal Array Elements II
Medium

Given a non-empty integer array, find the minimum number of moves required to make all array elements equal, where a move is incrementing a selected element by 1 or decrementing a selected element by 1.

You may assume the array's length is at most 10,000.

Example:

Input:
[1,2,3]

Output:
2

Explanation:
Only two moves are needed (remember each move increments or decrements one element):

[1,2,3]  =>  [2,2,3]  =>  [2,2,2]
'''

# Math
def minMoves2(nums: [int]):
	# In my solution, I sort the list using merge sort or quick sort with time O(nlogn)
	#	then the median will be the settle point. 
	# Or we don't have to find the median if we use two pointers, because: 
	# 	(value_hi - median) + (median - value_low) == value_hi - value_low

	# There is a better solution to use Quick-select, but that is too complex for this question
	nums.sort()
	result = 0
	l, r = 0, len(nums) - 1
	while l < r: 
		result += nums[r] - nums[l]
		l += 1
		r -= 1
	return result

print(minMoves2([1,2,3]))
print(minMoves2([1,10,1]))


'''
169. Majority Element
Easy

Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

Example 1:

Input: [3,2,3]
Output: 3
Example 2:

Input: [2,2,1,1,1,2,2]
Output: 2
'''

# There are multiple ways to solve this question
#	1. We can sort the array, then median will be the majority element. Time: O(nlogn) 
#	2. We can use dictionary to count the occurance of each element. Time: O(n)


################ Other ###############

'''
367. Valid Perfect Square
Easy

Given a positive integer num, write a function which returns True if num is a perfect square else False.

Note: Do not use any built-in library function such as sqrt.

Example 1:

Input: 16
Output: true
Example 2:

Input: 14
Output: false
'''

def isPerfectSquare(num: int):
	# This is a binary search problem. 
	# We can set left = 2, and right = num/2
	#   then while l < r, we get middle = (l + r) / 2 and compare m * m == num
	#   if yes, then return true. 
	#   if no, then we move l or r accordingly depends on how m * m compares to num
	
	# However, we can also solve this problem mathematically.
	# We can find that the differences between two perfect squares has a pattern:
	# 0, 1, 4, 9, 16, 25 --> differences are: 1, 3, 5, 7, 9, ...
	diff = 1
	while num >= 0:
		if num == 0:
			return True
		num -= diff
		diff += 2
	return False


'''
326. Power of Three
Easy

Given an integer, write a function to determine if it is a power of three.

Example 1:

Input: 27
Output: true
Example 2:

Input: 0
Output: false
Example 3:

Input: 9
Output: true
Example 4:

Input: 45
Output: false
Follow up:
Could you do it without using any loop / recursion?
'''
def isPowerOfThree(n: int):
	# This should normally done easily with a loop or recursion. However,

	# Approach 4 is brilliant: 
	# 1162261467 is the largest int dividable by 3, and since 3 is a prime number, all its factors are 
	# 3^1, 3^2, 3^3, ... Therefore, as long as n can divide 1162261467, it is also a power of 3
	return n > 0 and 1162261467 % n == 0


'''
238. Product of Array Except Self
Medium

Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array (including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? (The output array does not count as extra space for the purpose of space complexity analysis.)
'''
# math
def productExceptSelf(nums: [int]):
	multLeft = 1
	result = []
	for i in range(0, len(nums)):
		result.append(multLeft)
		multLeft *= nums[i]
	multRight = 1
	for i in range(len(nums) - 1, -1, -1):
		result[i] *= multRight
		multRight *= nums[i]
	return result

print(productExceptSelf([1,2,3,4]))


'''
628. Maximum Product of Three Numbers
Easy

Given an integer array, find three numbers whose product is maximum and output the maximum product.

Example 1:

Input: [1,2,3]
Output: 6
 

Example 2:

Input: [1,2,3,4]
Output: 24
 

Note:

The length of the given array will be in range [3,104] and all elements are in the range [-1000, 1000].
Multiplication of any three numbers in the input won't exceed the range of 32-bit signed integer.
'''

# math
def maximumProduct(nums: [int]):
	# There are 2 possible ways of getting the max product:
	#	1. The 3 largest numbers in the list multiply together
	#	2. The 2 smallest numbers, which are negative nums, and 1 largest number multiplied together

	# We have two options: 
	#	1. Sort the list, then return max(nums[0]*nums[1]*nums[-1], nums[-1]*nums[-2]*nums[-3]). -> (O(nlogn))
	#	2. One scan, and keep track of the 3 largest numbers and 2 smallest numbers.  -> (O(n))

	# In this solution, I'll just use the first option to be simple
	nums.sort()
	return max(nums[0]*nums[1]*nums[-1], nums[-1]*nums[-2]*nums[-3])

print(maximumProduct([1,2,3,4]))