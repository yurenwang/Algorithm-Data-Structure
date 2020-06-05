'''
############################################### Bit Manipulation #####################

0. 原理
461. Hamming Distance
136. Single Number
268. Missing Number
137. Single Number II
260. Single Number III
190. Reverse Bits
231. Power of Two
342. Power of Four
693. Binary Number with Alternating Bits
476. Number Complement
371. Sum of Two Integers
318. Maximum Product of Word Lengths
338. Counting Bits

'''

'''
0. 原理
基本原理

0s 表示一串 0，1s 表示一串 1。

x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x = 0       x & x = x       x | x = x
利用 x ^ 1s = ~x 的特点，可以将一个数的位级表示翻转；利用 x ^ x = 0 的特点，可以将三个数中重复的两个数去除，只留下另一个数。

1^1^2 = 2
利用 x & 0s = 0 和 x & 1s = x 的特点，可以实现掩码操作。一个数 num 与 mask：00111100 进行位与操作，只保留 num 中与 mask 的 1 部分相对应的位。

01011011 &
00111100
--------
00011000
利用 x | 0s = x 和 x | 1s = 1s 的特点，可以实现设值操作。一个数 num 与 mask：00111100 进行位或操作，将 num 中与 mask 的 1 部分相对应的位都设置为 1。

01011011 |
00111100
--------
01111111
位与运算技巧

n&(n-1) 去除 n 的位级表示中最低的那一位 1。例如对于二进制表示 01011011，减去 1 得到 01011010，这两个数相与得到 01011010。

01011011 &
01011010
--------
01011010
n&(-n) 得到 n 的位级表示中最低的那一位 1。-n 得到 n 的反码加 1，也就是 -n=~n+1。例如对于二进制表示 10110100，-n 得到 01001100，相与得到 00000100。

10110100 &
01001100
--------
00000100
n-(n&(-n)) 则可以去除 n 的位级表示中最低的那一位 1，和 n&(n-1) 效果一样。

移位运算

>> n 为算术右移，相当于除以 2n，例如 -7 >> 2 = -2。

11111111111111111111111111111001  >> 2
--------
11111111111111111111111111111110
>>> n 为无符号右移，左边会补上 0。例如 -7 >>> 2 = 1073741822。

11111111111111111111111111111001  >>> 2
--------
00111111111111111111111111111111
<< n 为算术左移，相当于乘以 2n。-7 << 2 = -28。

11111111111111111111111111111001  << 2
--------
11111111111111111111111111100100
mask 计算

要获取 111111111，将 0 取反即可，~0。

要得到只有第 i 位为 1 的 mask，将 1 向左移动 i-1 位即可，1<<(i-1) 。例如 1<<4 得到只有第 5 位为 1 的 mask ：00010000。

要得到 1 到 i 位为 1 的 mask，(1<<i)-1 即可，例如将 (1<<4)-1 = 00010000-1 = 00001111。

要得到 1 到 i 位为 0 的 mask，只需将 1 到 i 位为 1 的 mask 取反，即 ~((1<<i)-1)。

Java 中的位操作

static int Integer.bitCount();           // 统计 1 的数量
static int Integer.highestOneBit();      // 获得最高位
static String toBinaryString(int i);     // 转换为二进制表示的字符串
'''


'''
461. Hamming Distance
Easy

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, calculate the Hamming distance.

Note:
0 ≤ x, y < 231.

Example:

Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

The above arrows point to positions where the corresponding bits are different.
'''

class Solution:
    # Time: O(1)  Space: O(1)
    def hammingDistance(self, x: int, y: int) -> int:
        # Three Solutions: 
        #   1. Built in bit count function.
        #   2. Find all differences using z = x ^ y, then keep shifting z to the right by 1 bit and check
        #       the last digit. Count += 1 if it is 1.
        #   3. Find all differences using z = x ^ y, then keep removing the last 1 using z & (z - 1) and
        #       increase count by 1 each time. 
        
        # Solution 3:
        z = x ^ y
        count = 0
        while z:
            z &= (z - 1)
            count += 1
        return count


'''
136. Single Number
Easy

Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:

Input: [2,2,1]
Output: 1
Example 2:

Input: [4,1,2,1,2]
Output: 4
'''

class Solution2:
    # Bit Manipulation solution
    # For two same integers a and b, a^b == 0. After we xor all integers in the list, we are left with 
    #   1 last item that is not duplicated
    
    # Time: O(N)  Space: O(1)
    def singleNumber(self, nums: [int]) -> int:
        curr = 0
        for n in nums:
            curr ^= n
        return curr


'''
268. Missing Number
Easy

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

Example 1:

Input: [3,0,1]
Output: 2
Example 2:

Input: [9,6,4,2,3,5,7,0,1]
Output: 8
Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?
'''

class Solution3:
    def missingNumber(self, nums: [int]) -> int:
        # Three solutions:
        '''
        1. Use a Set. Put everything in list in, then loop through each item in list to check if it 
            is not in it.   Time: O(N)  Space: O(N)
        '''
        # s = set(nums)
        # for n in range(len(nums) + 1):
        #     if n not in s:
        #         return n
        
        '''
        2. Math. Calculate the sum using (1 + N) * N / 2. The subtract every n in nums from it. The 
            remaining value would be result.    Time: O(N)  Space: O(1)
        '''
        # n = len(nums)
        # s = (1 + n) * n // 2
        # for v in nums: 
        #     s -= v
        # return s
    
        '''
        3. Bit Manipulation. 
        For the list with index table below:
            Index: 0  1  2  3  4
            Value: 0  1  3  4  5
        We can see that for each value v, there is a matching index i, except for N, and the missing 
        values's index. Therefore, we can use N to xor all values and all indexes, and the remaining 
        thing will be the missing value.    Time: O(N)  Space: O(1)
        '''
        n = len(nums)
        for i, v in enumerate(nums):
            n ^= i ^ v
        return n


'''
137. Single Number II
Medium

Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:

Input: [2,2,3,2]
Output: 3
Example 2:

Input: [0,1,0,1,0,1,99]
Output: 99
'''

class Solution4:
    # Time: O(N)  Space: O(1)
    def singleNumber(self, nums: [int]) -> int:
        # Bit Manipulation
        # We need two bitmasks. We only modify the first one if the second is not modified, and only
        #   modify the second one if the first one is not modified. This way, if a number n occurs 3
        #   times, the first time mask1 will be modified to n; the second time mask1 will be modified 
        #   back to 0, and mask2 will be modified to n (because mask1 now is unchanged); and the third
        #   time mask1 will remain 0, and mask2 modified back to 0
        
        # This way, we separate nums that appears 1 time from those with 3 times, as a number that appears
        #   only one time will leave mask1 modified to n, and one that appears 3 times will leave mask1
        #   unchanged, as 0
        
        mask1 = 0
        mask2 = 0
        for n in nums:
            mask1 = ~mask2 & (mask1 ^ n)
            mask2 = ~mask1 & (mask2 ^ n)
        return mask1



'''
260. Single Number III
Medium

Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.

Example:

Input:  [1,2,1,3,2,5]
Output: [3,5]
Note:

The order of the result is not important. So in the above example, [5, 3] is also correct.
Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?
'''

class Solution5:
    def singleNumber(self, nums: [int]) -> [int]:
        # Two solutions
        '''
        HashMap (Dictionary) solution:
        Time: O(N)  Space: O(N)
        '''
        # from collections import Counter
        # count = Counter(nums)
        # return [n for n in count if count[n] == 1]
    
        '''
        Bit Manipulation solution
        1. First loop through the whole list with a bitmask and xor operations. We will get a value which
        is x ^ y, because x and y are the only two values that appears odd number of times.
        This way, we know the difference between x and y. 
        2. Then, we use lastOne = bitmask & (-bitmask) to get the last '1' in the bitmask. (x & (-x))
        always returns the last '1' in x
        3. Then we loop the nums again, but this time only nums with '1' at the same position as the 
        lastOne we got last step can enter the modification. This way, all others will not affect the 
        bitmask, as they always appear even mumber of times. However, since we are guaranteed that one
        of the x or y will not have '1' here, only one number can enter this loop and make changes to 
        the bitmask
        4. So we get the first number, as bitmask. Then we can get the second one easily with 
        second = bitmask ^ first
        
        Time: O(N)  Space: O(1)
        '''
        diff = 0
        for n in nums:
            diff ^= n
        
        lastOne = diff & (-diff)

        first = 0
        for n in nums:
            if n & lastOne:
                first ^= n
        
        return [first, first ^ diff]
        
        
'''
190. Reverse Bits
Easy

Reverse bits of a given 32 bits unsigned integer.


Example 1:

Input: 00000010100101000001111010011100
Output: 00111001011110000010100101000000
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
Example 2:

Input: 11111111111111111111111111111101
Output: 10111111111111111111111111111111
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
 

Note:

Note that in some languages such as Java, there is no unsigned integer type. In this case, both input and output will be given as signed integer type and should not affect your implementation, as the internal binary representation of the integer is the same whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above the input represents the signed integer -3 and the output represents the signed integer -1073741825.
 

Follow up:

If this function is called many times, how would you optimize it?
'''

class Solution6:
    # Time: O(log2 N)  Space: O(1)
    def reverseBits(self, n: int) -> int:
        # Bit Manipulation
        # I will use bit by bit operations for this solution. However, the optimized is to use shift and
        #   mask like solution #3 which only takes O(1) time.

		# If function need to be called many times, we can seperate n into 4 bytes, then reverse each byte,
		#	then combine.
        
        # Starting from the last bit. For bit originally at ith place from right, put it at ith place
        #   from left instead
        
        result = 0
        power = 31
        while power >= 0:
            last = n & 1
            n = n >> 1
            result += (last << power)
            power -= 1
        return result


'''
231. Power of Two
Easy

Given an integer, write a function to determine if it is a power of two.

Example 1:

Input: 1
Output: true 
Explanation: 20 = 1
Example 2:

Input: 16
Output: true
Explanation: 24 = 16
Example 3:

Input: 218
Output: false
'''

class Solution7:
    # Simple bit manipulation
    
    # Time: O(1)  Space: O(1)
    def isPowerOfTwo(self, n: int) -> bool:
        # return n > 0 and bin(n).count('1') == 1
        
        return n > 0 and n & (n - 1) == 0

	
'''
342. Power of Four
Easy

Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

Example 1:

Input: 16
Output: true
Example 2:

Input: 5
Output: false
Follow up: Could you solve it without loops/recursion?
'''

class Solution8:
    # Time: O(1)  Space: O(1)
    def isPowerOfFour(self, num: int) -> bool:
        # Using iteration or recursion is obvious, so I will not include it here
        
        # Bit Manipulation solution:
        # Num nees to be > 0, it only should has one '1' in the bit representation, and that digit has
        #   to be on even positions. 
        
        # We get all even positions using n & 0xaaaaaaaa (which is 1010101010101010)
        return num > 0 and (num & (num - 1) == 0) and (num & 0xaaaaaaaa == 0)


'''
693. Binary Number with Alternating Bits
Easy

Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have different values.

Example 1:
Input: 5
Output: True
Explanation:
The binary representation of 5 is: 101
Example 2:
Input: 7
Output: False
Explanation:
The binary representation of 7 is: 111.
Example 3:
Input: 11
Output: False
Explanation:
The binary representation of 11 is: 1011.
Example 4:
Input: 10
Output: True
Explanation:
The binary representation of 10 is: 1010.
'''

class Solution9:
    # Time: O(1)  We have a loop but n's size is kept at 32 digit. So it is a linear time
    # Space: O(1)
    def hasAlternatingBits(self, n: int) -> bool:
        # Keep tracking the last 2 digits of bit representation of n. 
        # The last digit = n & 1
        # The second last digit we get from our last iteration
        last = n & 1
        n >>= 1   # shift one digit so we drop the previous last
        while n: 
            if n & 1 == last:
                return False
            last = n & 1
            n >>= 1
        return True
        

'''
476. Number Complement
Easy

Given a positive integer num, output its complement number. The complement strategy is to flip the bits of its binary representation.
 

Example 1:

Input: num = 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
Example 2:

Input: num = 1
Output: 0
Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.
 

Constraints:

The given integer num is guaranteed to fit within the range of a 32-bit signed integer.
num >= 1
You could assume no leading zero bit in the integer’s binary representation.
This question is the same as 1009: https://leetcode.com/problems/complement-of-base-10-integer/
'''

class Solution10:
    # This question is the same as https://leetcode.com/problems/complement-of-base-10-integer/submissions/
    
    # Time: O(1)  Space: O(1)
    def findComplement(self, num: int) -> int:
        # Construct a bitmask with all 1's and flip num using bitmast ^ num
        bitmask = num
        bitmask |= (bitmask >> 1)
        bitmask |= (bitmask >> 2)
        bitmask |= (bitmask >> 4)
        bitmask |= (bitmask >> 8)
        bitmask |= (bitmask >> 16)
        
        return bitmask ^ num


'''
371. Sum of Two Integers
Easy

Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = -2, b = 3
Output: 1
'''

class Solution11:
    # This is a solution copied from the Discussion board. 
    # The only difference between his and my solution, is that in his solution, he gets the max and min
    #   integer number, and make it as a mask to limit the recursion of b. My own solution will work
    #   for languages like Java but not for Python
    def getSum(self, a: int, b: int) -> int:
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        while b != 0:
            # ^ get different bits and & gets double 1s, << moves carry
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return a if a <= MAX else ~(a ^ mask)
        

'''
318. Maximum Product of Word Lengths
Medium

Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. You may assume that each word will contain only lower case letters. If no such two words exist, return 0.

Example 1:

Input: ["abcw","baz","foo","bar","xtfn","abcdef"]
Output: 16 
Explanation: The two words can be "abcw", "xtfn".
Example 2:

Input: ["a","ab","abc","d","cd","bcd","abcd"]
Output: 4 
Explanation: The two words can be "ab", "cd".
Example 3:

Input: ["a","aa","aaa","aaaa"]
Output: 0 
Explanation: No such pair of words.
'''

class Solution12:
    # Time: O(N*N + L) where N is number of words we have in our input, and L is the total length of all
    #   words. (when N < 2^26. When N > 2^26, since that's the max number of bitmasks we can possibly
    #   have, the time is O(2^26*2^26 + L) = O(L))
    # Space: O(N). (when N < 2^26. When N > 2^26, since that's the max number of bitmasks we can possibly
    #   have, the time is O(2^26) = O(1))
    def maxProduct(self, words: [str]) -> int:
        # Since words only contian lower cases, we can create a bitmask of length 26 for each word.
        
        # First, we process the words list and create a bitmask for each word. In the bitmask, from right
        #   to left, we use '1' if a char exists, and '0' if not. For example, word 'dad' would be like
        #   '000....0001001'. Time complexity of this step will be O(L), where L is the total length of
        #   all words in the input list.
        # At the same time, since for ex, word 'aabb' will have the same bitmask as 'ab', we put our 
        #   bitmasks into a dictionary, which records the max length of each bitmask, because we only want
        #   to calculate the max length
        # Then, we use two nested loops to iterate throught the words list, and update the result every
        #   time we see two words that do not share the same bitmask. Time complexity of this step will
        #   be O(N*N), where N is the num of words we have in our input.
        
        
        # Create bitmasks:
        def createMask(word):
            result = 0
            for c in word:
                result |= (1 << ord(c) - ord('a'))
            return result
                
        # Put bitmasks in hashmaps:
        maxLength = {}
        for w in words:
            mask = createMask(w)
            maxLength[mask] = max(maxLength.get(mask, 0), len(w))
        
        # Compare words to each other to get result:
        result = 0
        for i in maxLength:
            for j in maxLength:
                if i & j == 0:
                    result = max(result, maxLength[i] * maxLength[j])
        return result


'''
338. Counting Bits
Medium

Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
Follow up:

It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
Space complexity should be O(n).
Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.
'''

class Solution13:
    # Time: O(N)    Space: O(N)
    def countBits(self, num: int) -> [int]:
        # As we know, we can remove the last bit of a number n by doing n & (n - 1). 
        # Therefore, for each number n, we know that the count of 1s of it is 1 + m where m = n&(n-1)
        # We can use Dynamic programming to calculate the counts
        
        counts = [0] * (num + 1)
        for i in range(1, num + 1):
            counts[i] = counts[i & (i - 1)] + 1
        return counts
