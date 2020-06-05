'''
########################################### String ##########################

1. 字符串循环移位包含
2. 字符串循环移位 
3. 字符串中单词的翻转
242. Valid Anagram
409. Longest Palindrome
205. Isomorphic Strings
647. Palindromic Substrings
9. Palindrome Number
696. Count Binary Substrings

'''

'''
1. 字符串循环移位包含
编程之美 3.1

s1 = AABCD, s2 = CDAA
Return : true
给定两个字符串 s1 和 s2，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。

s1 进行循环移位的结果是 s1s1 的子字符串，因此只要判断 s2 是否是 s1s1 的子字符串即可。

2. 字符串循环移位 
编程之美 2.17

s = "abcd123" k = 3
Return "123abcd"
将字符串向右循环移动 k 位。

将 abcd123 中的 abcd 和 123 单独翻转，得到 dcba321，然后对整个字符串进行翻转，得到 123abcd。

3. 字符串中单词的翻转
程序员代码面试指南

s = "I am a student"
Return "student a am I"
将每个单词翻转，然后将整个字符串翻转。
'''



'''
242. Valid Anagram
Easy

Given two strings s and t , write a function to determine if t is an anagram of s.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters? How would you adapt your solution to such case?
'''

class Solution:
    # Time: O(N)  Space: O(1)
    def isAnagram(self, s: str, t: str) -> bool:
        
        # The idea is to use a dictionary to store all letters and number of count they appears in both
        #   string s and string t. Then compare if all the counts are the same
        # Since strings only contain lowercase alphabets, we can use a list of length 26 as the hashmap
        #   to lower the space usage to O(1)
        # We will use a smart way to +1 for occurance in s, and -1 for occurance in t
        
        if len(s) != len(t):
            return False
        counts = [0] * 26
        for i in range(len(s)):
            counts[ord(s[i]) - ord('a')] += 1
            counts[ord(t[i]) - ord('a')] -= 1
            
        # Then check if all count is 0
        for c in counts:
            if c != 0:
                return False
        return True


'''
409. Longest Palindrome
Easy

Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.

This is case sensitive, for example "Aa" is not considered a palindrome here.

Note:
Assume the length of given string will not exceed 1,010.

Example:

Input:
"abccccdd"

Output:
7

Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.
'''

class Solution2:
    
    # Time: O(N)  Space: O(1) as the alphabetic size is fixed
    def longestPalindrome(self, s: str) -> int:
        # First create a dictionary to store each character and its count
        # I'll use collections.Counter() to create such dict fast
        from collections import Counter
        counts = Counter(s)
        result = 0
        
        # increase result if any count is even number
        for val, count in counts.items():
            result += count // 2 * 2
            # we can contain 1 item that count is odd as we can put it in the center
            if result % 2 == 0 and count % 2 == 1:
                result += 1
        
        return result


'''
205. Isomorphic Strings
Easy

Given two strings s and t, determine if they are isomorphic.

Two strings are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.

Example 1:

Input: s = "egg", t = "add"
Output: true
Example 2:

Input: s = "foo", t = "bar"
Output: false
Example 3:

Input: s = "paper", t = "title"
Output: true
Note:
You may assume both s and t have the same length.
'''

class Solution3:
    # loop through s and t once, and check the current occurence of each character
    # Return false as long as they don't match
    
    # Time: O(N)  Space: O(1)
    def isIsomorphic(self, s: str, t: str) -> bool:
        countS = {}
        countT = {}
        for i in range(len(s)):
            si = s[i]
            ti = t[i]
            if countS.get(si) != countT.get(ti):
                return False
            # put i in the dict so that we note down the last occurence of one character. In this way, 
            #   the next time we see the same character we can compare that if they always occur at the
            #   same location
            countS[si] = i
            countT[ti] = i
        return True


'''
647. Palindromic Substrings
Medium

Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example 1:

Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
 

Example 2:

Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
 

Note:

The input string length won't exceed 1000.
'''

class Solution4:
    def countSubstrings(self, s: str) -> int:
        # With this method, we will expand from each 1, or 2 char in the string
        
        # Time: O(N^2)  Space: O(1)
        
        # There is an O(N) solution using 'Manacher's Algorithm', 
        #   but it is beyond what I need for interview as it is very complex
        ln = len(s)
        result = 0
        
        for i in range(ln):
            # expand from 1 char
            l = r = i
            while l >= 0 and r < ln and s[l] == s[r]:
                result += 1
                l -= 1
                r += 1
                
            # expand from 2 char
            l = i
            r = i + 1
            while l >= 0 and r < ln and s[l] == s[r]:
                result += 1
                l -= 1
                r += 1
                
        return result


'''
9. Palindrome Number
Easy

Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example 1:

Input: 121
Output: true
Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
Follow up:

Coud you solve it without converting the integer to a string?
'''

class Solution5:
    # Time: O(log10 X)  Space: O(1)
    def isPalindrome(self, x: int) -> bool:
        # The easiest way is to convert the integer to a string, then use two pointers to solve it
        
        # For the follow up question, I'll solve it without converting to string
        # We are going to reverse the second half of the integer and compare it to the first half of it.
        # We don't want to invert the whole integer as it could possibly exceed the max integer value
        # For ex, for int 1221, we revert the second half to become '12' and compare it to the first
        #   half, which is '12', and we return True
        # We know that we've reached half when reverted number is >= x
        
        # if x < 0, it has to be False; if x ends with digit 0, it has to be false if x != 0
        if (x < 0 or (x % 10 == 0 and x != 0)):
            return False
        
        reverted = 0
        while reverted < x:
            reverted = reverted * 10 + x % 10
            x //= 10
        
        # for odd length of digits, we can use reverted // 10 to get rid of the extra digit
        return x == reverted or x == reverted // 10


'''
696. Count Binary Substrings
Easy

Give a string s, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

Example 1:
Input: "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".

Notice that some of these substrings repeat and are counted the number of times they occur.

Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
Example 2:
Input: "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
Note:

s.length will be between 1 and 50,000.
s will only consist of "0" or "1" characters.
'''

class Solution6:
    # Time: O(N)  Space: O(1)
    def countBinarySubstrings(self, s: str) -> int:
        # Linear Scan
        # We can group the string into group of 0's and 1's, and for each 2 connected group, the number
        #   of desired substrings that can be formed is the min of the two sizes
        #   Ex: 00011 is group of size 3 and size 2, and they can form min(3, 2) = 2 substrings
        if len(s) < 2:
            return 0
        
        # prev is the len of previous group, curr is the len of curr group
        prev, curr, result = 0, 1, 0
        for i in range(1, len(s)):
            # if s[i] != s[i-1], it means we've ended a group
            if s[i] != s[i - 1]:
                result += min(prev, curr)
                prev, curr = curr, 1
            else:
                curr += 1
        
        # Need to add the last min of prev and curr as it is not covered in the loop
        return result + min(prev, curr)