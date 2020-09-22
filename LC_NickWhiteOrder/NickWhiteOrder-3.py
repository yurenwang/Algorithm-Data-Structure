'''
Leetcode problems following the order of Nick White's playlist - III

771. Jewels and Stones
700. Search in a Binary Search Tree
852. Peak Index in a Mountain Array
917. Reverse Only Letters
59. Spiral Matrix II
929. Unique Email Addresses
804. Unique Morse Code Words
595. Big Countries
905. Sort Array By Parity
728. Self Dividing Numbers

'''

'''
771. Jewels and Stones
Easy

You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

Example 1:

Input: J = "aA", S = "aAAbbbb"
Output: 3
Example 2:

Input: J = "z", S = "ZZ"
Output: 0
Note:

S and J will consist of letters and have length at most 50.
The characters in J are distinct.

'''

class Solution1:
    def numJewelsInStones(self, J: str, S: str) -> int:
        # easy question, use a set to contain J and loop through S to get the count
        # Time: O(N) where N is length of S. Length of J is ignored as it is at most 52
        # Space: O(1) since there are at most 2*26 different kinds of jewels
        jewels = set([c for c in J])
        count = 0
        for c in S:
            if c in jewels:
                count += 1
        return count


'''
700. Search in a Binary Search Tree
Easy

Given the root node of a binary search tree (BST) and a value. You need to find the node in the BST that the node's value equals the given value. Return the subtree rooted with that node. If such node doesn't exist, you should return NULL.

For example, 

Given the tree:
        4
       / \
      2   7
     / \
    1   3

And the value to search: 2
You should return this subtree:

      2     
     / \   
    1   3
In the example above, if we want to search the value 5, since there is no node with value 5, we should return NULL.

Note that an empty tree is represented by NULL, therefore you would see the expected output (serialized tree format) as [], not null.

'''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution2:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        # recursion is easy but space consuming so I'll write the iteration solution
        # Use a pointer
        # Time: O(N)  Space: O(1)
        p = root
        while p:
            if p.val == val:
                return p
            p = p.left if p.val > val else p.right
        return None


'''
852. Peak Index in a Mountain Array
Easy

Let's call an array arr a mountain if the following properties hold:

arr.length >= 3
There exists some i with 0 < i < arr.length - 1 such that:
arr[0] < arr[1] < ... arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]
Given an integer array arr that is guaranteed to be a mountain, return any i such that arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1].

 
Example 1:

Input: arr = [0,1,0]
Output: 1

Example 2:

Input: arr = [0,2,1,0]
Output: 1

Example 3:

Input: arr = [0,10,5,2]
Output: 1

Example 4:

Input: arr = [3,4,5,1]
Output: 2

Example 5:

Input: arr = [24,69,100,99,79,78,67,36,26,19]
Output: 2
 

Constraints:

3 <= arr.length <= 104
0 <= arr[i] <= 106
arr is guaranteed to be a mountain array.

'''

class Solution3:
    def peakIndexInMountainArray(self, arr: [int]) -> int:
        # Easy: Linear scan
        # Time: O(N)  Space: O(1)
        for i in range(1, len(arr)):
            if arr[i] < arr[i - 1]:
                return i - 1
            
        # Faster: Binary Search
        # Time: O(logN)  Space: O(1)
        l = 0
        r = len(arr) - 1
        while l < r:
            m = (l + r) / 2
            if arr[m] < arr[m + 1]:
                l = m
            else:
                r = m + 1
        return m + 1


'''
917. Reverse Only Letters
Easy

Given a string S, return the "reversed" string where all characters that are not a letter stay in the same place, and all letters reverse their positions.


Example 1:

Input: "ab-cd"
Output: "dc-ba"

Example 2:

Input: "a-bC-dEf-ghIj"
Output: "j-Ih-gfE-dCba"

Example 3:

Input: "Test1ng-Leet=code-Q!"
Output: "Qedo1ct-eeLg=ntse-T!"
 

Note:

S.length <= 100
33 <= S[i].ASCIIcode <= 122 
S doesn't contain \ or "

'''

class Solution4:
    def reverseOnlyLetters(self, S: str) -> str:
        # Use two pointers, one from left and one from right
        # Time: O(N)  Space: O(N)
        # We could also use a stack
        code_list = list(S)
        l, r = 0, len(S) - 1
        while l < r:
            if not code_list[l].isalpha():
                l += 1
                continue
            if not code_list[r].isalpha():
                r -= 1
                continue
            code_list[l], code_list[r] = code_list[r], code_list[l]
            l += 1
            r -= 1
        return ''.join(code_list)
            

'''
59. Spiral Matrix II
Medium

Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

Example:

Input: 3
Output:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]

'''

class Solution5:
    def generateMatrix(self, n: int) -> [[int]]:
        # Iteration in a spiral way
        # Time: O(N*N)  Space: O(1)
        
        result = [[0] * n for _ in range(n)] 
        count = 0
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # 4 directions
        dir_count = 0
        cur_dir = 0
        i = 0
        j = -1
        
        while count < n * n:
            new_i = i + dirs[cur_dir][0]    # next cell
            new_j = j + dirs[cur_dir][1]
            # check if the next cell is out of range or is already updated
            if new_i < 0 or new_i >= n or new_j < 0 or new_j >= n or result[new_i][new_j] != 0:
                dir_count += 1              # change direction if yes
                cur_dir = dir_count % 4     # update the direction
                new_i = i + dirs[cur_dir][0]# get the new, proper next cell based on the new direction
                new_j = j + dirs[cur_dir][1]
            i, j = new_i, new_j             # update i and j
            count += 1                      # update count
            result[new_i][new_j] = count    # update result
            
        return result


'''
929. Unique Email Addresses
Easy

Every email consists of a local name and a domain name, separated by the @ sign.

For example, in alice@leetcode.com, alice is the local name, and leetcode.com is the domain name.

Besides lowercase letters, these emails may contain '.'s or '+'s.

If you add periods ('.') between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name.  For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.  (Note that this rule does not apply for domain names.)

If you add a plus ('+') in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered, for example m.y+name@email.com will be forwarded to my@email.com.  (Again, this rule does not apply for domain names.)

It is possible to use both of these rules at the same time.

Given a list of emails, we send one email to each address in the list.  How many different addresses actually receive mails? 

 
Example 1:

Input: ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails
 

Note:

1 <= emails[i].length <= 100
1 <= emails.length <= 100
Each emails[i] contains exactly one '@' character.
All local and domain names are non-empty.
Local names do not start with a '+' character.

'''

class Solution6:
    # Time: O(N)  Space: O(N)
    # use a set to check for the existance
    def numUniqueEmails(self, emails: [str]) -> int:
        email_set = set()
        for e in emails:
            e_split = e.split('@')
            tmp = []
            for c in e_split[0]:
                if c == '.':
                    continue
                elif c == '+':
                    break
                else:
                    tmp.append(c)
            tmp.append('@')
            tmp.append(e_split[1])
            new_email = ''.join(tmp)
            if new_email not in email_set:
                email_set.add(new_email)
        return len(email_set)


'''
804. Unique Morse Code Words
Easy

International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on.

For convenience, the full table for the 26 letters of the English alphabet is given below:

[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. For example, "cab" can be written as "-.-..--...", (which is the concatenation "-.-." + ".-" + "-..."). We'll call such a concatenation, the transformation of a word.

Return the number of different transformations among all words we have.

Example:
Input: words = ["gin", "zen", "gig", "msg"]
Output: 2
Explanation: 
The transformation of each word is:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

There are 2 different transformations, "--...-." and "--...--.".
Note:

The length of words will be at most 100.
Each words[i] will have length in range [1, 12].
words[i] will only consist of lowercase letters.

'''

class Solution7:
    def uniqueMorseRepresentations(self, words: [str]) -> int:
        # Use a dictionary for the key, transformation pair, and use a set to check for existance
        # Time: O(N)  Space: O(N)
        keys = [c for c in 'abcdefghijklmnopqrstuvwxyz']
        values = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        key_dict = dict(zip(keys, values))
        trans_set = set()
        
        for w in words:
            trans_list = []
            for c in w:
                trans_list.append(key_dict[c])
            trans = ''.join(trans_list)
            trans_set.add(trans)
        
        return len(trans_set)


'''
595. Big Countries
Easy

SQL Schema
There is a table World

+-----------------+------------+------------+--------------+---------------+
| name            | continent  | area       | population   | gdp           |
+-----------------+------------+------------+--------------+---------------+
| Afghanistan     | Asia       | 652230     | 25500100     | 20343000      |
| Albania         | Europe     | 28748      | 2831741      | 12960000      |
| Algeria         | Africa     | 2381741    | 37100000     | 188681000     |
| Andorra         | Europe     | 468        | 78115        | 3712000       |
| Angola          | Africa     | 1246700    | 20609294     | 100990000     |
+-----------------+------------+------------+--------------+---------------+
A country is big if it has an area of bigger than 3 million square km or a population of more than 25 million.

Write a SQL solution to output big countries' name, population and area.

For example, according to the above table, we should output:

+--------------+-------------+--------------+
| name         | population  | area         |
+--------------+-------------+--------------+
| Afghanistan  | 25500100    | 652230       |
| Algeria      | 37100000    | 2381741      |
+--------------+-------------+--------------+

'''

# Write your MySQL query statement below   # Solution8
select w.name, w.population, w.area from World w
where w.population > 25000000 or w.area > 3000000


'''
905. Sort Array By Parity
Easy

Given an array A of non-negative integers, return an array consisting of all the even elements of A, followed by all the odd elements of A.

You may return any answer array that satisfies this condition.

 
Example 1:

Input: [3,1,2,4]
Output: [2,4,3,1]
The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.
 

Note:

1 <= A.length <= 5000
0 <= A[i] <= 5000

'''

class Solution9:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        # Use two pointers and swap in place
        # Time: O(N)  Space: O(1)
        l, r = 0, len(A) - 1
        while l < r:
            if A[l] % 2 == 1 and A[r] % 2 == 0:
                A[l], A[r] = A[r], A[l]
                l += 1
                r -= 1
            # If both are odd, then move right to left 1 step becase right one don't need to be moved
            elif A[l] % 2 == 1 and A[r] % 2 == 1:   
                r -= 1
            elif A[l] % 2 == 0 and A[r] % 2 == 0:
                l += 1
            else:
                l += 1
                r -= 1
        return A


'''
728. Self Dividing Numbers
Easy

A self-dividing number is a number that is divisible by every digit it contains.

For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

Also, a self-dividing number is not allowed to contain the digit zero.

Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.

Example 1:
Input: 
left = 1, right = 22
Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
Note:

The boundaries of each input argument are 1 <= left <= right <= 10000.

'''

class Solution10:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:  # One line solution. Time: O(N)
        return [n for n in range(left, right + 1) if all([int(c) != 0 and n % int(c) == 0 for c in (str(n))])]