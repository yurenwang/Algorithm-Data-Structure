'''
Leetcode Frequency Ranked Two

706. Design HashMap
394. Decode String
199. Binary Tree Right Side View
412. Fizz Buzz
609. Find Duplicate File in System
176. Second Highest Salary
362. Design Hit Counter
811. Subdomain Visit Count
986. Interval List Intersections
438. Find All Anagrams in a String

'''

'''
706. Design HashMap
Easy

Design a HashMap without using any built-in hash table libraries.

To be specific, your design should include these functions:

put(key, value) : Insert a (key, value) pair into the HashMap. If the value already exists in the HashMap, update the value.
get(key): Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
remove(key) : Remove the mapping for the value key if this map contains the mapping for the key.

Example:

MyHashMap hashMap = new MyHashMap();
hashMap.put(1, 1);          
hashMap.put(2, 2);         
hashMap.get(1);            // returns 1
hashMap.get(3);            // returns -1 (not found)
hashMap.put(2, 1);          // update the existing value
hashMap.get(2);            // returns 1 
hashMap.remove(2);          // remove the mapping for 2
hashMap.get(2);            // returns -1 (not found) 

Note:

All keys and values will be in the range of [0, 1000000].
The number of operations will be in the range of [1, 10000].
Please do not use the built-in HashMap library.
'''

# Solution 1
class Bucket:
    # This class represents a single bucket (storage cell) in our dictionary list. 
    # More comments are below in the main class. 
    def __init__(self):
        # items stores (key, value) pairs
        self.items = []
    
    
    # Insert or update a (key, value) in the curr bucket
    def put(self, key: int, value: int) -> None:
        found = False
        for i, keyVal in enumerate(self.items):
            if keyVal[0] == key:
                self.items[i] = (key, value)
                found = True
                break
        if not found:
            self.items.append((key, value))
        

    def get(self, key: int) -> int:         
        for k, v in self.items:
            if k == key:
                return v
        return -1
        

    def remove(self, key: int) -> None:
        for i, keyVal in enumerate(self.items):
            if keyVal[0] == key:
                self.items.pop(i)
                break
            

class MyHashMap:
    # We create a simple hash method that takes in an integer as a key, and returns an integer
    #   which is also the storage index of the (key, value) pair.
    # Use a list to represent the total storage space. 
    # For each item in the list, we store another list, because there might be multiple items
    #   using the same storage space when item numbers grow higher.
    # In that inner list, we store tuples of (key, val) pairs so that we can search and find 
    #   values based on keys when user wants. 

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # Let's use a big prime number as the total storage size, and use module (%) for the 
        #   hash method. Using a prime number is recommended, as it minimize the possibility
        #   of collisions. We use 2069 for this question.
        self.space = 2069
        self.myDict = [Bucket() for _ in range(self.space)]
        

    # O(1)
    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        self.myDict[key % self.space].put(key, value)
        

    # O(1)
    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        return self.myDict[key % self.space].get(key)
        

    # O(1)
    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        return self.myDict[key % self.space].remove(key)


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)


'''
394. Decode String
Medium

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

 

Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
Example 2:

Input: s = "3[a2[c]]"
Output: "accaccacc"
Example 3:

Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
Example 4:

Input: s = "abc3[cd]xyz"
Output: "abccdcdcdxyz"
'''

class Solution2:
    # Time: O(N)    Space: O(N)
    def decodeString(self, s: str) -> str:
        # Use two stacks, one to store the number (multiplier), one to store the string when
        #   there are multiple layers of brackets
        s_num = []
        s_str = []
        curr_str = ''
        i = 0
        
        while i < len(s):
            print(s_num, s_str)
            c = s[i]
            
            if c.isdigit():     # Calculate the multiplier
                curr_num = 0
                while s[i].isdigit():
                    curr_num = curr_num * 10 + int(s[i])
                    i += 1
                s_num.append(curr_num)
                i -= 1  # Need to -1 here otherwise 1 will be added twice
                
            elif c == '[':
                # Getting the open bracket means that we are entering a deeper level. In this
                #   case, we need to first put whatever we have (on the outer level) to the
                #   stack, so that in the future when we pop out from the stack, we can have
                #   it ready
                s_str.append(curr_str)
                curr_str = ''
                
            elif c == ']':
                # Pop the first item of both string and number, and multiply the result
                # Then attach the string from the outer level to the front of it, to get the
                #   final updated string
                multiplier = s_num.pop()
                str_before = s_str.pop()
                curr_str = ''.join([str_before] + [curr_str] * multiplier)
                
            else:
                # If we encounter a char, we simply add it to the current string, to get ready
                #   for appending it after the stack
                curr_str += c
                
            i += 1
        
        return curr_str
        

'''
199. Binary Tree Right Side View
Medium

Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example:

Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]
Explanation:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
 
'''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Time: O(N)  Space: O(W), W is the max width of the tree
class Solution3:
    def rightSideView(self, root: TreeNode) -> [int]:
        # This is a BFS/DFS question
        
        # We want a BFS to traverse through the whole tree, layer by layer. For each layer, 
        #   we track the width of the tree so that we can know which node is the right most
        #   item on that layer. Knowing all items on the right most of each layer, we can 
        #   know the right side view.
        
        if not root:
            return []
        
        # I couldn't use list for the queue as we have to pop from left for this question 
        #   to calculate the width accurately
        from collections import deque
        queue = deque()
        queue.append(root)
        result = []
        
        while queue:
            width = len(queue)
            for i in range(width):   # iterate the whole row every time
                curr = queue.popleft()
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
                if i == width - 1:
                    result.append(curr.val)     # right most, add to result
        
        return result
        

'''
412. Fizz Buzz
Easy

Write a program that outputs the string representation of numbers from 1 to n.

But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”. For numbers which are multiples of both three and five output “FizzBuzz”.

Example:

n = 15,

Return:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]
'''

class Solution4:
    # Time: O(N)  Space: O(1)
    def fizzBuzz(self, n: int) -> [str]:
        # The naive solution would be to use if statements to check if n % 15 == 0 first, then
        #   check for 3 and 5
        # However, if in the future we have more keywords to check, we would have too many 
        #   if conditions, which doesn't look good.
        
        # Two proper solutions would be to: 
        #   1. Use string concatenation, check 3 and 5 first, and concatenate the result, then
        #       check if result == '', if yes, append the number in the behind
        #   2. Use hashmap to store all key value pairs, ie: {3: 'Fizz', 5: 'Buzz'}, then use
        #       it to find 3 and 5, then check if result == '' and append number if needed
        
        # I'll use hashmap(dictionary), as it is the most scaleable
        d = {3: 'Fizz', 5: 'Buzz'}
        result = []
        
        for i in range(1, n + 1):
            curr = ''
            for key in d.keys():
                if i % key == 0:
                    curr += d[key]
            if curr == '':
                curr += str(i)
            
            result.append(curr)
        
        return result
        

'''
609. Find Duplicate File in System
Medium

Given a list of directory info including directory path, and all the files with contents in this directory, you need to find out all the groups of duplicate files in the file system in terms of their paths.

A group of duplicate files consists of at least two files that have exactly the same content.

A single directory info string in the input list has the following format:

"root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"

It means there are n files (f1.txt, f2.txt ... fn.txt with content f1_content, f2_content ... fn_content, respectively) in directory root/d1/d2/.../dm. Note that n >= 1 and m >= 0. If m = 0, it means the directory is just the root directory.

The output is a list of group of duplicate file paths. For each group, it contains all the file paths of the files that have the same content. A file path is a string that has the following format:

"directory_path/file_name.txt"

Example 1:

Input:
["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
Output:  
[["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]
 

Note:

No order is required for the final output.
You may assume the directory name, file name and file content only has letters and digits, and the length of file content is in the range of [1,50].
The number of files given is in the range of [1,20000].
You may assume no files or directories share the same name in the same directory.
You may assume each given directory info represents a unique directory. Directory path and file info are separated by a single blank space.
 

Follow-up beyond contest:
Imagine you are given a real file system, how will you search files? DFS or BFS?
If the file content is very large (GB level), how will you modify your solution?
If you can only read the file by 1kb each time, how will you modify your solution?
What is the time complexity of your modified solution? What is the most time-consuming part and memory consuming part of it? How to optimize?
How to make sure the duplicated files you find are not false positive?
'''

class Solution5:
	# This question is not hard. It's just long to read

    # Time: O(N)    Space: O(N), where N is n * l (n strings of average length of l)
    def findDuplicate(self, paths: [str]) -> [[str]]:
        # Use a hashmap(dictionary) to store files as keys, and its path as value
        # Use defaultdict so we don't get error when accessing a key that's not in it yet
        d = collections.defaultdict(lambda : [])
        for p in paths:
            p_input = p.split(' ')
            root = p_input[0]
            
            for file in p_input[1:]:
                name, _, content = file.partition('(')
                d[content[:-1]].append(root + '/' + name)
            
        return [x for x in d.values() if len(x) > 1]


'''
176. Second Highest Salary
Easy

SQL Schema
Write a SQL query to get the second highest salary from the Employee table.

+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
For example, given the above Employee table, the query should return 200 as the second highest salary. If there is no second highest salary, then the query should return null.

+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
'''

############# This is a SQL question so the below code is in SQL. Uncomment the code to run #############

# Solution6
# /* Write your T-SQL query statement below */
# SELECT MAX(Salary) AS SecondHighestSalary
# FROM Employee
# WHERE Salary NOT IN (
#     SELECT MAX(Salary) FROM Employee)


'''
362. Design Hit Counter
Medium

Design a hit counter which counts the number of hits received in the past 5 minutes.

Each function accepts a timestamp parameter (in seconds granularity) and you may assume that calls are being made to the system in chronological order (ie, the timestamp is monotonically increasing). You may assume that the earliest timestamp starts at 1.

It is possible that several hits arrive roughly at the same time.

Example:

HitCounter counter = new HitCounter();

// hit at timestamp 1.
counter.hit(1);

// hit at timestamp 2.
counter.hit(2);

// hit at timestamp 3.
counter.hit(3);

// get hits at timestamp 4, should return 3.
counter.getHits(4);

// hit at timestamp 300.
counter.hit(300);

// get hits at timestamp 300, should return 4.
counter.getHits(300);

// get hits at timestamp 301, should return 3.
counter.getHits(301); 
Follow up:
What if the number of hits per second could be very large? Does your design scale?
'''

# Solution 7
class HitCounter:
    # Use a list of length 300 to store all the counters, since we only need the hits 
    #   received in the past 5 minutes
    # Use another list of lenth 300 to store the corresponding timestamp, since we need those
    #   timestamp to determine if the new hit count will add onto the old one, or it will 
    #   replace the old one. 
    # Each number of hits at time i is stored at index i - 1
    
    # For the follow up, I think this solution is NOT very scalable, since we are recording 
    #   all the timestamps. 
    
    # Time: O(1) for hit, O(N) for getHit
    # Space: O(1) since our interval is fixed size, which in this case is 300
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.SIZE = 300
        self.counts = [0] * self.SIZE
        self.times = [0] * self.SIZE
        

    def hit(self, timestamp: int) -> None:
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        index = timestamp % self.SIZE - 1
        if self.times[index] != timestamp:
            self.counts[index] = 1
            self.times[index] = timestamp
        else:
            self.counts[index] += 1
        

    def getHits(self, timestamp: int) -> int:
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        result = 0
        for i in range(self.SIZE):
            if timestamp - self.times[i] < 300:
                result += self.counts[i]
        return result


# Your HitCounter object will be instantiated and called as such:
# obj = HitCounter()
# obj.hit(timestamp)
# param_2 = obj.getHits(timestamp)


'''
811. Subdomain Visit Count
Easy

A website domain like "discuss.leetcode.com" consists of various subdomains. At the top level, we have "com", at the next level, we have "leetcode.com", and at the lowest level, "discuss.leetcode.com". When we visit a domain like "discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.

Now, call a "count-paired domain" to be a count (representing the number of visits this domain received), followed by a space, followed by the address. An example of a count-paired domain might be "9001 discuss.leetcode.com".

We are given a list cpdomains of count-paired domains. We would like a list of count-paired domains, (in the same format as the input, and in any order), that explicitly counts the number of visits to each subdomain.

Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.

Example 2:
Input: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
Output: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
Explanation: 
We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 5 times. For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.

Notes:

The length of cpdomains will not exceed 100. 
The length of each domain name will not exceed 100.
Each address will have either 1 or 2 "." characters.
The input count in any count-paired domain will not exceed 10000.
The answer output can be returned in any order.
'''

class Solution8:
    # Easy question. Use hashmap to store the domains and their counts
    
    # Time: O(N)  Space: O(N)
    def subdomainVisits(self, cpdomains: [str]) -> [str]:
        # Use Hashmap
        # Partition input first to get the count and url
        countDict = collections.defaultdict(lambda : 0)
        
        for s in cpdomains:
            countStr, _, url = s.partition(' ')
            count = int(countStr)
            domains = url.split('.')
            
            for i in range(len(domains)):
                currDomain = '.'.join(domains[i:])
                
                countDict[currDomain] += count
            
        result = []
        for i in countDict:
            result.append(str(countDict[i]) + ' ' + i)
        
        return result


'''
986. Interval List Intersections
Medium

Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

(Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  The intersection of two closed intervals is a set of real numbers that is either empty, or can be represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

 

Example 1:
Go to https://leetcode.com/problems/interval-list-intersections/ to view the example image


Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
 

Note:

0 <= A.length < 1000
0 <= B.length < 1000
0 <= A[i].start, A[i].end, B[i].start, B[i].end < 10^9
'''

class Solution9:
    # Time: O(M + N)  Space: O(M + N) if we count the result list's size
    def intervalIntersection(self, A: [[int]], B: [[int]]) -> [[int]]:
        # Two pointers
        # Put pointer i at the first item in A, and j at the first item in B, check if two
        #   items intersect. Compare the end of interval with the previous ending for the 
        #   intervals, then put the intervals into the result list
        
        if not A or not B:
            return []
        
        i = j = 0
        result = []
        
        while i < len(A) and j < len(B):
            # Check if there's a intersection. Intersection begins at start and ends at end, 
            #   if start <= end
            start = max(A[i][0], B[j][0])
            end = min(A[i][1], B[j][1])
            
            if start <= end:
                result.append([start, end])
            
            # Then, move pointer forward for the earlier ended interval
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        
        return result
        

'''
438. Find All Anagrams in a String
Medium

Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

Example 1:

Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
Example 2:

Input:
s: "abab" p: "ab"

Output:
[0, 1, 2]

Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
'''
        
class Solution10:
    # Time: O(M + N), since we have a one pass solution on two strings of length M and N
    # Space: O(1), since our list length is fixed at 26
    def findAnagrams(self, s: str, p: str) -> [int]:
        ### Option 1:
        # Use a hashmap to count the chars appear in the current string of length l, where
        #   l is the length of p
        ### Option 2:
        # Use a list of length 26 to store the occurance of each character
        
        # Use a list:
        counts = [0] * 26
        target = [0] * 26
        ls = len(s)
        lp = len(p)
        if ls < lp:
            return []
        
        result = []
        
        # Count the target
        for c in p:
            target[ord(c) - ord('a')] += 1
        
        # Loop through string s using a Sliding Window
        for i in range(ls):
            # Add char at i first:
            counts[ord(s[i]) - ord('a')] += 1
        
            # Remove char at i - lp if necessary:
            if i >= lp:
                counts[ord(s[i - lp]) - ord('a')] -= 1
            
            # Compare to check if we get same amount of chars, if yes, we find a result
            if counts == target:
                result.append(i - lp + 1)
        
        return result
        
