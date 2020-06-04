'''
Leetcode Frequency Ranked - One

146. LRU Cache
994. Rotting Oranges
15. 3Sum
937. Reorder Data in Log Files
560. Subarray Sum Equals K
773. Sliding Puzzle
973. K Closest Points to Origin
1249. Minimum Remove to Make Valid Parentheses
692. Top K Frequent Words
253. Meeting Rooms II
953. Verifying an Alien Dictionary
380. Insert Delete GetRandom O(1)
224. Basic Calculator
981. Time Based Key-Value Store
221. Maximal Square

'''

'''
146. LRU Cache
Medium

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
'''

# To solve this problem, we can use a double linked list to record the order and to perform O(1) of 
#   get() and put(). We also need a dictionary to store the key value and node value, so that we can
#   look up any node in O(1)

# Another solution is simply to use an ordered Dictionary. However, that is a bonus in the interview but
#   this solution should be what the interviewer is looking for

class Node:
    def __init__(self, key = 0, val = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        
# Time: O(1)  Space: O(capacity)
class LRUCache:
    def __init__(self, capacity: int):
        # We want two dummy nodes, head and tail, so that we don't have to worry about None
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.nodeMap = {}
        self.size = 0
        self.capacity = capacity
    
    # Helper to add to the beginning of linked list
    def addHead(self, node):
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        node.prev = self.head
    
    # Helper to remove the last item from linked list
    def popTail(self):
        self.tail.prev = self.tail.prev.prev
        self.tail.prev.next = self.tail
    
    # Helper to remove a specific item from linked list
    def popNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev 

    # O(1)
    def get(self, key: int) -> int:
        # We want to get the node value from the map, as well as putting the node in front of the
        #   linked list so that it is the most recently visited
        # To move the node, we first remove it, then add it back, which add it to the front
        node = self.nodeMap.get(key, 0)
        if node:
            self.popNode(node)
            self.addHead(node)
            return node.val
        else:
            return -1
       
    # O(1)     
    def put(self, key: int, value: int) -> None:
        # Always insert to the front of the linkedlist so it is the most recently viewed item
        # If node exist for key, remove the node then add it back in so that it will appear in the front
        curr = self.nodeMap.get(key, 0)
        if curr:
            self.popNode(curr)
            self.size -= 1
        elif self.size == self.capacity:
            tailKey = self.tail.prev.key
            self.nodeMap.pop(tailKey)
            self.popTail()
            self.size -= 1
        node = Node(key, value)
        self.addHead(node)
        self.nodeMap[key] = node
        self.size += 1
        
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


'''
994. Rotting Oranges
Medium

In a given grid, each cell can have one of three values:

the value 0 representing an empty cell;
the value 1 representing a fresh orange;
the value 2 representing a rotten orange.
Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange.  If this is impossible, return -1 instead.

Example 1:

********** Visit https://leetcode.com/problems/rotting-oranges/ 
********** For the image


Input: [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
Example 2:

Input: [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation:  The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
Example 3:

Input: [[0,2]]
Output: 0
Explanation:  Since there are already no fresh oranges at minute 0, the answer is just 0.
 

Note:

1 <= grid.length <= 10
1 <= grid[0].length <= 10
grid[i][j] is only 0, 1, or 2.
'''

class Solution2:
    # Time: O(N)  Space: O(N)  where N is the size of the grid
    def orangesRotting(self, grid: [[int]]) -> int:
        # This is a BFS question
        # queue will contain (i, j, t), where i and j are the index of an orange, t is the time spent
        #   until this orange becomes rotten
        queue = []
        freshOranges = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]     # 4 directions
        h = len(grid)
        if not h:
            return -1
        w = len(grid[0])
        result = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                curr = grid[i][j]
                if curr == 2:
                    queue.append((i, j, 0))     # initialize the queue to initial rotten oranges
                elif curr == 1:
                    freshOranges += 1   
        
        for curr in queue:
            for d in directions:
                nextI = curr[0] + d[0]
                nextJ = curr[1] + d[1]
                if 0 <= nextI < h and 0 <= nextJ < w and grid[nextI][nextJ] == 1:   # found a fresh one
                    nextT = curr[2] + 1
                    queue.append((nextI, nextJ, nextT))     # insert to queue to continue next BFS layer
                    result = nextT       # update result with the time spent
                    freshOranges -= 1
                    grid[nextI][nextJ] = 2      # mark as rotten so we don't visit again
                    
        return result if freshOranges == 0 else -1


'''
15. 3Sum
Medium

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.

Example:

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''

class Solution3:
    # Time: O(N*N)  Space: O(logN) for the sorting algorithm
    def threeSum(self, nums: [int]) -> [[int]]:
        # First sort the list
        # Then put each item i into a set
        # Then use two pointers on the rest of the list to see if there are two that sum up to -i
        
        nums.sort()
        
        result = []
        l = len(nums)
        
        for i in range(l):
            # Break out of the loop if curr is > 0 as nums after it won't be able to sum to 0
            curr = nums[i]
            if curr > 0:
                break
            # Skip if curr is same as the previous num
            if i > 0 and curr == nums[i - 1]:
                continue
                
            # Two pointers solution
            p1 = i + 1
            p2 = l - 1
            while p1 < p2:
                v1 = nums[p1]
                v2 = nums[p2]
                
                # Increase p1 if v1 == previous num that is not nums[i], or if sum is less than 0
                sum = curr + v1 + v2
                if (p1 > i + 1 and v1 == nums[p1 - 1]) or sum < 0:
                    p1 += 1
                # Decrease p2
                elif (p2 < l - 1 and v2 == nums[p2 + 1]) or sum > 0:
                    p2 -= 1
                # Found a match
                else:
                    result.append([curr, v1, v2])
                    p1 += 1
                    p2 -= 1
           
        return result
                    

'''
937. Reorder Data in Log Files
Easy

You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

Each word after the identifier will consist only of lowercase letters, or;
Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.


Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
 

Constraints:

0 <= logs.length <= 100
3 <= logs[i].length <= 100
logs[i] is guaranteed to have an identifier, and a word after the identifier.
'''
            
class Solution4:
    # Time: O(nlogn)  Space: O(1)
    def reorderLogFiles(self, logs: [str]) -> [str]:
        # define a method to get a list of keys of a given string. Keys ordered based on importance.
        def findKey(str):
            # split by first space so we separate the identifier from values
            #   set the maximum split to 1 so we only split identifier and vals
            identifier, value = str.split(' ', 1)
            # if it's digit-log, return 1 as we want it to be put after letters, which is 0 as first key
            #   if it's letter-log, return 0, as well as less important keys, which are value, and id
            return (0, value, identifier) if value[0].isalpha() else (1, )
        
        logs.sort(key=findKey)
        return logs


'''
560. Subarray Sum Equals K
Medium

Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

Example 1:

Input:nums = [1,1,1], k = 2
Output: 2
 

Constraints:

The length of the array is in range [1, 20,000].
The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].
'''

class Solution5:
    # Time: O(N)  Space: O(N)
    def subarraySum(self, nums: [int], k: int) -> int:
        # O(N*N) solution is easy. Just iterate through the list for the starting index of the sublist,
        #   then keep adding until reachs or exceeds the target. Increase result by 1 if reaches, and
        #   skip the starting index if exceeds.
        
        # This solution is using Hashmap(Dictionary), which takes O(N) time and O(N) space. 
        #   1. We calculate the sum of all nums from the beginning of input list to index i as sumi. Then 
        #   we store the number of a sumi occurs in the dictionary (sumi, no. of occurance of sumi)
        #   2. While looping through the list to build the hashmap, we also check the number of occurance
        #   for (sumi - k) because if it exists, we know that we've find a sum == k for the sublist in 
        #   between two indexes. 
        #   3. We then update the result by num of occurance of (sumi - k)
        
        result = 0
        sumDict = {0: 1}
        currSum = 0
        
        for n in nums:
            currSum += n
            result += sumDict.get(currSum - k, 0)
            sumDict[currSum] = sumDict.get(currSum, 0) + 1
            
        return result


'''
773. Sliding Puzzle
Hard

On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0.

A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.

The state of the board is solved if and only if the board is [[1,2,3],[4,5,0]].

Given a puzzle board, return the least number of moves required so that the state of the board is solved. If it is impossible for the state of the board to be solved, return -1.

Examples:

Input: board = [[1,2,3],[4,0,5]]
Output: 1
Explanation: Swap the 0 and the 5 in one move.
Input: board = [[1,2,3],[5,4,0]]
Output: -1
Explanation: No number of moves will make the board solved.
Input: board = [[4,1,2],[5,0,3]]
Output: 5
Explanation: 5 is the smallest number of moves that solves the board.
An example path:
After move 0: [[4,1,2],[5,0,3]]
After move 1: [[4,1,2],[0,5,3]]
After move 2: [[0,1,2],[4,5,3]]
After move 3: [[1,0,2],[4,5,3]]
After move 4: [[1,2,0],[4,5,3]]
After move 5: [[1,2,3],[4,5,0]]
Input: board = [[3,2,4],[1,5,0]]
Output: 14
Note:

board will be a 2 x 3 array as described above.
board[i][j] will be a permutation of [0, 1, 2, 3, 4, 5].
'''

class Solution6:
#     Time Complexity: O(R * C * (R * C)!)O(R∗C∗(R∗C)!), where R, CR,C are the number of rows and columns in board. There are O((R * C)!)O((R∗C)!) possible board states.

# Space Complexity: O(R * C * (R * C)!)O(R∗C∗(R∗C)!).
    def slidingPuzzle(self, board: [[int]]) -> int:
        # BFS solution
        
        # turn the board into 1-D for easier comparision and modification
        boardOneD = [cell for row in board for cell in row]
        h = len(board)
        if h == 0:
            return -1
        w = len(board[0])
        
        # helper to swap elements and add to result list
        def swap(board, result, zero, diff):
            tmp = board[:]
            tmp[zero], tmp[zero + diff] = tmp[zero + diff], tmp[zero]
            
            result.append(tmp)
        
        # return all possible results after one slide
        def slideOnce(board):
            result = []
            zero = board.index(0)
            
            # can slide to right if it is not on the right most column
            if zero != 2 and zero != 5:
                swap(board, result, zero, 1)
            
            # can slide to left if it is not on the left most column
            if zero != 0 and zero != 3:
                swap(board, result, zero, -1)
                
            # can slide up if it is not on the first row
            if zero >= 3:
                swap(board, result, zero, -3)
                
            # can slide down if it is not on the last row
            if zero < 3:
                swap(board, result, zero, 3)
            
            return result
        
        # BFS queue
        queue = [(boardOneD, 0)]
        visited = set()
        visited.add(tuple(boardOneD))
        
        for b, count in queue:
            if b == [1,2,3,4,5,0]:  # reach target
                return count
            nextList = slideOnce(b)     # find all possible next
            for n in nextList:
                tn = tuple(n)
                if tn not in visited:   # only continue if next is not visited
                    queue.append((n, count + 1))
                    visited.add(tn)
        
        return -1


'''
973. K Closest Points to Origin
Medium

We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)

 

Example 1:

Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], K = 2
Output: [[3,3],[-2,4]]
(The answer [[-2,4],[3,3]] would also be accepted.)
 

Note:

1 <= K <= points.length <= 10000
-10000 < points[i][0] < 10000
-10000 < points[i][1] < 10000
'''

class Solution7:
    # Time: O(N) on average, O(N*N) worst case
    # Space: O(1)
    def kClosest(self, points: [[int]], K: int) -> [[int]]:
        # The naive solution would be to sort the list first, then pick the first K elements. This
        #   is very easy to implement, and will take O(NlogN) time.
        
        # However, We can benefit from the requirement, which does not require us to return the 
        #   output sorted, by doing a divide and conquer.
        # This is similar to doing a quick sort, but half way. We pick a pivot point p and put all
        #   items smaller than p to the left of it, and all larger ones to the right of it.
        # Then for the left bucket of size x, we compare x to K, if x > K, the Kth smallest item
        #   lands on the left bucket, then we recursively find the kClosest in the left bucket.
        #   And we can skip the right bucket. If x < K, vise versa, we calculate the (k-x)th closest
        #   in the right bucket. If x == K, we return the left bucket.
        # Since we only need to calculate 1 of the 2 buckets in each recursion, on average, the time
        #   usage is O(N) + O(N/2) + O(N/4) + ... == O(2N) == O(N)
        
        # In other word, we want to partially sort the list points, and only to make sure the first
        #   k nums are smaller than the last (n-k) ones, and we don't care about the order inside.
        
        # Helper: partition a list into two sections, one smaller or equal to the pivot, one larger
        #   return the index of the mid point (index of first item in second list)
        def partition(start, end):
            piv = sum(map(lambda x: x * x, points[end]))
            i = start       # i is the slow pointer that only move forward after a swap
            for j in range(start, end + 1):     # j is the fast pointer that always move forward
                # swap two pointers if the fast is <= pivot. We are sure that slow is > pivot because
                #   it would've been swapped already if it is not
                if sum(map(lambda x: x * x, points[j])) <= piv:
                    points[i], points[j] = points[j], points[i]
                    i += 1
            return i
        
        left = 0
        right = len(points) - 1
        
        while left <= right:
            mid = partition(left, right)
            lSize = mid - left
            if lSize > K:      # x > K, we continue with the left part
                right = mid - 2     # We can do mid - 2 because the num at mid - 1 will always be the 
                                        # pivot, because in our partition function, we swap if 
                                        # x <= piv, so we know that everything on it's left is <= the 
                                        # pivot.
            elif lSize < K:    # x < K, we continue with the right part
                K -= lSize
                left = mid
            else:              # x == K, so left has the right size
                return points[:mid]
        
        
'''
1249. Minimum Remove to Make Valid Parentheses
Medium

Given a string s of '(' , ')' and lowercase English characters. 

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
 

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
Example 2:

Input: s = "a)b(c)d"
Output: "ab(c)d"
Example 3:

Input: s = "))(("
Output: ""
Explanation: An empty string is also valid.
Example 4:

Input: s = "(a(b(c)d)"
Output: "a(b(c)d)"
 

Constraints:

1 <= s.length <= 10^5
s[i] is one of  '(' , ')' and lowercase English letters.
'''
        
class Solution8:
    # Time: O(N)  Space: O(N)
    
    # Two pass string builder solution
    # We can also use Stack to solve this problem
    def minRemoveToMakeValid(self, s: str) -> str:
        # We first loop through the string s to remove all closing brakets before opening ones
        numOfOpen = 0
        l1 = []
        for c in s:
            if c == '(':
                numOfOpen += 1
            elif c == ')':
                if not numOfOpen: continue
                numOfOpen -= 1
            l1.append(c)
        
        # Then loop through the result again, going backward, to remove additional opening brakets
        #   We want to loop backward because otherwise we might delete opening ones at the left first
        #   and results in closing ones before opening ones, ex: '()(' will results in ')('
        l1.reverse()
        l2 = []
        for c in l1:
            if c == '(' and numOfOpen:
                numOfOpen -= 1
                continue
            l2.append(c)
        
        # We reversed l1 to loop backward, so we need to reverse it back
        l2.reverse()
        return ''.join(l2)


'''
692. Top K Frequent Words
Medium

Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
Example 2:
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
Note:
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Input words contain only lowercase letters.
Follow up:
Try to solve it in O(n log k) time and O(n) extra space.
'''

class Solution9:
    # (Max) Heap
	# We can also use sorting, which takes a bit extra time (O(NlogN))
    
    # Time: O(k log N)  Space: O(N)
    def topKFrequent(self, words: [str], k: int) -> [str]:
        # Use a dictionary to count the occurance of each word, then use a maxHeap of size k to
        #   find out the most frequent k words, by defining a customized comparing function
        
        import heapq
        from collections import Counter 
        
        counts = Counter(words)
        # use -count here so we arrange the largest count first in our heap (because negative of it
        #   will then be the smallest)
        heap = [(-count, string) for string, count in counts.items()]
        heapq.heapify(heap)
        
        return [heapq.heappop(heap)[1] for _ in range(k)]


'''
253. Meeting Rooms II
Medium

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

Example 1:

Input: [[0, 30],[5, 10],[15, 20]]
Output: 2
Example 2:

Input: [[7,10],[2,4]]
Output: 1
NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
'''

class Solution10:
    # Time: O(NlogN) because of the sorting
    # Space: O(N)
    def minMeetingRooms(self, intervals: [[int]]) -> int:
        # We will seperate the start times from the end times and create 2 lists, one for each
        # This will break the concept of meetings but it won't matter as we only care about the 
        #   usage of rooms, and if a start point is smaller than a end point, it means that those
        #   two meetings collapse with each other and we need two rooms for them
        
        starts = [x[0] for x in intervals]
        ends = [x[1] for x in intervals]
        starts.sort()
        ends.sort()
        
        # p1 goes along starts, and p2 goes along ends, if p1 < p2, it means we need an extra room,
        #   else, it means we no longer need to consider the current p2 and can move forward
        p1, p2 = 0, 0
        result = 0
        while p1 < len(starts):
            if starts[p1] < ends[p2]:
                result += 1
            else:
                p2 += 1
            p1 += 1
        
        return result
            

'''
953. Verifying an Alien Dictionary
Easy

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographicaly in this alien language.


Example 1:

Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
Example 2:

Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.
Example 3:

Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
Explanation: The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character (More info).
 

Constraints:

1 <= words.length <= 100
1 <= words[i].length <= 20
order.length == 26
All characters in words[i] and order are English lowercase letters.
'''

class Solution11:
    def isAlienSorted(self, words: [str], order: str) -> bool:
        # Create a map that map a letter to the lexicographicaly order in the alien order so
        #   that when we compare two strings, we can get the result in O(1) time
        # Then, we just loop through the words and return false once we find a misplaced str
        
        wordMap = {}
        for i in range(len(order)):
            wordMap[order[i]] = i
            
        # Return negative int if w1 should be in front of w2
        def compare(w1, w2):
            p1 = p2 = 0
            diff = 0    
            # if diff == 0, it means we haven't find out a difference yet
            while p1 < len(w1) and p2 < len(w2) and not diff:
                diff = wordMap[w1[p1]] - wordMap[w2[p2]]    # use the map to get the diff in order
                p1 += 1
                p2 += 1
            
            if diff:    # if we've found a difference, we return it
                return diff
            else:       # if not, the shorter one should be in the front
                return len(w1) - len(w2)
            
        # loop through the words, return if we find one out of ouder
        for i in range(len(words) - 1):
            if compare(words[i], words[i + 1]) > 0:
                return False
        
        return True


'''
380. Insert Delete GetRandom O(1)
Medium

Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
Example:

// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();
'''

# Solution 12
class RandomizedSet:
    # Time: O(1) O(1) O(1)
    
    # Use a dictionary to achieve the O(1) insertion and removal
    # Use a list to achieve the O(1) getting random

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.myDict = {}
        self.myList = []
        

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.myDict:
            return False
        
        self.myDict[val] = len(self.myList)
        self.myList.append(val)
        
        return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        # to remove from a list in constant time, we first swap the element that we want to remove
        #   with the last element in the list, then we pop the last item from the list
        
        if val not in self.myDict:
            return False
        
        currIndex = self.myDict[val]
        lastItem = self.myList[-1]
        
        # remove current val
        self.myDict[lastItem] = currIndex   # modify the last item key in our dict
        self.myDict.pop(val)    # pop val key from dict
   
        self.myList[currIndex] = lastItem   # paste the last item on curr val's place
        self.myList.pop()   # pop current last item
        
        return True
        

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        import random
        
        return self.myList[random.randint(0, len(self.myList) - 1)]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()


'''
224. Basic Calculator
Hard

Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .

Example 1:

Input: "1 + 1"
Output: 2
Example 2:

Input: " 2-1 + 2 "
Output: 3
Example 3:

Input: "(1+(4+5+2)-3)+(6+8)"
Output: 23
Note:
You may assume that the given expression is always valid.
Do not use the eval built-in library function.
'''

class Solution13:
    # Time: O(N)  Space: O(N)
    def calculate(self, s: str) -> int:
        # For this kind of problems with brackets, we use Stack
        # Go from left to right, push into a Stack, once we reach a closing bracket ')', we pop 
        #   from the stack until we reach the '('. This way, we can calculate everything inside the 
        #   bracket first. 
        # Sicne stack is first in last out, we will actually traverse through the string in its 
        #   reversed order, which will cause issue for subtraction. For ex, '5-3' will become '3-5'.
        # To fix this, we reverse current number once we see a '-', then do addition only. For ex, 
        # '5-3' will become -3 + 5. This will eliminate the issue as addition doesn't care about 
        # order. 
        
        # The last round of calculation will be left in the stack since we don't pop the stack unless
        #   we've reached a closing bracket. Therefore, we include the whole string inside a pair of 
        #   brackets so that the stack will always be emptied in the end
        
        s = '(' + s + ')'
        stack = []
        
        for c in s:
            if c == ' ':
                continue
            elif c == ')':  # Calculate the value inside a bracket as it has higher priority
                num = 0
                digit = 0
                otherNum = 0
                while stack:
                    n = stack.pop()
                    if n == '+':
                        # Once we've reached a + or - sign, we should do the calculation
                        otherNum = num + otherNum  
                        num = 0
                        digit = 0
                    elif n == '-':
                        otherNum = -num + otherNum  # -num because we want to subtract the val 
                        num = 0
                        digit = 0
                    elif n == '(': 
                        # Append the result back to the stack for outer bracket calculation use
                        result = num + otherNum
                        stack.append(str(result))
                        break
                    else:                   # In this case, n must be a number
                        num = int(n) * 10 ** digit + num
                        digit += 1
            else:
                stack.append(c)
        
        return stack.pop()


'''
981. Time Based Key-Value Store
Medium

Create a timebased key-value store class TimeMap, that supports two operations.

1. set(string key, string value, int timestamp)

Stores the key and value, along with the given timestamp.
2. get(string key, int timestamp)

Returns a value such that set(key, value, timestamp_prev) was called previously, with timestamp_prev <= timestamp.
If there are multiple such values, it returns the one with the largest timestamp_prev.
If there are no values, it returns the empty string ("").
 

Example 1:

Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
Output: [null,null,"bar","bar",null,"bar2","bar2"]
Explanation:   
TimeMap kv;   
kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
kv.get("foo", 1);  // output "bar"   
kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
kv.set("foo", "bar2", 4);   
kv.get("foo", 4); // output "bar2"   
kv.get("foo", 5); //output "bar2"   

Example 2:

Input: inputs = ["TimeMap","set","set","get","get","get","get","get"], inputs = [[],["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
Output: [null,null,null,"","high","high","low","low"]
 

Note:

All key/value strings are lowercase.
All key/value strings have length in the range [1, 100]
The timestamps for all TimeMap.set operations are strictly increasing.
1 <= timestamp <= 10^7
TimeMap.set and TimeMap.get functions will be called a total of 120000 times (combined) per test case.
'''

# Solution 14
class TimeMap:
    # Use a dictionary to store the key, with it's value and timestamp
    # Since 'The timestamps for all TimeMap.set operations are strictly increasing.', we are sure
    #   that the items will be put inside in increasing order of timestamp, we can use binary search
    #   to find a specific item in O(logN) time

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.timeDict = {}
        
    # O(1)
    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.timeDict:
            self.timeDict[key].append((value, timestamp))
        else:
            self.timeDict[key] = [(value, timestamp)]
        
    # O(logN)
    def get(self, key: str, timestamp: int) -> str:
        # Binary Search by comparing timestamp
        if not key in self.timeDict:
            return ""
        vals = self.timeDict[key]
                
        # We can also use libraries for binary search, ex: 'bisect_right'
        # I implemented the binary search my myself for the best performance. 
        def binarySearch(values, key):
            l = len(values)
            lo = 0
            hi = l - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if values[mid][1] <= key: 
                    # if curr is smaller than key but the next is larger than key (or curr is already
                    #   the last item), return curr
                    if mid == l - 1 or values[mid + 1][1] > key:
                        return vals[mid][0]
                
                    else:   # else, we should search the right part
                        lo = mid + 1
                else:   # search the left part
                    hi = mid - 1
            return ""
                    
        return binarySearch(vals, timestamp)


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)


'''
221. Maximal Square
Medium

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example:

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4
'''

class Solution15:
    # Time: O(W*H)  Space: O(W)
    def maximalSquare(self, matrix: [[str]]) -> int:
        # Dynamic Programming
        # Let dp[i][j] = the side length of the largest square with the bottom right corner as
        #   matrix[i][j]. We can get dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        # This is because any small matrix on the left, top, or top left side of point [i][j]
        #   will make the total square side size be limited to min + 1
    
        # Regular dp will use a 2D dp matrix of size W*H. However, we can optimize the space usage
        #   to W, as we only need to use one row (i-1) on the dp matrix
        
        h = len(matrix)
        if h == 0:
            return 0
        w = len(matrix[0])
        dp = [int(x) for x in matrix[0]]
        curMax = max(dp)
        
        for i in range(1, h):
            nex = [int(matrix[i][0])] + [0] * (w - 1)
            for j in range(1, w):
                val = min(dp[j], nex[j - 1], dp[j - 1]) + 1 if matrix[i][j] == '1' else 0
                curMax = max(curMax, val)
                nex[j] = val
            curMax = max(curMax, max(nex))
            dp = nex
                
        return curMax * curMax

