'''
Leetcode problems following the order of Nick White's playlist - VI

1184. Distance Between Bus Stops
348. Design Tic-Tac-Toe
819. Most Common Word
496. Next Greater Element I
1019. Next Greater Node In Linked List
821. Shortest Distance to a Character
859. Buddy Strings
925. Long Pressed Name
838. Push Dominoes
824. Goat Latin

'''


'''
1184. Distance Between Bus Stops
Easy

A bus has n stops numbered from 0 to n - 1 that form a circle. We know the distance between all pairs of neighboring stops where distance[i] is the distance between the stops number i and (i + 1) % n.

The bus goes along both directions i.e. clockwise and counterclockwise.

Return the shortest distance between the given start and destination stops.

Example 1: (Images at https://leetcode.com/problems/distance-between-bus-stops/)

Input: distance = [1,2,3,4], start = 0, destination = 1
Output: 1
Explanation: Distance between 0 and 1 is 1 or 9, minimum is 1.
 
Example 2:

Input: distance = [1,2,3,4], start = 0, destination = 2
Output: 3
Explanation: Distance between 0 and 2 is 3 or 7, minimum is 3.
 
Example 3:

Input: distance = [1,2,3,4], start = 0, destination = 3
Output: 4
Explanation: Distance between 0 and 3 is 6 or 4, minimum is 4.
 
Constraints:

1 <= n <= 10^4
distance.length == n
0 <= start, destination < n
0 <= distance[i] <= 10^4

'''

class Solution1:
    def distanceBetweenBusStops(self, distance: [int], start: int, destination: int) -> int:
        # Time: O(N)  Space: O(1)
        # there are always 2 choices:
        l = len(distance)
        choice1 = choice2 = 0
        small, large = min(start, destination), max(start, destination)
        for i in range(small, large):
            choice1 += distance[i]
        
        for i in range(large, small + l):
            choice2 += distance[i % l]
            
        return min(choice1, choice2)


'''
348. Design Tic-Tac-Toe
Medium

Assume the following rules are for the tic-tac-toe game on an n x n board between two players:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves are allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
Implement the TicTacToe class:

TicTacToe(int n) Initializes the object the size of the board n.
int move(int row, int col, int player) Indicates that player with id player plays at the cell (row, col) of the board. The move is guaranteed to be a valid move.
Follow up:
Could you do better than O(n2) per move() operation?

 

Example 1:

Input
["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
Output
[null, 0, 0, 0, 0, 0, 0, 1]

Explanation
TicTacToe ticTacToe = new TicTacToe(3);
Assume that player 1 is "X" and player 2 is "O" in the board.
ticTacToe.move(0, 0, 1); // return 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).
| | | |

ticTacToe.move(0, 2, 2); // return 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).
| | | |

ticTacToe.move(2, 2, 1); // return 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).
| | |X|

ticTacToe.move(1, 1, 2); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).
| | |X|

ticTacToe.move(2, 0, 1); // return 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).
|X| |X|

ticTacToe.move(1, 0, 2); // return 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).
|X| |X|

ticTacToe.move(2, 1, 1); // return 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).
|X|X|X|
 

Constraints:

2 <= n <= 100
player is 1 or 2.
1 <= row, col <= n
(row, col) are unique for each different call to move.
At most n2 calls will be made to move.

'''

# Solution2
class TicTacToe:

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.n = n
        self.hori_sum = [0] * n
        self.ver_sum = [0] * n
        self.tl_br = 0
        self.tr_bl = 0
        

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        # Time: O(N)  Space: O(1)
        
        # add 1 to sums if it is player 1 and subtract 1 from sums if it is not
        addition = 1 if player == 1 else -1
        
        self.hori_sum[row] += addition
        self.ver_sum[col] += addition
        if row == col:
            self.tl_br += addition
        if row + col + 1 == self.n:
            self.tr_bl += addition
        
        if max(self.hori_sum) == self.n or max(self.ver_sum) == self.n or self.tl_br == self.n or self.tr_bl == self.n:
            return 1
        if min(self.hori_sum) + self.n == 0 or min(self.ver_sum) + self.n == 0 or self.tl_br + self.n == 0 or self.tr_bl + self.n == 0:
            return 2
        return 0
        


# Your TicTacToe object will be instantiated and called as such:
# obj = TicTacToe(n)
# param_1 = obj.move(row,col,player)


'''
819. Most Common Word
Easy

Given a paragraph and a list of banned words, return the most frequent word that is not in the list of banned words.  It is guaranteed there is at least one word that isn't banned, and that the answer is unique.

Words in the list of banned words are given in lowercase, and free of punctuation.  Words in the paragraph are not case sensitive.  The answer is in lowercase.

 

Example:

Input: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph. 
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"), 
and that "hit" isn't the answer even though it occurs more because it is banned.
 

Note:

1 <= paragraph.length <= 1000.
0 <= banned.length <= 100.
1 <= banned[i].length <= 10.
The answer is unique, and written in lowercase (even if its occurrences in paragraph may have uppercase symbols, and even if it is a proper noun.)
paragraph only consists of letters, spaces, or the punctuation symbols !?',;.
There are no hyphens or hyphenated words.
Words only consist of letters, never apostrophes or other punctuation symbols.

'''

class Solution3:
    def mostCommonWord(self, paragraph: str, banned: [str]) -> str:
        # Time: O(M+N) where M is length of paragraph and N is size of banned list
        # Space: O(M+N)
        # it is like a pipeline:
        
        # First remove all punctuations and make it lower case
        pure_letter = ''.join([c if c == ' ' or c.isalpha() else ' ' for c in paragraph]).lower()
        
        # Then split into a list and remove all banned words
        tmp = pure_letter.split()
        ban_set = set(banned)
        l = [s for s in tmp if s not in ban_set]
        
        # Then count and find the most occured word
        counter = collections.Counter(l)
        max_occur = max(counter.values())
        
        # Then check each word and return the result
        for w in counter:
            if w not in ban_set and counter[w] == max_occur:
                return w


'''
496. Next Greater Element I
Easy

You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2. Find all the next greater numbers for nums1's elements in the corresponding places of nums2.

The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. If it does not exist, output -1 for this number.

Example 1:
Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
Output: [-1,3,-1]
Explanation:
    For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
    For number 1 in the first array, the next greater number for it in the second array is 3.
    For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
Example 2:
Input: nums1 = [2,4], nums2 = [1,2,3,4].
Output: [3,-1]
Explanation:
    For number 2 in the first array, the next greater number for it in the second array is 3.
    For number 4 in the first array, there is no next greater number for it in the second array, so output -1.
Note:
All elements in nums1 and nums2 are unique.
The length of both nums1 and nums2 would not exceed 1000.

'''

class Solution4:
    def nextGreaterElement(self, nums1: [int], nums2: [int]) -> [int]:
        # Use a stack and a dictionary to find out first larger items right to a number
        # In the stack, we push every time when we encounter a smaller than top number, if new number is
        #   larger than the stack top, we keep popping the stack until we find a larger one, and for all
        #   numbers popped from the stack, we know that their next large right number is the current one.
        
        # Time: O(M+N)  Space: O(M), M is size of nums2 and N is size of nums1
        stack = []
        d = {}
        for n in nums2:
            while stack and stack[-1] < n:
                d[stack.pop()] = n
            stack.append(n)
        return [d[n] if n in d else -1 for n in nums1]


'''
1019. Next Greater Node In Linked List
Medium

We are given a linked list with head as the first node.  Let's number the nodes in the list: node_1, node_2, node_3, ... etc.

Each node may have a next larger value: for node_i, next_larger(node_i) is the node_j.val such that j > i, node_j.val > node_i.val, and j is the smallest possible choice.  If such a j does not exist, the next larger value is 0.

Return an array of integers answer, where answer[i] = next_larger(node_{i+1}).

Note that in the example inputs (not outputs) below, arrays such as [2,1,5] represent the serialization of a linked list with a head node value of 2, second node value of 1, and third node value of 5.

 

Example 1:

Input: [2,1,5]
Output: [5,5,0]
Example 2:

Input: [2,7,4,3,5]
Output: [7,0,5,5,0]
Example 3:

Input: [1,7,5,1,9,2,5,1]
Output: [7,9,9,9,0,5,0,0]
 

Note:

1 <= node.val <= 10^9 for each node in the linked list.
The given list has length in the range [0, 10000].

'''

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution5:
    def nextLargerNodes(self, head: ListNode) -> [int]:
        # Similar to question 496, we use a stack and a dictionary to find out the next larger
        # For all questions about finding next larger or smaller value, we should consider using Stack
        
        # Time: O(M)  Space: O(M)
        stack = []
        d = {}
        curr = head
        i = 0
        
        while curr:
            v = curr.val
            while stack and stack[-1][0] < v:
                d[stack.pop()] = v
            stack.append((v, i))
            curr = curr.next
            i += 1
                    
        res = [0] * (i)
        for n, i in d:
            res[i] = d[(n, i)]
        
        return res


'''
821. Shortest Distance to a Character
Easy

Given a string S and a character C, return an array of integers representing the shortest distance from the character C in the string.

Example 1:

Input: S = "loveleetcode", C = 'e'
Output: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
 

Note:

S string length is in [1, 10000].
C is a single character, and guaranteed to be in string S.
All letters in S and C are lowercase.

'''

class Solution6:
    def shortestToChar(self, S: str, C: str) -> [int]:
        # Find the shortest destence to C on the left first, then find it on the right, then get min
        # Scan throught the list to find distances
        
        # Time: O(N)  Space: O(N)
        distance = []
        tmp = len(S) + 1
        for i in range(len(S)):
            if S[i] == C:
                tmp = 0
            else:
                tmp += 1
            distance.append(tmp)
        
        tmp = len(S) + 1
        for i in range(len(S) - 1, -1, -1):
            if S[i] == C:
                tmp = 0
            else:
                tmp += 1
            distance[i] = min(distance[i], tmp)
            
        return distance


'''
859. Buddy Strings
Easy

Given two strings A and B of lowercase letters, return true if you can swap two letters in A so the result is equal to B, otherwise, return false.

Swapping letters is defined as taking two indices i and j (0-indexed) such that i != j and swapping the characters at A[i] and A[j]. For example, swapping at indices 0 and 2 in "abcd" results in "cbad".

===== Example 1:

Input: A = "ab", B = "ba"
Output: true
Explanation: You can swap A[0] = 'a' and A[1] = 'b' to get "ba", which is equal to B.

===== Example 2:

Input: A = "ab", B = "ab"
Output: false
Explanation: The only letters you can swap are A[0] = 'a' and A[1] = 'b', which results in "ba" != B.

===== Example 3:

Input: A = "aa", B = "aa"
Output: true
Explanation: You can swap A[0] = 'a' and A[1] = 'a' to get "aa", which is equal to B.

===== Example 4:

Input: A = "aaaaaaabc", B = "aaaaaaacb"
Output: true

===== Example 5:

Input: A = "", B = "aa"
Output: false
 
Constraints:

0 <= A.length <= 20000
0 <= B.length <= 20000
A and B consist of lowercase letters.

'''

class Solution7:
    def buddyStrings(self, A: str, B: str) -> bool:
        # meaningless question
        
        # Time: O(N)  Space: O(N)
        
        if len(A) != len(B): 
            return False
        
        if A == B:
            s = set()
            for a in A:
                if a in s:
                    return True
                s.add(a)
            return False
        
        pairs = []
        for a, b in zip(list(A), list(B)):
            if a != b:
                pairs.append((a, b))
            if len(pairs) > 2:
                return False
        
        return len(pairs) == 2 and pairs[0] == pairs[1][::-1]
        

'''
925. Long Pressed Name
Easy

Your friend is typing his name into a keyboard.  Sometimes, when typing a character c, the key might get long pressed, and the character will be typed 1 or more times.

You examine the typed characters of the keyboard.  Return True if it is possible that it was your friends name, with some characters (possibly none) being long pressed.


Example 1:

Input: name = "alex", typed = "aaleex"
Output: true
Explanation: 'a' and 'e' in 'alex' were long pressed.
Example 2:

Input: name = "saeed", typed = "ssaaedd"
Output: false
Explanation: 'e' must have been pressed twice, but it wasn't in the typed output.
Example 3:

Input: name = "leelee", typed = "lleeelee"
Output: true
Example 4:

Input: name = "laiden", typed = "laiden"
Output: true
Explanation: It's not necessary to long press any character.
 

Constraints:

1 <= name.length <= 1000
1 <= typed.length <= 1000
The characters of name and typed are lowercase letters.

'''

class Solution8:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        # two pointers
        # Time: O(N)  Space: O(1)
        prev = None
        p1 = p2 = 0
        
        while p1 < len(name):
            if p2 >= len(typed):
                return False
            n, t = name[p1], typed[p2]
            if n == t:
                prev = n
                p1 += 1
                p2 += 1
            elif t == prev:
                p2 += 1
            else:
                return False
        
        while p2 < len(typed):
            if typed[p2] != prev:
                return False
            p2 += 1
        
        return True


'''
838. Push Dominoes
Medium

There are N dominoes in a line, and we place each domino vertically upright.

In the beginning, we simultaneously push some of the dominoes either to the left or to the right.

(Check the image at https://leetcode.com/problems/push-dominoes/)

After each second, each domino that is falling to the left pushes the adjacent domino on the left.

Similarly, the dominoes falling to the right push their adjacent dominoes standing on the right.

When a vertical domino has dominoes falling on it from both sides, it stays still due to the balance of the forces.

For the purposes of this question, we will consider that a falling domino expends no additional force to a falling or already fallen domino.

Given a string "S" representing the initial state. S[i] = 'L', if the i-th domino has been pushed to the left; S[i] = 'R', if the i-th domino has been pushed to the right; S[i] = '.', if the i-th domino has not been pushed.

Return a string representing the final state. 

Example 1:

Input: ".L.R...LR..L.."
Output: "LL.RR.LLRRLL.."
Example 2:

Input: "RR.L"
Output: "RR.L"
Explanation: The first domino expends no additional force on the second domino.
Note:

0 <= N <= 10^5
String dominoes contains only 'L', 'R' and '.'

'''

class Solution9:
    def pushDominoes(self, dominoes: str) -> str:
        # We can group the dominoes into groups, formed by 'R'.....'L'
        # And anything outside of groups are not matter since they will stand still anyways
        
        # Time: O(N)  Space: O(N)
        res = list(dominoes)
        
        # Get rid of all dominoes that are not pushed and
        # Add a left to the beginning of list so that we can compare each item at i with item at i+1, and
        #   adding a left to the beginning can cover the first group, in our case, everything before (1, 'L')
        # Add a right to the end to cover the last part as well
        dominoes = [(-1, 'L')] + [(i, c) for (i, c) in enumerate(dominoes) if c != '.'] + [(len(dominoes), 'R')]
        
        # Get all groups
        for (i, x), (j, y) in zip(dominoes, dominoes[1:]):
            # if both bondaries are same direction, then everything in it will become the same direction
            if x == y:
                for k in range(i + 1, j):
                    res[k] = x
            # if two bondaries are different direction, and it is ('R...L'), then all dominoes in the middle will
            #   fall to the center, if it is odd number inside, the middle one will not move
            if x == 'R' and y == 'L':
                for k in range(i + 1, j):
                    if k < (i + j) / 2:
                        res[k] = 'R'
                    elif k > (i + j) / 2:
                        res[k] = 'L'
            # if two bondaries are different and is ('L...R'), we don't have to do anything
            
        return ''.join(res)
                    
        
'''
824. Goat Latin
Easy

A sentence S is given, composed of words separated by spaces. Each word consists of lowercase and uppercase letters only.

We would like to convert the sentence to "Goat Latin" (a made-up language similar to Pig Latin.)

The rules of Goat Latin are as follows:

If a word begins with a vowel (a, e, i, o, or u), append "ma" to the end of the word.
For example, the word 'apple' becomes 'applema'.
 
If a word begins with a consonant (i.e. not a vowel), remove the first letter and append it to the end, then add "ma".
For example, the word "goat" becomes "oatgma".
 
Add one letter 'a' to the end of each word per its word index in the sentence, starting with 1.
For example, the first word gets "a" added to the end, the second word gets "aa" added to the end and so on.
Return the final sentence representing the conversion from S to Goat Latin. 

 

Example 1:

Input: "I speak Goat Latin"
Output: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
Example 2:

Input: "The quick brown fox jumped over the lazy dog"
Output: "heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
 

Notes:

S contains only uppercase, lowercase and spaces. Exactly one space between each word.
1 <= S.length <= 150.

'''

class Solution10:
    def toGoatLatin(self, S: str) -> str:
        # meaningless question
        vowels = {'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U'}
        words = enumerate(S.split())
        res = []
        for i, w in words:
            l = list(w)
            if l[0] in vowels:
                l.append('ma')
            else:
                l.append(l[0])
                l.pop(0)
                l.append('ma')
            l.extend(['a'] * (i + 1))
            res.append(''.join(l))
        
        return ' '.join(res)