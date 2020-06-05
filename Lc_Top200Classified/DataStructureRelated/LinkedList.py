'''
########################################### Linked List #########################

160. Intersection of Two Linked Lists
206. Reverse Linked List
21. Merge Two Sorted Lists
83. Remove Duplicates from Sorted List
19. Remove Nth Node From End of List
24. Swap Nodes in Pairs
445. Add Two Numbers II
234. Palindrome Linked List
725. Split Linked List in Parts
328. Odd Even Linked List

'''


'''
160. Intersection of Two Linked Lists
Easy

Write a program to find the node at which the intersection of two singly linked lists begins.

For example, the following two linked lists:


begin to intersect at node c1.

 

Example 1:


Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
Output: Reference of the node with value = 8
Input Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,0,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
 

Example 2:


Input: intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Reference of the node with value = 2
Input Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [0,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
 

Example 3:


Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: null
Input Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
Explanation: The two lists do not intersect, so return null.
 

Notes:

If the two linked lists have no intersection at all, return null.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in O(n) time and use only O(1) memory.
'''

# This one is pretty easy. 
# Two ways:
#	1. Using a HashTable. Return if find a key that exist 
#	2. Using two pointers. Let p1 to go through in listNode A, and p2 to go through in listNodeB. Once any
#		of it reaches to the end, redirect it to keep looping through the other listNode. If there is an 
#		intersection, two pointers will meet at the intersection. Otherwise they'll meet at null.

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 2-pointer solution
# Time: O(M + N)  Space: O(1)
def getIntersectionNode(headA: ListNode, headB: ListNode):
	p1 = headA
	p2 = headB
	while(p1 != p2):
		if p1 == None:
			p1 = headB
		else:
			p1 = p1.next
		if p2 == None:
			p2 = headA
		else:
			p2 = p2.next
	return p1


'''
206. Reverse Linked List
Easy

Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?
'''

# This is pretty easy. 

'''
21. Merge Two Sorted Lists
Easy

Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
'''

# This is pretty easy too.

'''
83. Remove Duplicates from Sorted List
Easy

Given a sorted linked list, delete all duplicates such that each element appear only once.

Example 1:

Input: 1->1->2
Output: 1->2
Example 2:

Input: 1->1->2->3->3
Output: 1->2->3
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def deleteDuplicates(head: ListNode):
	p = head
	while p:
		while p.next and p.next.val == p.val:
			p.next = p.next.next
		p = p.next
	return head


'''
19. Remove Nth Node From End of List
Medium

Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.

Follow up:

Could you do this in one pass?
'''

# Use two pointers. 


'''
24. Swap Nodes in Pairs
Medium

Given a linked list, swap every two adjacent nodes and return its head.

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example:

Given 1->2->3->4, you should return the list as 2->1->4->3.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# Time: O(N)  Space: O(1)
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # two pointers:
        if not head or not head.next:
            return head
        p1 = head
        # set the dummy node to keep the head, and to set the initial prev
        dummy = ListNode(-1)
        dummy.next = head
        prev = dummy
        while p1 and p1.next:
            # set up p2 and swap p1 and p2
            p2 = p1.next
            p1.next, p2.next = p2.next, p1
            # connect prev with the one in the front after swap
            prev.next = p2
            # update prev
            prev = p1
            # keep going
            p1 = p1.next
        return dummy.next


'''
445. Add Two Numbers II
Medium

You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# Time: O(N)  Space: O(N)
class Solution2:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        # Use two stacks to store the numbers in reversed order
        s1 = []
        s2 = []
        p1 = l1
        p2 = l2
        while p1:
            s1.append(p1.val)
            p1 = p1.next
        while p2:
            s2.append(p2.val)
            p2 = p2.next 
        
        carry = 0
        nextDigit = None
        while s1 or s2:
            x = s1.pop() if s1 else 0
            y = s2.pop() if s2 else 0
            tmp = carry + x + y
            currDigit = ListNode(tmp % 10)
            currDigit.next = nextDigit
            carry = tmp // 10
            nextDigit = currDigit

        return ListNode(carry, nextDigit) if carry else nextDigit


'''
234. Palindrome Linked List
Easy

Given a singly linked list, determine if it is a palindrome.

Example 1:

Input: 1->2
Output: false
Example 2:

Input: 1->2->2->1
Output: true
Follow up:
Could you do it in O(n) time and O(1) space?
'''

##### Method 1 #####
# A normal way to solve it is to store it in a list, then use two pointers to determine if it is palindrome
# Time: O(N)  Space: O(N)  

##### Method 2 #####
# Optimized way is to reverse the second half of the linked list in place, then determine if it's palindrome 
# Time: O(N)  Space: O(1)

# Specifically, the steps we need to do are:
# 1. Find the end of the first half. (use two runner, one moves 2 steps, and one moves 1 step each time)
# 2. Reverse the second half.
# 3. Determine whether or not there is a palindrome. (Easy, just use two pointers moving forward simutaneously)
# 4. Restore the list.
# 5. Return the result.


'''
725. Split Linked List in Parts
Medium

Given a (singly) linked list with head node root, write a function to split the linked list into k consecutive linked list "parts".

The length of each part should be as equal as possible: no two parts should have a size differing by more than 1. This may lead to some parts being null.

The parts should be in order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal parts occurring later.

Return a List of ListNode's representing the linked list parts that are formed.

Examples 1->2->3->4, k = 5 // 5 equal parts [ [1], [2], [3], [4], null ]
Example 1:
Input:
root = [1, 2, 3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The input and each element of the output are ListNodes, not arrays.
For example, the input root has root.val = 1, root.next.val = 2, \root.next.next.val = 3, and root.next.next.next = null.
The first element output[0] has output[0].val = 1, output[0].next = null.
The last element output[4] is null, but it's string representation as a ListNode is [].
Example 2:
Input: 
root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
Explanation:
The input has been split into consecutive parts with size difference at most 1, and earlier parts are a larger size than the later parts.
Note:

The length of root will be in the range [0, 1000].
Each value of a node in the input will be an integer in the range [0, 999].
k will be an integer in the range [1, 50].
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# Time: O(N + k)  Space: O(k)

# The implementation of the logic is kinda hard. Make sure to review.
class Solution3:
    def splitListToParts(self, root: ListNode, k: int):
        # Calculate the total length l first. Then l // k will be the length of each section. 
        # For the first l % k section, each will receive one extra member
        l = 0
        p = root
        while p:
            l += 1
            p = p.next
            
        # Use python built in divmod() to calculate the quotient and the remainder of l / k
        width, extra = divmod(l, k)
        
        result = []
        p = root
        # Loop through range k to fill in k listNodes in the result list
        for i in range(k):
            head = p
            result.append(head)
            # use a loop to reach the end of the current result section
            for j in range(width + (i < extra) - 1):
                if p:
                    p = p.next
            # disconnect the current section from next section if p is not null yet
            if p: 
                p.next, p = None, p.next
        
        return result


'''
328. Odd Even Linked List
Medium

Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
Example 2:

Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL
Note:

The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ...
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# Time: O(n)  Space: O(1)
class Solution4:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next or not head.next.next:
            return head
        p = head
        isCurrOdd = True
        firstEven = head.next
        
        # point all nodes to the next next item
        while p.next.next:
            tmp = p.next
            p.next = p.next.next
            p = tmp
            isCurrOdd = not isCurrOdd
        
        # if we meet the last odd node, simply connect it to the first even
        if isCurrOdd:
            p.next = firstEven
        # if we meet the last even, point next to None, and connect the next item to first even
        else:
            last = p.next
            p.next = None
            last.next = firstEven
        return head
