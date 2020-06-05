'''
####################################### Stack and Queue ####################

232. Implement Queue using Stacks
225. Implement Stack using Queues
155. Min Stack
20. Valid Parentheses
739. Daily Temperatures
503. Next Greater Element II

'''

'''
232. Implement Queue using Stacks
Easy

Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.
Example:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false
Notes:

You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).
'''

class MyQueue:
    # Solve this using a very smart way. 
    # For push, always push to s1.
    # For pop, if s2 is empty, pop everything from s1 and push into s2, then pop from top of s2
    #   if s2 isn't empty, simply pop the top item
    
    # push takes O(1) time
    # pop takes O(N) time for worse case if s2 is empty BUT takes O(1) time on average

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # Use list to represent Stack 
        self.s1 = []
        self.s2 = []
        # Note the first item to make peek() faster
        self.first = None
        

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        if not self.s1:
            self.first = x
        self.s1.append(x)
        

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()
        

    def peek(self) -> int:
        """
        Get the front element.
        """
        # Utilize 'first' to make peek() faster than pop() when s2 is empty
        if not self.s2:
            return self.first
        return self.s2[-1]
        

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return not self.s1 and not self.s2
        

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


'''
225. Implement Stack using Queues
Easy

Implement the following operations of a stack using queues.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
empty() -- Return whether the stack is empty.
Example:

MyStack stack = new MyStack();

stack.push(1);
stack.push(2);  
stack.top();   // returns 2
stack.pop();   // returns 2
stack.empty(); // returns false
Notes:

You must use only standard operations of a queue -- which means only push to back, peek/pop from front, size, and is empty operations are valid.
Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or deque (double-ended queue), as long as you use only standard operations of a queue.
You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).
'''

class MyStack:
    # Well, either push() or pop() has to take O(N) time, while the other takes O(1)
    ### Three approaches:
    # 1. Simply push to end for push(), and when pop(), pop one by one and push to another queue, then
    #   return the last one. After it's done, swap q1 and q2
    # 2. When push(), push to a new queue, and pop everything from old queue to new queue. When pop(),
    #   simply pop.
    # 3. (Optimized so I will implement this one here) This one only use one queue. When push(), also 
    #   pop everything from q and push it back into it. pop() is simple pop.
    
    # I'll use a list to represent a queue, which is inefficient. Just assume poping from left is O(1)

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = []
        
    # O(N)
    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.pop(0))

    # O(1)
    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.q.pop(0)

    def top(self) -> int:
        """
        Get the top element.
        """
        return self.q[0]

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.q) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()


'''
155. Min Stack
Easy

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
 

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
 

Constraints:

Methods pop, top and getMin operations will always be called on non-empty stacks.
'''

class MinStack:
    # Use a List to store values, and another list to store mins so far

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.vals = []
        self.mins = []
        
    # O(1)
    def push(self, x: int) -> None:
        self.vals.append(x)
        if not self.mins or x <= self.mins[-1]:
            self.mins.append(x)

    # O(1)
    def pop(self) -> None:
        if self.mins and self.mins[-1] == self.vals.pop():
            self.mins.pop()

    def top(self) -> int:
        return self.vals[-1]

    def getMin(self) -> int:
        return self.mins[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


'''
20. Valid Parentheses
Easy

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true
'''

class Solution1:
    def isValid(self, s: str) -> bool:
        # put all the chars in a stack and pop it if the closing bracket comes
        map = {')': '(', '}': '{', ']': '['}
        stack = []
        for c in s:
            if c in map.keys():
                if not stack or stack.pop() != map.get(c):
                    return False
            else:
                stack.append(c)
        return not stack
                

'''
739. Daily Temperatures
Medium

Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].
'''

class Solution2:
    # Time: O(N)  Space: O(N)
    def dailyTemperatures(self, T: [int]) -> [int]:
        # Use a stack to store the indexes of the input T. Then if we find out that a new temperature 
        #   is higher than the temp of stack top, we know the days to wait is the differences of the two
        #   indexes
        stack = []
        result = [0] * len(T)
        # loop through all indexes of input
        for i in range(len(T)):
            # pop all days from the stack if their temp is lower than the curr, since we've found the 
            #   results for them
            while stack and T[stack[-1]] < T[i]:
                prev = stack.pop()
                result[prev] = i - prev
            stack.append(i)
        return result


'''
503. Next Greater Element II
Medium

Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element. The Next Greater Number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.

Example 1:
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number; 
The second 1's next greater number needs to search circularly, which is also 2.
Note: The length of given array won't exceed 10000.
'''

class Solution3:
    # Time: O(N)  Space: O(N)
    def nextGreaterElements(self, nums: [int]) -> [int]:
        # Use Stack to store the indexes of elements
        
        # We will loop for 2*N times, where N is the length of nums, because if we only loop for N time,
        #   some of the values will not be correct as they haven't seen the values before it in the circle
        l = len(nums)
        result = [-1] * l
        stack = []
        for i in range(l * 2):
            curr = nums[i % l]
            # if item on stack top is smaller than curr, it finds its next larger item, which is curr
            while stack and nums[stack[-1]] < curr:
                result[stack.pop()] = curr
            stack.append(i % l)
        return result


