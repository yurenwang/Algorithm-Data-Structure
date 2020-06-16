'''
Implement a LinkedList with Python

What you need:
  1. Necessary classes and data structure
  2. append()
  3. prepend()
  4. insert()
  5. remove()
  6. reverse()
  7. printLinkedList()
'''

# Class for a single Node
class Node:
  def __init__(self, value = None):
    self.value = value
    self.next = None


# Class for our linked list
class LinkedList:
  def __init__(self, headValue):
    self.head = Node(headValue)
    self.tail = self.head
    self.length = 1


  # Appending to the end
  def append(self, val):
    newNode = Node(val)
    self.tail.next = newNode
    self.tail = newNode
    self.length += 1


  # Prepending to the front
  def prepend(self, val):
    newNode = Node(val)
    newNode.next = self.head
    self.head = newNode
    self.length += 1
    

  # Inserting to the middle at specified index
  def insert(self, index, val):
    if index <= 0:
      self.prepend(val)
      return
    if index >= self.length:
      self.append(val)
      return
  
    newNode = Node(val)
    prev = self.__traverseTo(index)
    
    newNode.next = prev.next
    prev.next = newNode


  # Removing the node at specified index
  def remove(self, index):
    # TODO: Check input
    
    prev = self.__traverseTo(index)
    prev.next = prev.next.next


  # Reversing a LinkedList
  def reverse(self):
    if self.length <= 1:
      return

    first = self.head
    second = first.next
    first.next = None
    self.tail = first

    while second:
      tmp = second.next
      second.next = first
      first = second
      second = tmp

    self.head = first


  # Helper: Traverse to the specified index
  def __traverseTo(self, index):
    currIndex = 1
    prev = self.head
    while currIndex < index:
      prev = prev.next
      currIndex += 1
    return prev


  # Print the LinkedList
  def printLinkedList(self):
    arr = [self.head.value]
    curr = self.head.next

    while curr:
      arr.append(curr.value)
      curr = curr.next

    print(arr)


# Tests
myLinkedList = LinkedList(100)
myLinkedList.append(3)
myLinkedList.append(5)
myLinkedList.prepend(7)

myLinkedList.printLinkedList()

myLinkedList.insert(2, 1)
myLinkedList.insert(-1, 8)
myLinkedList.insert(7, 9)

myLinkedList.printLinkedList()

myLinkedList.remove(1)
myLinkedList.remove(5)

myLinkedList.printLinkedList()

myLinkedList.reverse()

myLinkedList.printLinkedList()
