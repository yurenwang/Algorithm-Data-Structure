# Find the first recurring item in a list

# I used a set (or hashtable) to put item into it everytime it occurs, and as long as we find an item that exists in the set, we return it

# Time: O(n)  Space: O(n)

def firstRecurringChar(input):
  # First check if input is valid
  if (input == None or type(input) != list):
    return None

  # Store seen items in a set and loop, and check existance
  existed = set()
  for curr in input:
    if curr in existed:
      return curr
    existed.add(curr)

  return None

# Test
print(firstRecurringChar([1, 2, 3, 4]))
print(firstRecurringChar('Wrong Input'))
print(firstRecurringChar([1, 2, 3, 4, 2]))
print(firstRecurringChar([1, 2, 3, 4, 5, 4, 3]))