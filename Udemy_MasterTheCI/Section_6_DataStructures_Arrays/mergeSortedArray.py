# Merge two arrays that are sorted

# I'm using two pointers, one point for each array, then compare the value and move pointer accordingly

# Time: O(m + n) where m and n are length of list1 and list2

def mergeSortedArray(list1, list2):
  # check for special cases
  if list1 is None or type(list1) != list or len(list1) == 0:
    if list2 is None or type(list2) != list or len(list2) == 0:
      return []
    return list2
  if list2 is None or type(list2) != list or len(list2) == 0:
    return list1

  result = []

  i = 0
  j = 0
  # loop to add smaller item into the result
  while i < len(list1) or j < len(list2):
    if i >= len(list1) or list1[i] > list2[j]:
      result.append(list2[j])
      j += 1
    elif j >= len(list2) or list2[j] > list1[i]:
      result.append(list1[i])
      i += 1

  return result

# test
print(mergeSortedArray([1, 3, 5], [2, 4, 6]))
print(mergeSortedArray(None, [2, 4, 6]))
print(mergeSortedArray([1, 3, 5], []))