# Given 2 arrays, create a function that let's a user know (true/false) whether these two arrays contain any common items
# For Example:
# const array1 = ['a', 'b', 'c', 'x'];//const array2 = ['z', 'y', 'i'];
# should return false.
# -----------
# const array1 = ['a', 'b', 'c', 'x'];//const array2 = ['z', 'y', 'x'];
# should return true.

# 2 parameters - arrays - no size limit
# return true or false

array1 = ['a', 'b', 'c', 'c']
array2 = ['z', 'y', 'c']

# Solution check if each item is contained in the other list
def have_common_item(array1, array2):
  if array1 is None or array2 is None:
    return False

  dict = {}

  for item in array1:
    dict[item] = True

  for item in array2:
    if item in dict:
      return True

  return False

result = have_common_item(array1, array2)
print(result)