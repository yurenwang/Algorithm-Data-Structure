# Reverse a string

# I split the string into a list, then reverse using two pointers and while loop, then join back to a string
# Remember to check invalid input!

# Time: O(n) Space: O(n)

def reverseAString(inputString):
  if inputString is None or type(inputString) != str:
    return ''

  stringList = [char for char in inputString]

  i = 0
  j = len(stringList) - 1

  while i < j:
    tmp = stringList[i]
    stringList[i] = stringList[j]
    stringList[j] = tmp
    i += 1
    j -= 1

  return ''.join(stringList)

print(reverseAString('Hello'))
print(reverseAString('Frederick'))
print(reverseAString(''))
print(reverseAString(None))
print(reverseAString(4))

#######################################################
# Use string concatenation with reverse order
# Time: O(n)
def reverseAString2(inputString):
  if inputString is None or type(inputString) != str:
    return ''

  result = ''
  for char in inputString:
    result = char + result

  return result

print(reverseAString2('Hello'))
print(reverseAString2('Frederick'))
print(reverseAString2(''))
print(reverseAString2(None))
print(reverseAString2(4))