sampleArray = ['a', 'b', 'c']
array2 = ['e', 'f']
sampleArray.append('d')

print(sampleArray)

sampleArray.extend(array2)
sampleArray.insert(0, 'x')

print(sampleArray)
print(max(sampleArray))