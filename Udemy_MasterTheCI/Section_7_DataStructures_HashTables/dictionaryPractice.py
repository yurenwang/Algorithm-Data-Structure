def myFunc():
  print('ahhhhhh!')

myDict = {
  'age': 54,
  'name': 'Fred',
  'magic': True,
  'scream': myFunc
}

print(myDict['age'])
print(myDict['name'])
print(myDict['magic'])
myDict['scream']()