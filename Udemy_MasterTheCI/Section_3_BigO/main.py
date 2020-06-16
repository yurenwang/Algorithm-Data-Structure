def allPairs(input):
  for first in input:
    for second in input:
      if second != first:
        print('(%s, %r)' % (first, second))


allPairs([1, 2, 3, 4])
