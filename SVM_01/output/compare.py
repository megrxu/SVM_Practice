import numpy

a = numpy.loadtxt('result.txt')
b = numpy.loadtxt('predicttestdatalabel.txt')

print(sum(a != b))
