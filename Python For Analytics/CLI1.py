import sys

print ('Total Parameter given at command line: ', len(sys.argv))
print ('Values given in command line: ', sys.argv)
print ('First Parameter given at command line: ', sys.argv[0])
print ('Second Parameter given at command line: ', sys.argv[1])
print ('First Parameter given at command line: ', type(sys.argv[1]))
print ('Resultant value of all parameters: ', sys.argv[1] + sys.argv[2])
print ('Result using Function: ', int(sys.argv[1]) + int(sys.argv[2]))
print ('Result using Eval: ', eval(sys.argv[1]) + eval(sys.argv[2]))
