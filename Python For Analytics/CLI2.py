import sys

if len(sys.argv) == 3:
    print('Sum of two arguments: ', eval(sys.argv[1]) + eval(sys.argv[2]))
else:
    print('Please Enter three argument including file name.')
          
