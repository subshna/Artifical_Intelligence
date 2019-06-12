import sys

i=1
res=0
while(i!=len(sys.argv)):
    res+= int(sys.argv[i])
    i+=1
print('Sum of all values given in cmd line is: ', res)
          
