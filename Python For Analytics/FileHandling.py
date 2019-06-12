# File handling - read, readline(), readlines(), append
fd = open('SampleTestFile.txt','r')
print('The file handle opened at ',fd)
print('The file name ',fd.name)
print('The file Status ',fd.closed)
print('The file mode ',fd.mode)
Contentlist = fd.readlines()
print('File details: ', Contentlist)
Contentlist.append('11. New line added to the File')
print('After Appending - File details: ', Contentlist)
fd.close()

# File Handling - Write to file
fd1 = open('SampleTestFile1.txt','w')
print('The file handle opened at ',fd1)
print('The file name ',fd1.name)
print('The file Status ',fd1.closed)
print('The file mode ',fd1.mode)
text = 'This is the Line added to the file'
fd1.write(text)
fd1.close()
