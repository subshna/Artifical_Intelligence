from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

in_filepath = 'E:\Subash\AI\data_in.txt'

fldetails = open(in_filepath, 'r')
for lines in fldetails.readlines():
    tokenizing = word_tokenize(lines)
    postagging = pos_tag(tokenizing)
    chunk_Sent = ne_chunk(postagging)
    print chunk_Sent
    chunk_Sent.draw()