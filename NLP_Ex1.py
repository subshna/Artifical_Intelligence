import pandas as pd
from nltk import sent_tokenize, ne_chunk

in_filepath = 'E:\Subash\AI\data_in.csv'
out_filepath = 'E:\Subash\AI\data_out.csv'

df = pd.read_csv(in_filepath, encoding='utf-8')
df['Sent_tokenized'] = df.apply(lambda row: sent_tokenize(row['Comment']), axis=1)
df['Sent_tokenized'].to_csv(out_filepath, index=False, encoding='utf-8')

for sent in df['Sent_tokenized']:
    print (ne_chunk(sent))

print 'Done'