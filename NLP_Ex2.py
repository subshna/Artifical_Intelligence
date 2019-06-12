import pandas as pd
from nltk import word_tokenize

in_filepath = 'E:\Subash\AI\data_in.csv'
out_filepath = 'E:\Subash\AI\data_out.csv'

df = pd.read_csv(in_filepath, encoding='utf-8')
df['Word_tokenized'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)
df['Word_tokenized'].to_csv(out_filepath, index=False, encoding='utf-8')
print 'Done'