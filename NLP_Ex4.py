from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

in_filepath = 'E:\Subash\AI\data_in.txt'

fldetails = open(in_filepath, 'r')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatize = WordNetLemmatizer()

for lines in fldetails.readlines():
    word_Tokenization = word_tokenize(lines)
    filterwords = []
    for words in word_Tokenization:
        if words not in stop_words:
            filterwords.append(words)
    print 'Stop Words: -->', filterwords
    print 'Stemming: -->'
    print " ".join([stemmer.stem(word) for word in lines.split()])
    print 'Lemmatization: -->'
    print " ".join([lemmatize.lemmatize(word) for word in lines.split()])