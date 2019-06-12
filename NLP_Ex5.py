in_filename = 'E:\Subash\AI\Sentiment_Weightage.txt'
TotalScore = 0
Dict_Sentiment = {}
Sentiment_Dialouge = 'This is the Good climate but humidity is terrible'.lower()

fldetails = open(in_filename, 'r')
for lines in fldetails.readlines():
    Senti_Words, Score = lines.split('\t')
    Dict_Sentiment[Senti_Words] = int(Score)

for eachWord in Sentiment_Dialouge.split():
    WordScore = int(Dict_Sentiment.get(eachWord, 0))
    TotalScore += WordScore
print 'Senti Score for Sentence: ', TotalScore
if TotalScore < 0:
    print 'Statement is Negative'
elif TotalScore > 0:
    print 'Statement is Positive'
else:
    print 'Statement is Neutral'