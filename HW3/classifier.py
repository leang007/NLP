import nltk
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
training_corpus = [
                   ('I am exhausted of this work.', 'Class_B'),
                   ("I can't cooperate with this", 'Class_B'),
                   ('He is my badest enemy!', 'Class_B'),
                   ('My management is poor.', 'Class_B'),
                   ('I love this burger.', 'Class_A'),
                   ('This is an brilliant place!', 'Class_A'),
                   ('I feel very good about these dates.', 'Class_A'),
                   ('This is my best work.', 'Class_A'),
                   ("What an awesome view", 'Class_A'),
                   ('I do not like this dish', 'Class_B'),
                    #new_A
                   ('I love to go shopping', 'Class_A'),
                   ('I am rich and famous.', 'Class_A'),
                   ('I could not be better','Class_A'),
                   ('She is my happiness','Class_A'),
                   ('Her smile makes me feel good','Class_A'),
                   #new_B
                   ('I can not go anywhere.', 'Class_B'),
                   ('I just got chewed out by my boss!', 'Class_B'),
                   ('Covid-19 make me feel bad', 'Class_B'),
                   ('I am done with you','Class_B'),
                   ('I do not trust him','Class_B')
                   ]
test_corpus =   [
                 ("I am not feeling well today.", 'Class_B'), 
                 ("I feel brilliant!", 'Class_A'), 
                 ('Gary is a friend of mine.', 'Class_A'), 
                 ("I can't believe I'm doing this.", 'Class_B'), 
                 ('The date was good.', 'Class_A'),
                 ('I do not enjoy my job', 'Class_B'),
                 #new
                 ('Covid-19 make me sad', 'Class_B'),
                 ('I do not love him', 'Class_B'),
                 ('This dish was terrible' , 'Class_B'),
                 ('Shopping mall is open now', 'Class_A'),
                 ('I am out of money!','Class_B')
                ]
model = NBC(training_corpus) 
print("Their codes are amazing.",model.classify("Their codes are amazing."))
print("I don't like their computer.",model.classify("I don't like their computer."))
print(model.accuracy(test_corpus))




