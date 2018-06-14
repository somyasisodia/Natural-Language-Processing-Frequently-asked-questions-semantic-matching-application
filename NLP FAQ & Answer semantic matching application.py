
# coding: utf-8

# In[2]:

import sys
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordDependencyParser
import logging
import json
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
import pysolr


# In[3]:

file=open("C:/Users/Somya/Anaconda3/Library/bin/train.txt","r")

questions = list()
answers = list()
questions_bag = list()
answers_bag = list()
question_answer_bag=list()
faq_bag=list()
filtered_sentences=list()
lemm_bag=list()
stem_bag=list()
pos_tagged_bag=list()
synset_bag=list()
hypernym_bag=list()
hyponym_bag=list()
holonym_bag=list()
meronym_bag=list()
parsing_bag=list()
        
for line in file:
    if line.strip():
        if line.startswith("Q:"):
            questions.append(line)
            questions_bag.append(word_tokenize(line.strip()))
        else:
            answers.append(line)
            answers_bag.append(word_tokenize(line.strip()))


for i in range(len(questions)):
    faq_bag.append(questions_bag[i]+answers_bag[i])
    question_answer_bag.append(questions[i]+answers[i])
    
# REMOVING STOP WORDS
stop_words = set(stopwords.words("english"))

for i in range(len(questions)):
    filtered_sentence = [w for w in faq_bag[i] if not w in stop_words]
    filtered_sentences.append(filtered_sentence)
    

    
   
for sentence in filtered_sentences:
    bag1=list()
    bag2=list()
    bag3=list()
    bag4=list()
    for word in sentence:
        word=wn.synsets(word)
    
        for l in word:
            for hyponym in l.hyponyms():
                for lemma in hyponym.lemmas():
                    bag1.append(lemma.name())
                    
            for hypernym in l.hypernyms():
                for lemma in hypernym.lemmas():
                    bag2.append(lemma.name())
                    
            for holonym in l.member_holonyms():
                for lemma in holonym.lemmas():
                    bag3.append(lemma.name())
                    
            for meronym in l.member_meronyms():
                for lemma in meronym.lemmas():
                    bag4.append(lemma.name())
                    
    hyponym_bag.append(bag1)
    hypernym_bag.append(bag2)
    holonym_bag.append(bag3)
    meronym_bag.append(bag4)
 
    
#path_to_jar="G:/Somya/Semester_Spring/NLP/Project/stanford-parser-full-2018-02-27/stanford-parser.jar"
#path_to_models_jar='G:/Somya/Semester_Spring/NLP/Project/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
#dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

nlp = StanfordCoreNLP(r'G:\Somya\Semester_Spring\NLP\Project\stanford-corenlp-full-2017-06-09')
for l in question_answer_bag:
    bag=list()
    for sentence in sent_tokenize(l):
        #dep = next(dependency_parser.raw_parse(sentence))
        bag.append(nlp.dependency_parse(sentence))
    parsing_bag.append(bag)

#LEMMATIZE WORDS
lemmatizer = WordNetLemmatizer()
for l in filtered_sentences:
    lemm=list()
    for word in l:
        lemm.append(lemmatizer.lemmatize(word))
    lemm_bag.append(lemm)


#STEM WORDS
stemmer = PorterStemmer()
for l in filtered_sentences:
    stem=list()
    for word in l:
        stem.append(stemmer.stem(word))
    stem_bag.append(stem)


#POS TAGGING
for l in faq_bag:
    tokens_pos= pos_tag(l)
    pos_tagged_bag.append(tokens_pos)


# In[38]:

with open('C:/Users/Somya/NlpProjectFeatureOutputs/WordsWithoutStopWords.txt', "w") as text_file:
    for l in filtered_sentences:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Lemmas.txt', "w") as text_file:
    for l in lemm_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Stems.txt', "w") as text_file:
    for l in stem_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/POS_Tag.txt', "w") as text_file:
    for l in pos_tagged_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/ParsedSentences.txt', "w") as text_file:
    for l in parsing_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Holonyms.txt', "w") as text_file:
    for l in holonym_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Hypernyms.txt', "w") as text_file:
    for l in hypernym_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Meronyms.txt', "w") as text_file:
    for l in meronym_bag:
            text_file.write('%s\n' %l)
            
with open('C:/Users/Somya/NlpProjectFeatureOutputs/Hyponyms.txt', "w") as text_file:
    for l in hyponym_bag:
            text_file.write('%s\n' %l)


# In[4]:

features=list()
for i in range(len(questions)):
    features.append(filtered_sentences[i] + lemm_bag[i] + stem_bag[i] + pos_tagged_bag[i] + parsing_bag[i] + holonym_bag[i] + hypernym_bag[i] + meronym_bag[i] + hyponym_bag[i])
    #features.extend(lemm_bag[i])


# In[48]:

#Task 2 Computation
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(question_answer_bag)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

target=list(range(1, 68))
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, target)
print(clf)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

text_clf = text_clf.fit(questions, target)

import numpy as np
test_list=list()
user_input = input("enter the FAQ?  ") 
question=user_input
test_list.append(question)
predicted = text_clf.predict_proba(test_list)
np.mean(predicted == target)
x=np.argsort(predicted)[0][-10:]
print(x)


for result in x:
    print(questions[result]+'\n'+answers[result]+'\n\n')


# In[87]:

## Task 4 computation
solr = pysolr.Solr('http://localhost:8983/solr/nlpTask4', timeout=10)
id=1
for l in features:
    name='document '+str(id)
    doc={'id':id, 'name': name, 'text': l}
    solr.add([doc])
    id += 1


# In[98]:

#Task 4 results
user_input = input("enter the FAQ?  ") 
results = solr.search(user_input)

for result in results:
    print(result['name'])

for result in results:
    print(result['name'])
    print(questions[int(result['id'])-1]+'\n'+answers[int(result['id'])-1]+'\n\n')


# In[99]:

#Task 2 results

import numpy as np
test_list=list()
user_input = input("enter the FAQ?  ") 
question=user_input
test_list.append(question)
predicted = text_clf.predict_proba(test_list)
np.mean(predicted == target)
x=np.argsort(predicted)[0][-10:]
x=x[::-1]
print(x)
print()
print()
for result in x:
    print(questions[result]+'\n'+answers[result]+'\n\n')


# In[ ]:



