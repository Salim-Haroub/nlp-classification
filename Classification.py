#!/usr/bin/env python
# coding: utf-8

# In[113]:


import nltk
import random
from nltk.corpus import movie_reviews, twitter_samples, names, nps_chat
from random import shuffle


# In[166]:


def gender_features(word):
    return{'last_letter': word[-1]}

labeled_names = ([(name, 'male') for name in names.words('male.txt')]
                 + [(name, 'female') for name in names.words('female.txt')])

shuffle(labeled_names)


# In[167]:


from nltk.classify import apply_features

train_set = apply_features(gender_features, labeled_names[500:])

test_set = apply_features(gender_features, labeled_names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[168]:


print('Erin is a ' + classifier.classify(gender_feature('Erin')))
print('Josh is a ' + classifier.classify(gender_feature('Josh')))


# In[169]:


print(nltk.classify.accuracy(classifier,test_set))


# In[170]:


print(classifier.show_most_informative_features(10))


# In[191]:


errors = []

for (name, tag) in labeled_names[:500]:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append([tag, guess, name])


# In[192]:


for (tag, guess, name) in sorted(errors):
    print('correct = {:<8} guessed = {:<8s} name = {:<30}'.format(tag, guess, name))


# In[193]:


train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]


# In[195]:


train_set2 = apply_features(gender_feature, train_names)
devtest_set = apply_features(gender_feature, devtest_names)
test_set2 = apply_features(gender_feature, test_names)


# In[198]:


classifier2 = nltk.NaiveBayesClassifier.train(train_set2)


# In[203]:


print('Erin is a ' + classifier2.classify(gender_feature('Erin')))


# In[206]:


print(nltk.classify.accuracy(classifier2, devtest_set))


# In[207]:


classifier2.show_most_informative_features(10)


# In[209]:


nltk.classify.accuracy(classifier2, test_set2)


# In[215]:


errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))


# In[216]:


for (tag, guess, name) in errors:
    print('correct = {:<8} guess = {:<8s} name = {:<30}'.format(tag, guess, name))


# In[220]:


#refining feature function
def gender_feature2(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}


# In[229]:


train_set3 = apply_features(gender_feature2, train_names)
devtest_set2 = apply_features(gender_feature2, devtest_names)
test_set3 = apply_features(gender_feature2, test_names)
Classifier = nltk.classify.NaiveBayesClassifier.train(train_set3)


# In[232]:


nltk.classify.accuracy(Classifier, devtest_set2)


# In[233]:


Classifier.show_most_informative_features(10)


# In[234]:


nltk.classify.accuracy(Classifier, test_set3)


# In[237]:


error = []
for (name, tag) in devtest_names:
    guess = Classifier.classify(gender_feature2(name))
    if guess != tag:
        error.append((tag, guess, name))


# In[252]:


for (tag, guess, name) in error:
    print('correct = {:<20} guess = {:<20} name = {}'.format(tag, guess, name))


# In[ ]:




