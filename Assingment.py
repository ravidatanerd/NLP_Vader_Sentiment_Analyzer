#!/usr/bin/env python
# coding: utf-8

# ## Name: Ravi Dawar

# ### Task #1: Perform vector arithmetic on your own words

# 
# #### Write code that evaluates vector arithmetic on your own set of related words.The goal is to come as close to an expected word as possible.
# 

# In[1]:


# Import spaCy and load the language library. Remember to use a larger model!

import spacy
nlp = spacy.load('en_core_web_lg')


# In[2]:


# Choose the words you wish to compare, and obtain their vectors
gregarous = nlp.vocab['gregarious'].vector
egregious = nlp.vocab['egregious'].vector
convivial = nlp.vocab['convivial'].vector


# In[3]:


# Import spatial and define a cosine_similarity function
from scipy import spatial
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)


# In[4]:


# Write an expression for vector arithmetic
# For example: new_vector = word1 - word2 + word3



new_vector = gregarous - egregious + convivial
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))




# In[5]:


# List the top ten closest vectors in the vocabulary to the result of the expression above
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])


# In[6]:


# CHALLENGE: Write a function that takes in 3 strings, performs a-b+c arithmetic, and returns a top-ten result

def vector_math(a,b,c):
    from scipy import spatial
    cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
    a = nlp.vocab[a].vector
    b = nlp.vocab[b].vector
    c = nlp.vocab[c].vector
    new_vector = a - b + c
    computed_similarities = []

    for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word, similarity))

    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
    top_ten=([w[0].text for w in computed_similarities[:10]])
    return(top_ten)


# In[7]:


# Test the function on known words:
vector_math('king','man','woman')


# ### Task #2: Perform VADER Sentiment Analysis on your own review
# 
# 

# #### Write code that returns a set of SentimentIntensityAnalyzer polarity scores based on your own written review.

# In[8]:


# Import SentimentIntensityAnalyzer and create an sid object
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[9]:


# Write a review as one continuous string (multiple sentences are ok)
review = "This superhero flick is basic but still engrossing, enthralling, and rewarding on many levels, just one of which is seeing Marvel's first female superhero get her own movie."


# In[10]:


# Obtain the sid scores for your review
scoreMetric=sid.polarity_scores(review)
scoreMetric


# #### Note:  Considering Compound score for separating the Postive, Negative or Neutral sentiments. Anything below a score of -0.05 we tag as negative and anything above 0.05 we tag as positive and neutral tag for score between negative and postive score

# In[11]:


# CHALLENGE: Write a function that takes in a review and returns a score of "Positive", "Negative" or "Neutral"

def review_rating(string):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    metrics=sid.polarity_scores(string)
    c_score =metrics['compound']
    if c_score >= 0.5:
        sentiment="Positive"
    elif (c_score> -0.05) & (c_score < 0.05):
        sentiment="Neutral"
    else: 
        sentiment="Negative"
        
    return(sentiment,c_score)




# In[12]:


# Test the function on your review above:
review_rating(review)

