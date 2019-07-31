import numpy as np
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the data frame containing information taken from article.csv
df_art = pd.read_csv('article_data.csv')

# extract the text from the content column of the data frame
target_text = df_art.content.iloc[0]
text = []
for indx in range(len(df_art)):
    text.append(df_art.content.iloc[indx]) 


# initialize the count vectorizer
vectorizer = CountVectorizer(stop_words='english')

# get the vocabulary from the target document and get word counts
vectorizer.fit_transform([target_text])

#print(vectorizer.get_feature_names())
# compute the term frequency matrix where each row corresponds to a document
freq_term_matrix = vectorizer.transform(text)

# make the transformer that will compute the tfidf weights
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

# transform the term frequency matrix to tf-idf matrix
# the first row is the feature vector for the target document
# the other rows are the feature vectors for the non-target documents
tf_idf_matrix = tfidf.transform(freq_term_matrix)

#print(np.shape(tf_idf_matrix))

# Compute the cosine similarity between all documents
# Note that the first entry is 1.
# i.e. the target document is 100% similar to itself
# entry j is the similarity between the target document and document j, for j in [1:13]
cos_sim_vec = cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix)


cos_sim_df = pd.DataFrame(cos_sim_vec[0],columns= ['Cos_Sim'])
#print(df_art)
#print(cos_sim_df)


df_final = pd.concat([df_art, cos_sim_df], axis = 1)
df_final = df_final.sort_values(by = ['Cos_Sim'],ascending =False)
print(df_final.head(6))






