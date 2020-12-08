import re
import nltk

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from nltk.stem.snowball import SnowballStemmer
from EX3.Neflix_preprocess import load_files_from_disk

movies_df, _ = load_files_from_disk(False, False)
stemmer = SnowballStemmer("english", ignore_stopwords=False)

def normalize(X):
  normalized = []
  for x in X:
    words = nltk.word_tokenize(x)
    normalized.append(' '.join([stemmer.stem(word) for word in words if re.match('[a-zA-Z]+', word)]))
  return normalized

pipe = Pipeline([
  ('normalize', FunctionTransformer(normalize, validate=False)),
  ('counter_vectorizer', CountVectorizer(
    max_df=0.8, max_features=200000,
    min_df=0.2, stop_words='english',
    ngram_range=(1,3)
  )),
  ('tfidf_transform', TfidfTransformer())
])

tfidf_matrix = pipe.fit_transform([x for x in movies_df['plot']])

similarity_distance = 1 - cosine_similarity(tfidf_matrix)
mergings = linkage(similarity_distance, method='complete')
dendrogram_ = dendrogram(mergings,
               labels=[x for x in movies_df["title"]],
               leaf_rotation=90,
               leaf_font_size=16,
              )

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

plt.show()