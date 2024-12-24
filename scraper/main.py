"""
%load_ext autoreload
%autoreload 2
"""

from pprint import pprint
import numpy as np
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import spacy
import polars as pl
import util
from gensim.models import CoherenceModel, LdaModel
from gensim.utils import simple_preprocess
from IPython.display import display
import pyLDAvis
import pyLDAvis.gensim

# %%
path = "data/ADHD_submissions.zst"
keywords = ["hack", "tip", "advice"]
columns = [
    "id",
    "title",
    "selftext",
    "created_utc",
    "num_comments",
    "score",
    # "link_flair_type",
    # "gilded",
]

# %%
with open(path, "rb") as f:
    stream = util.get_zst_file_json_stream(f)
    submission = next(stream)
    submission["created_utc"]

# %%
with open(path, "rb") as f:
    print(f"Processing file {path}")

    records = []
    for row in util.get_zst_file_json_stream(f):
        row["title"] = row["title"].lower()
        row["selftext"] = row["selftext"].lower()

        if any([word in row["title"] or word in row["selftext"] for word in keywords]):
            records.append({col: row[col] for col in columns})

    data = pl.DataFrame(records)

    print(f"collected {len(data)} submissions")

# %%
df = data.unique("title")
df = df.filter(~pl.col("selftext").str.contains(r"^\[(removed)|(deleted)\]$"))
df = df.head(100)


# %%
def lemmatize(text: str, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return " ".join([x.lemma_ for x in nlp(text) if x.pos_ in allowed_postags])


# %%
df = df.with_columns(
    pl.col("selftext")
    .map_elements(lemmatize, strategy="threading", return_dtype=pl.String)
    .alias("lemmatized_words")
)


# %%
def gen_words(text: str):
    return gensim.utils.simple_preprocess(text, deacc=True)


# %%
df = df.with_columns(
    pl.col("lemmatized_words")
    .map_elements(gen_words, strategy="threading", return_dtype=pl.List(str))
    .alias("gen_words")
)

# %%
id2word = corpora.Dictionary(df["gen_words"])

corpus = []
for text in df["gen_words"]:
    new = id2word.doc2bow(text)
    corpus.append(new)

# %%
model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=30,
    random_state=300,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
)

# %%
vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds="mmds", R=30)
