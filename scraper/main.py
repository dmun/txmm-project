"""
%load_ext autoreload
%autoreload 2
"""

import time
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
import spacy
import polars as pl
from spacy.tokens import Doc
import util
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import random
from pprint import pprint

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
def get_submissions(path: str):
  with open(path, "rb") as f:
    print(f"Processing file {path}")

    row_it = util.get_zst_file_json_stream(f)
    schema = {
      "archived": pl.Boolean,
      "id": pl.Utf8,
      "url": pl.Utf8,
      "title": pl.Utf8,
      "author": pl.Utf8,
      "selftext": pl.Utf8,
      "score": pl.Int32,
      "created_utc": pl.Int64,
      "hidden": pl.Boolean,
      "link_flair_richtext": pl.List(pl.Utf8),
      # "link_flair_text": pl.Null,
      "link_flair_type": pl.Utf8,
      "locked": pl.Boolean,
      "num_comments": pl.Int32,
      "num_crossposts": pl.Int32,
      "over_18": pl.Boolean,
      "retrieved_on": pl.Int64,
      "send_replies": pl.Boolean,
      "spoiler": pl.Boolean,
      "stickied": pl.Boolean,
    }
    df = pl.LazyFrame(row_it, schema=schema, strict=False, infer_schema_length=1000)

    df = df.filter(
      (pl.col("title").str.contains(r"\?"))
      & (pl.col("selftext").str.contains("|".join(keywords)))
    ).select(columns)

    # print(f"collected {len(df.collect())} submissions")
    return df


df_submissions = get_submissions("data/ADHD_submissions.zst")

# %%
ids = df_submissions.select("id").collect()["id"].to_list()

# %%
df_submissions.select("title").collect().sample()["title"][0]


# %%
def get_comments(path: str):
  with open(path, "rb") as f:
    print(f"Processing file {path}")

    row_it = util.get_zst_file_json_stream(f)
    schema = {
      "archived": pl.Boolean,
      "link_id": pl.Utf8,
      "parent_id": pl.Utf8,
      "id": pl.Utf8,
      "url": pl.Utf8,
      "title": pl.Utf8,
      "author": pl.Utf8,
      "body": pl.Utf8,
      "score": pl.Int32,
      "created_utc": pl.Int64,
      "hidden": pl.Boolean,
      "link_flair_richtext": pl.List(pl.Utf8),
      "link_flair_type": pl.Utf8,
      "locked": pl.Boolean,
      "num_comments": pl.Int32,
      "num_crossposts": pl.Int32,
      "over_18": pl.Boolean,
      "retrieved_on": pl.Int64,
      "send_replies": pl.Boolean,
      "spoiler": pl.Boolean,
      "stickied": pl.Boolean,
    }

    return pl.LazyFrame(row_it, schema=schema, strict=False)


df = get_comments("data/ADHD_comments.zst")

# %%
df.describe()

# %%
comments = (
  df.filter(
    pl.col("link_id")
    .map_elements(
      lambda x: x.removeprefix("t3_"),
      return_dtype=pl.Utf8,
      strategy="threading",
    )
    .is_in(ids)
    & pl.col("link_id").eq(pl.col("parent_id"))
    & pl.col("score").gt(50)
  )
  .select("body")
  .head(10000)
  .collect()["body"]
  .to_list()
)
len(comments)


# %%
r = random.randint(0, len(comments))
comments[r]


# %%
def preprocess(doc: Doc):
  allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]
  items = set()
  for word in doc:
    if word.pos_ not in allowed_postags:
      continue
    if word.text in stopwords.words():
      continue
    items.add(word.lemma_)
  return items


# %%
def process_with_spacy_pipe(texts, batch_size=2000, n_process=1):
  nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
  results = []

  # Process in batches using spaCy's pipe
  for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
    processed = preprocess(doc)
    results.append(processed)

  return results


# %%
start = time.time()
res = process_with_spacy_pipe(comments, batch_size=2000, n_process=4)
print(f"took {time.time() - start} seconds")


# %%
def gen_words(texts):
  final = []
  for text in texts:
    new = gensim.utils.simple_preprocess(" ".join(text), deacc=True)
    final.append(new)
  return final


words = gen_words(res)

# %%
bigram_phrases = gensim.models.Phrases(words, min_count=5, threshold=50)
trigram_phrases = gensim.models.Phrases(bigram_phrases[words], threshold=50)

# %%
bigrams = gensim.models.phrases.Phrases(bigram_phrases.vocab)
trigrams = gensim.models.phrases.Phrases(trigram_phrases.vocab)


def create_bigrams(texts):
  return [bigrams[doc] for doc in texts]


def create_trigrams(texts):
  return [trigrams[doc] for doc in texts]


data_bigrams = create_bigrams(words)
data_trigrams = create_trigrams(words)

# %%
id2word = corpora.Dictionary(data_trigrams)

corpus = []
# for text in words:
for text in data_trigrams:
  new = id2word.doc2bow(text)
  corpus.append(new)

# %%
tfidf = gensim.models.TfidfModel(corpus, id2word=id2word)

low_value = 0.1
high_value = 0.8  # Adjust this value based on your needs

tfwords = []
tfwords_missing = []
new_corpus = []

for i in range(len(corpus)):
  bow = corpus[i]
  low_value_words = []
  high_value_words = []
  for id, value in tfidf[bow]:
    if value < low_value:
      low_value_words.append(id)
    elif value > high_value:
      high_value_words.append(id)
    else:
      tfwords.append((id, value))
  tfwords_missing.append(low_value_words + high_value_words)
  new_bow = [
    b for b in bow if b[0] not in low_value_words and b[0] not in high_value_words
  ]
  new_corpus.append(new_bow)


# %%
model = LdaModel(
  corpus=new_corpus,
  id2word=id2word,
  num_topics=30,  # Adjusted number of topics
  random_state=300,
  update_every=1,
  chunksize=100,  # Increased chunksize
  passes=20,  # Increased number of passes
  alpha="symmetric",  # Changed alpha
)

# %%
vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds="mmds", R=30)

# %%
print("keep" in stopwords.words())
