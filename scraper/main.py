"""
%load_ext autoreload
%autoreload 2
"""

import polars as pl
from IPython.display import display
import util

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

    df = pl.DataFrame(records)

    print(f"collected {len(df)} submissions")

# %%
df = df.unique("title")

# %%
df.sort("score", descending=True).head(10)
