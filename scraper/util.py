import traceback
from typing import BinaryIO, Iterator, TypedDict

import orjson as json
import zstandard

class Comment(TypedDict):
  author: str
  created_utc: str
  controversiality: int
  score: int
  subreddit_id: str
  link_id: str
  score_hidden: bool
  parent_id: str
  subreddit: str
  ups: int
  distinguished: None
  body: str
  id: str
  name: str
  downs: int
  archived: bool
  author_flair_css_class: None
  retrieved_on: int
  edited: bool
  author_flair_text: None
  gilded: int


class Submission(TypedDict):
  archived: bool
  author: str
  author_flair_background_color: str
  author_flair_css_class: None
  author_flair_text: None
  author_flair_text_color: str
  brand_safe: bool
  can_gild: bool
  contest_mode: bool
  created_utc: int
  distinguished: None
  domain: str
  edited: bool
  gilded: int
  hidden: bool
  hide_score: bool
  id: str
  is_crosspostable: bool
  is_reddit_media_domain: bool
  is_self: bool
  is_video: bool
  link_flair_css_class: None
  link_flair_richtext: list
  link_flair_text: None
  link_flair_text_color: str
  link_flair_type: str
  locked: bool
  media: None
  media_embed: list
  no_follow: bool
  num_comments: int
  num_crossposts: int
  over_18: bool
  parent_whitelist_status: str
  permalink: str
  retrieved_on: int
  rte_mode: str
  score: int
  secure_media: None
  secure_media_embed: list
  selftext: str
  send_replies: bool
  spoiler: bool
  stickied: bool
  subreddit: str
  subreddit_id: str
  subreddit_name_prefixed: str
  subreddit_type: str
  suggested_sort: None
  thumbnail: str
  thumbnail_height: None
  thumbnail_width: None
  title: str
  url: str
  whitelist_status: str


def get_zst_file_json_stream(
  f: BinaryIO, chunk_size=1024 * 1024 * 10
) -> Iterator:
  decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)
  current_string = ""

  def yieldLinesJson():
    nonlocal current_string
    lines = current_string.split("\n")
    current_string = lines[-1]
    for line in lines[:-1]:
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        print("Error parsing line: " + line)
        traceback.print_exc()
        continue

  zst_reader = decompressor.stream_reader(f)
  while True:
    try:
      chunk = zst_reader.read(chunk_size)
    except zstandard.ZstdError:
      print("Error reading zst chunk")
      traceback.print_exc()
      break
    if not chunk:
      break
    current_string += chunk.decode("utf-8", "replace")

    yield from yieldLinesJson()

  yield from yieldLinesJson()

  if len(current_string) > 0:
    try:
      yield json.loads(current_string)
    except json.JSONDecodeError:
      print("Error parsing line: " + current_string)
      print(traceback.format_exc())
      pass
