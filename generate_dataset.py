import os
import json

with open("./annotations/celeb_anno.json", "r") as f:
    anno = json.load(f)

print(anno["victim_ids"])
