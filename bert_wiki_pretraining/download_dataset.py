import tensorflow_datasets as tfds
import argparse
from tqdm import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("dataset_folder", type=str)
args = parser.parse_args()
dataset_folder = args.dataset_folder

ds = tfds.load('wikipedia/20230601.en', split='train', shuffle_files=False)
with open(os.path.join(dataset_folder, 'tfds_wiki.jsonl'), 'wt') as fout:
    for batch in tqdm(ds):
        out = {'text': batch['text'].numpy().decode("utf-8"), 'title': batch['title'].numpy().decode("utf-8")}
        a = fout.write(json.dumps(out))
        a = fout.write('\n')

