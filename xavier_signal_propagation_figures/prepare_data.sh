#!/usr/bin/env bash

DATASET_FOLDER="$HOME/dataset_bert"
mkdir -p "$DATASET_FOLDER"

VOCAB_FILE="$HOME/dataset_bert/vocab.txt"
echo "Please download the BERT base lower cased vocab manually, and place it in $VOCAB_FILE!"

[! -f $VOCAB_FILE] && exit

echo "****** Downloading dataset **********"
python download_dataset.py "$DATASET_FOLDER"

INPUT_FILE="$DATASET_FOLDER/tfds_wiki.jsonl"
OUTPUT_PREFIX="$DATASET_FOLDER/wikipedia-en"

echo "****** Processing  dataset for MLM task **********"
python tools/preprocess_data.py --input "$INPUT_FILE" --output-prefix "$OUTPUT_PREFIX" --vocab vocab.txt --dataset-impl mmap --tokenizer-type BertWordPieceLowerCase --split-sentences --chunk-size 2000 --workers=128

echo "****** Data Preparation Complete Done **********"