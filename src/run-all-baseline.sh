#!/bin/bash
/home/dev/src/train.py --max-num-epochs 50 --model neuralbow /home/dev/trained_models ../resources/data/python/final_graphs2/jsonl/train ../resources/data/python/final_graphs2/jsonl/valid ../resources/data/python/final_graphs2/jsonl/test
/home/dev/src/train.py --max-num-epochs 50 --model rnn /home/dev/trained_models ../resources/data/python/final_graphs2/jsonl/train ../resources/data/python/final_graphs2/jsonl/valid ../resources/data/python/final_graphs2/jsonl/test
/home/dev/src/train.py --max-num-epochs 50 --model selfatt /home/dev/trained_models ../resources/data/python/final_graphs2/jsonl/train ../resources/data/python/final_graphs2/jsonl/valid ../resources/data/python/final_graphs2/jsonl/test
/home/dev/src/train.py --max-num-epochs 50 --model 1dcnn /home/dev/trained_models ../resources/data/python/final_graphs2/jsonl/train ../resources/data/python/final_graphs2/jsonl/valid ../resources/data/python/final_graphs2/jsonl/test
/home/dev/src/train.py --max-num-epochs 50 --model convselfatt /home/dev/trained_models ../resources/data/python/final_graphs2/jsonl/train ../resources/data/python/final_graphs2/jsonl/valid ../resources/data/python/final_graphs2/jsonl/test

