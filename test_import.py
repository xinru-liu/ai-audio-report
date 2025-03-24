# test_imports.py
print("Testing imports...")

import os
print("os: OK")

import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import whisper
print("whisper: OK")

import transformers
print("transformers version:", transformers.__version__)

import spacy
print("spacy: OK")

import nltk
print("nltk: OK")

print("All imports successful!")
