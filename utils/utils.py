import json
import os

def configInfo(file):
    with open(file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config