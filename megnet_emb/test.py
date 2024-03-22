import gzip
import json

with gzip.open("my_models_benchmark.json.gz", "rb") as f:
    a = json.loads(f.read())
print(type(a))
print(a)