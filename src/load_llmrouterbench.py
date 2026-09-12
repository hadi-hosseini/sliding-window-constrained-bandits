import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


DATA_PATH = ("/home/hadi/CRLT/dataset/LLMRouterBench/bench-release")
OUTPUT_CSV = ("/home/hadi/CRLT/llmrouting_processed/llmrouterbench.csv")
DATASETS = ["gpqa", "mmlupro", "livecodebench", "arenahard"]
MODEL_KEYWORDS = ["gpt", "claude", "gemini", "deepseek", "qwen3"]

encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embedding_cache = {}


def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = (encoder.encode(text, normalize_embeddings=True).tolist())
    return embedding_cache[text]


rows = []
query_ids = {}
next_id = 0


for dataset in DATASETS:
    dataset_dir = os.path.join(DATA_PATH, dataset)

    if not os.path.exists(dataset_dir):
        print("Missing:", dataset_dir)
        continue

    files = glob.glob(dataset_dir + "/**/*.json", recursive=True)
    print(dataset, len(files), "files")


    for file in tqdm(files):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model_name", None)

        if model_name is None:
            continue

        if not any(k in model_name.lower() for k in MODEL_KEYWORDS):
            continue

        for record in data["records"]:
            query = record.get("origin_query")
            reward = record.get("score")
            cost = record.get("cost")

            if (query is None or reward is None or cost is None):
                continue

            if query not in query_ids:
                query_ids[query] = next_id
                next_id += 1

            qid = query_ids[query]
            emb = get_embedding(query)

            rows.append({
                    "query_id": qid,
                    "embedding": emb,
                    "model": model_name,
                    "reward": float(reward),
                    "cost": float(cost),
                    "dataset": dataset
                    })

df = pd.DataFrame(rows)
print("Queries:", df.query_id.nunique())
print("Rows:", len(df))
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)