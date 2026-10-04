# import os
# import json
# import glob
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA


# DATA_PATH = "./dataset/LLMRouterBench/bench-release"
# OUTPUT_CSV = "./llmrouting_processed/llmrouterbench_features.csv"
# DATASETS = ["gpqa", "mmlupro", "livecodebench", "arenahard"]
# MODEL_NAMES = ["gpt-5", "claude-sonnet-4", "gemini-2.5-pro", "gemini-2.5-flash", "deepseek-v3-0324", "qwen3-235b-a22b-thinking-2507"]
# PCA_DIM = 64

# encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# embedding_cache = {}

# def get_embedding(text):
#     if text not in embedding_cache:
#         embedding_cache[text] = (encoder.encode(text, normalize_embeddings=True).tolist())

#     return embedding_cache[text]


# rows = []
# query_ids = {}
# next_id = 0


# for dataset in DATASETS:
#     dataset_dir = os.path.join(DATA_PATH, dataset)

#     if not os.path.exists(dataset_dir):
#         print("Missing:", dataset_dir)
#         continue

#     files = glob.glob(dataset_dir + "/**/*.json", recursive=True)
#     print(dataset, len(files), "files")

#     for file in tqdm(files):
#         with open(file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         model_name = data.get("model_name", None)
#         if model_name is None:
#             continue

#         if model_name.lower() not in [m.lower() for m in MODEL_NAMES]:
#             continue

#         for record in data["records"]:
#             query = record.get("origin_query")
#             reward = record.get("score")
#             cost = record.get("cost")

#             if (query is None or reward is None or cost is None):
#                 continue

#             if query not in query_ids:
#                 query_ids[query] = next_id
#                 next_id += 1

#             qid = query_ids[query]
#             emb = get_embedding(query)

#             rows.append(
#                 {
#                     "query_id": qid,
#                     "embedding": emb,
#                     "model": model_name,
#                     "reward": float(reward),
#                     "cost": float(cost),
#                     "dataset": dataset
#                 }
#             )


# df = pd.DataFrame(rows)


# models = sorted(df.model.unique())
# model_to_id = {model: i for i, model in enumerate(models)}
# K = len(models)
# print("Models:")

# for i, model in enumerate(models):
#     print(i, model)


# query_embeddings = (df[["query_id", "embedding"]].drop_duplicates("query_id").sort_values("query_id"))
# all_query_embeddings = np.array(query_embeddings["embedding"].tolist(), dtype=np.float32)
# print("Original embedding shape:", all_query_embeddings.shape)

# pca = PCA(n_components=PCA_DIM, random_state=42)
# reduced_embeddings = pca.fit_transform(all_query_embeddings)
# norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
# reduced_embeddings = (reduced_embeddings / np.maximum(norms, 1e-12))
# query_to_embedding = {int(qid): reduced_embeddings[i] for i, qid in enumerate(query_embeddings["query_id"])}

# print("Reduced embedding shape:", reduced_embeddings.shape)
# print("Explained variance:", pca.explained_variance_ratio_.sum())


# def create_feature(query_id, model):
#     x = np.asarray(query_to_embedding[int(query_id)], dtype=np.float32)
#     e = np.zeros(K, dtype=np.float32)
#     e[model_to_id[model]] = 1.0

#     return np.kron(e, x).tolist()


# features = []

# for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
#     phi = create_feature(row.query_id, row.model)
#     features.append(phi)

# df["feature"] = features
# df = df[["query_id", "feature", "model", "reward", "cost", "dataset"]]
# print("Queries:", df.query_id.nunique())
# print("Rows:", len(df))
# print("Models:", df.model.nunique())
# print("PCA dimension:", PCA_DIM)
# print("Feature dimension:", len(df.iloc[0]["feature"]))
# df.to_csv(OUTPUT_CSV, index=False)
# print("Saved:", OUTPUT_CSV)





# separate the features from the csv to make it faster

import ast
import numpy as np
import pandas as pd

INPUT_CSV = "./llmrouting_processed/llmrouterbench_features.csv"
OUTPUT_CSV = "./llmrouting_processed/llmrouterbench_final.csv"
FEATURE_FILE = "./llmrouting_processed/features.npy"

K = 6
df = pd.read_csv(INPUT_CSV)
query_model_count = df.groupby("query_id")["model"].nunique()
valid_queries = query_model_count[query_model_count == K].index
df = df[df["query_id"].isin(valid_queries)].reset_index(drop=True)
features = np.stack([np.asarray(ast.literal_eval(x), dtype=np.float32) for x in df["feature"]], axis=0)
df.to_csv(OUTPUT_CSV, index=False)
np.save(FEATURE_FILE, features)

print("Remaining queries:", df["query_id"].nunique())
print("Remaining rows:", len(df))
print("Feature shape:", features.shape)
print("Saved:", OUTPUT_CSV)
print("Saved:", FEATURE_FILE)