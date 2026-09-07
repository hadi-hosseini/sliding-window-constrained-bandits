from pathlib import Path
import pandas as pd
import numpy as np
import duckdb


DATA_DIR = Path("/home/hadi/CRLT/dataset/KuaiRand-1K/data")
CATEGORY_FILE = Path("/home/hadi/CRLT/dataset/KuaiRand-1K/kuairand_video_categories.csv")

OUTPUT_DIR = Path("./kuairand_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 8
REWARD = "is_click"
SMOOTHING = 20.0

log_files = sorted(DATA_DIR.glob("log_*_1k.csv"))


required_columns = [
    "user_id",
    "video_id",
    "time_ms",
    "is_click",
]


# Optional columns
optional_columns = [
    "long_view",
    "is_rand",
]

usecols = required_columns + [c for c in optional_columns]


frames = []

for path in log_files:
    temp = pd.read_csv(path, usecols=lambda c: c in usecols)
    temp["source_file"] = path.name
    frames.append(temp)

interactions = pd.concat(frames, ignore_index=True)

summary = pd.DataFrame({
    "metric": ["num_interactions", "num_users", "num_videos", "click_rate"],
    "value": [len(interactions), interactions["user_id"].nunique(), interactions["video_id"].nunique(), interactions["is_click"].mean()]
})

summary.to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    index=False
)

db = duckdb.connect()
db.register("interactions", interactions)

category_columns = db.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM read_csv_auto(
        '{CATEGORY_FILE}',
        header=true
    )
    """
).df()


query = f"""
SELECT
    i.user_id,
    i.video_id,
    i.time_ms,
    i.is_click,

    c.first_level_category_id AS category_id,
    c.first_level_category_name AS category_name

FROM interactions AS i

INNER JOIN read_csv_auto(
    '{CATEGORY_FILE}',
    header=true
) AS c

ON CAST(i.video_id AS BIGINT)
   =
   CAST(c.final_video_id AS BIGINT)

WHERE
    c.first_level_category_id IS NOT NULL
    AND c.first_level_category_name IS NOT NULL
    AND c.first_level_category_name <> 'UNKNOWN'
"""

data = db.execute(query).df()
category_stats = (data.groupby(["category_id", "category_name"], as_index=False).agg(n_interactions=("is_click", "size"),
                                                                                     n_users=("user_id", "nunique"),
                                                                                     clicks=("is_click", "sum"),
                                                                                     click_rate=("is_click", "mean")).sort_values("n_interactions",
                                                                                                                                  ascending=False))


category_stats.to_csv(OUTPUT_DIR / "all_first_level_categories.csv", index=False)

top_categories = (category_stats.head(K).copy())
top_categories["action_id"] = np.arange(K)
top_categories.to_csv(OUTPUT_DIR / "top_categories.csv", index=False)


top_ids = set(top_categories["category_id"])
experiment_data = data[data["category_id"].isin(top_ids)].copy()


action_map = dict(zip(top_categories["category_id"], top_categories["action_id"]))
experiment_data["action_id"] = (experiment_data["category_id"].map(action_map))
global_rates = (experiment_data.groupby("category_id")["is_click"].mean().to_dict())


theta = (experiment_data.groupby(["user_id", "category_id", "category_name", "action_id",], as_index=False)
    .agg(n_interactions=("is_click", "size"), clicks=("is_click", "sum"), raw_click_rate=("is_click", "mean"))
)


theta["global_click_rate"] = (theta["category_id"] .map(global_rates))
theta["theta_hat"] = (theta["clicks"] + SMOOTHING * theta["global_click_rate"]) / (theta["n_interactions"]+ SMOOTHING)


theta = theta.sort_values(["user_id", "action_id"])
theta.to_csv(OUTPUT_DIR / "user_category_preferences.csv", index=False)


user_stats = (theta.groupby("user_id")
    .agg(
        num_categories=("category_id", "nunique"),
        total_interactions=("n_interactions", "sum"),
        min_category_interactions=("n_interactions", "min"),
        max_category_interactions=("n_interactions", "max"),
    )
    .reset_index()
)


user_stats["has_all_K_categories"] = (user_stats["num_categories"] == K)
user_stats.to_csv(OUTPUT_DIR / "user_coverage.csv", index=False)

sequence_data = experiment_data[["user_id", "time_ms", "action_id", "category_id", "category_name", "is_click"]].sort_values(["user_id", "time_ms"])
sequence_data.to_csv(OUTPUT_DIR / "category_interactions.csv", index=False)

valid_users = user_stats[user_stats["has_all_K_categories"]]

if len(valid_users) > 0:
    example_user = valid_users.iloc[0]["user_id"]
    example = theta[theta["user_id"] == example_user].sort_values("action_id")

    print("User:", example_user)
    print(example[["action_id", "category_name", "n_interactions", "raw_click_rate", "theta_hat"]].to_string(index=False))
    theta_star = example["theta_hat"].to_numpy()
    print("\ntheta* =", theta_star)