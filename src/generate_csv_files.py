import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def load_adult_file(path, is_test=False):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # skip metadata line in adult.test
            if is_test and line.startswith("|"):
                continue

            parts = [x.strip() for x in line.split(",")]

            if len(parts) != 15:
                continue

            # remove trailing dot in test labels
            if is_test:
                parts[-1] = parts[-1].rstrip(".")

            rows.append(parts)

    return pd.DataFrame(rows, columns=columns)

train_df = load_adult_file("adult.data", is_test=False)
test_df = load_adult_file("adult.test", is_test=True)

train_df.to_csv("adult_train.csv", index=False)
test_df.to_csv("adult_test.csv", index=False)

full_df = pd.concat(
    [train_df.assign(split="train"), test_df.assign(split="test")],
    ignore_index=True
)
full_df.to_csv("adult_full.csv", index=False)

print(train_df.head())
print(test_df.head())