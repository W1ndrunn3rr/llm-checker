import pandas as pd
from openai import OpenAI
import time


labels_count = {0: 0, 1: 0}


def get_embedding(client, text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=text, model=model)

    return response.data[0].embedding


def process_data():
    embedding_rows = []

    df = pd.read_csv("../data/data.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    client: OpenAI = OpenAI()

    correct_rows = 0

    for _, row in df.iterrows():
        try:
            if labels_count[row["generated"]] > 5_000:
                continue

            embedding = get_embedding(client, row["text"])

            embedding_rows.append(
                {
                    "text": row["text"],
                    **{f"dim_{i}": val for i, val in enumerate(embedding)},
                    "label": row["generated"],
                }
            )

            correct_rows += 1

            if correct_rows > 10_000:
                break
            if correct_rows % 100 == 0:
                print(f"Total progress: {round((correct_rows/12_000) * 100, 2)}%")
            labels_count[row["generated"]] += 1

            time.sleep(0.001)

        except Exception as e:
            print(f"Error {str(e)}")
            continue

    result_df = pd.DataFrame(embedding_rows)
    result_df.to_csv("embeddings.csv", index=False)

    print("Data saved!")


if __name__ == "__main__":
    process_data()
