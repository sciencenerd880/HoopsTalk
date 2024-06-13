import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(module_url)


def evaluate_cosine(text1, text2):
    if type(text1) == str:
        text1 = [text1]
        text2 = [text2]
    text1_emb = use_model(text1).numpy()
    text2_emb = use_model(text2).numpy()

    return np.clip(
        tf.reduce_sum(
            tf.multiply(text1_emb, text2_emb),
            axis=1
        ).numpy(),
        .0,
        1.
    )


def evaluate_dataset(df):
    # ASSUMPTION:
    # - real caption is `caption` column is the label
    # - predicted caption is `pred_caption` column is the prediction
    return {
        "cosine": evaluate_cosine(df["caption"], df["pred_caption"])
    }


if __name__ == "__main__":
    text1 = "kids are talking by the door. Dogs sitting on door"
    text2 = "kids talking on door. Dogs are sitting by the door"
    print(evaluate_cosine(text1, text2))
    print(evaluate_cosine("this is ae a test rest pep did", "this is ad test rest pep did"))

    df = pd.DataFrame({
        "caption": ["hello there general kenobi", "foo bar foobar", "this is a dog"],
        "pred_caption": ["hello there! General Kenobi!!", "foobar foo bar", "a dog, it is"]
    })

    print(evaluate_dataset(df))
