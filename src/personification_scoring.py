import numpy as np
import pandas as pd
import json
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# rouge1: N-gram appearance scoring
# rougeL: compute longest common subsequence (LCS)

def evaluate(caption, pred_caption):
    return scorer.score(caption, pred_caption)

def average(list_like):
    return sum(list_like) / len(list_like)


def evaluate_dataset(df):
    # ASSUMPTION:
    # - real caption is `caption` column is the label
    # - predicted caption is `pred_caption` column is the prediction
    result = {
        "trump": df.apply(lambda x: evaluate(x["gen_commentary"], x["trump_caption"]), axis=1),
        "david": df.apply(lambda x: evaluate(x["gen_commentary"], x["david_caption"]), axis=1),
        "tst_trump": df.apply(lambda x: evaluate(x["gen_commentary"], x["tst_trump_text"]), axis=1)
    }

    average_score = {}

    for captioner in result:
        rouge1_precision = []
        rouge1_recall = []
        rouge1_f1 = []

        rougeL_precision = []
        rougeL_recall = []
        rougeL_f1 = []
        
        for scores in result[captioner]:
            rouge1_precision.append(scores["rouge1"].precision)
            rouge1_recall.append(scores["rouge1"].recall)
            rouge1_f1.append(scores["rouge1"].fmeasure)
        
            rougeL_precision.append(scores["rougeL"].precision)
            rougeL_recall.append(scores["rougeL"].recall)
            rougeL_f1.append(scores["rougeL"].fmeasure)
    
        # get average scores
        average_score[captioner] = {
            "rouge1_precision": average(rouge1_precision),
            "rouge1_recall": average(rouge1_recall),
            "rouge1_f1": average(rouge1_f1),

            "rougeL_precision": average(rougeL_precision),
            "rougeL_recall": average(rougeL_recall),
            "rougeL_f1": average(rougeL_f1),
        }
    
    return average_score


def visualize_heatmap(evaluated_result_df):
    plt.figure(figsize=(15, 7))
    sns.heatmap(evaluated_result_df, annot=True)
    plt.savefig("evaluated_result.png")
    plt.close()

if __name__ == "__main__":

    df = pd.read_csv("./data/text/GPT4o/personified_15words_with_tst.csv")

    evaluated_result = evaluate_dataset(df)
    evaluated_result_df = pd.DataFrame(evaluated_result.values())
    evaluated_result_df.index = evaluated_result.keys()
    # print(evaluated_result_df)
    visualize_heatmap(evaluated_result_df)
    evaluated_result_df.to_csv("evaluated_result.csv", index=False)
