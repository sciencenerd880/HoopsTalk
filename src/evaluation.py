import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure you have downloaded the necessary NLTK data files
nltk.download('punkt')

def bleu(df, description, commentary_tsf, commentary_wo_tsf):
    # Initialize BLEU score calculation
    smoothie = SmoothingFunction().method4
    tsf_bleu_scores = []
    wo_tsf_bleu_scores = []

    for desc, comm_tsf, comm_wo_tsf in zip(description, commentary_tsf, commentary_wo_tsf):
        reference_tokens = [nltk.word_tokenize(desc)]
        comm_tsf_tokens = nltk.word_tokenize(comm_tsf)
        comm_wo_tsf_tokens = nltk.word_tokenize(comm_wo_tsf)
        tsf_score = sentence_bleu(reference_tokens, comm_tsf_tokens, smoothing_function=smoothie)
        wo_tsf_score = sentence_bleu(reference_tokens, comm_wo_tsf_tokens, smoothing_function=smoothie)
        tsf_bleu_scores.append(tsf_score)
        wo_tsf_bleu_scores.append(wo_tsf_score)

    # Add BLEU scores to the DataFrame
    df['BLEU_Score_tsf'] = tsf_bleu_scores
    df['BLEU_Score_notsf'] = wo_tsf_bleu_scores

    # Calculate average BLEU scores
    avg_bleu_tsf = sum(tsf_bleu_scores) / len(tsf_bleu_scores)
    avg_bleu_wo_tsf = sum(wo_tsf_bleu_scores) / len(wo_tsf_bleu_scores)

    print(f"Average BLEU Score with tsf: {avg_bleu_tsf:.4f}")
    print(f"Average BLEU Score without tsf: {avg_bleu_wo_tsf:.4f}")
    
    return df

def rouge(df, description, commentary_tsf, commentary_wo_tsf):
    # Initialize ROUGE score calculation
    # ROUGE-1: Measures unigram overlap, evaluating the presence of individual words.
    # ROUGE-2: Measures bigram overlap, evaluating the presence of pairs of consecutive words.
    # ROUGE-L: Measures the longest common subsequence, evaluating the longest matching sequence of words in order.
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    tsf_rouge_scores = {'rouge1': [], 'rougeL': []}
    wo_tsf_rouge_scores = {'rouge1': [], 'rougeL': []}

    for desc, comm_tsf, comm_wo_tsf in zip(description, commentary_tsf, commentary_wo_tsf):
        tsf_scores = scorer.score(desc, comm_tsf)
        wo_tsf_scores = scorer.score(desc, comm_wo_tsf)

        for key in tsf_scores:
            tsf_rouge_scores[key].append(tsf_scores[key].fmeasure)
            wo_tsf_rouge_scores[key].append(wo_tsf_scores[key].fmeasure)

    # Add ROUGE scores to the DataFrame
    for key in tsf_rouge_scores:
        df[f'ROUGE_{key.upper()}_tsf'] = tsf_rouge_scores[key]
        df[f'ROUGE_{key.upper()}_notsf'] = wo_tsf_rouge_scores[key]

    # Calculate average ROUGE scores
    for key in tsf_rouge_scores:
        avg_tsf_rouge = sum(tsf_rouge_scores[key]) / len(tsf_rouge_scores[key])
        avg_wo_tsf_rouge = sum(wo_tsf_rouge_scores[key]) / len(wo_tsf_rouge_scores[key])
        print(f"Average ROUGE-{key.upper()} Score with tsf: {avg_tsf_rouge:.4f}")
        print(f"Average ROUGE-{key.upper()} Score without tsf: {avg_wo_tsf_rouge:.4f}")

    return df

# Load the CSV file
file_path = './data/text/GPT4o/0021800013-dal-vs-phx_commentary_results.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Assuming the columns are named 'description', 'actionType', 'gen_commentary_with_tsf', and 'gen_commentary_without_tsf'
description = df['description'].astype(str).tolist()
commentary_tsf = df['gen_commentary_with_tsf'].astype(str).tolist()
commentary_wo_tsf = df['gen_commentary_without_tsf'].astype(str).tolist()

df = bleu(df, description, commentary_tsf, commentary_wo_tsf)

df = rouge(df, description, commentary_tsf, commentary_wo_tsf)

# Save the DataFrame with BLEU scores to a new Excel file
output_file_path = './data/evaluation/0021800013-dal-vs-phx_commentary_evaluation.xlsx'
df.to_excel(output_file_path, index=False)

print("BLEU scores calculated and saved to:", output_file_path)