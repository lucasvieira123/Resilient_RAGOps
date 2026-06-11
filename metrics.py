import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

def calculate_metrics(generated_summary, reference_summary, language="english"):
    """
    Calculates metrics for a pair of texts (generated vs reference).
    """
    metrics = {}

    # Force string conversion and strip whitespace (avoids errors with null values)
    generated_summary = str(generated_summary).strip()
    reference_summary = str(reference_summary).strip()

    if not generated_summary or not reference_summary or generated_summary == "nan" or reference_summary == "nan":
        return {
            'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0,
            'bleu': 0.0, 'meteor': 0.0, 'bert_score_f1': 0.0
        }

    # === ROUGE ===
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = scorer.score(reference_summary, generated_summary)
        metrics['rouge_1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge_2'] = rouge_scores['rouge2'].fmeasure
        metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        metrics.update({'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0})

    # === BLEU ===
    try:
        tokenized_ref = [word_tokenize(reference_summary.lower(), language=language)]
        tokenized_gen = word_tokenize(generated_summary.lower(), language=language)

        if len(tokenized_gen) == 0 or len(tokenized_ref[0]) == 0:
            metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = sentence_bleu(tokenized_ref, tokenized_gen, weights=(0.25, 0.25, 0.25, 0.25))
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        metrics['bleu'] = 0.0

    # === METEOR ===
    try:
        if len(word_tokenize(generated_summary, language=language)) == 0 or len(word_tokenize(reference_summary, language=language)) == 0:
            metrics['meteor'] = 0.0
        else:
            metrics['meteor'] = meteor_score([word_tokenize(reference_summary, language=language)], word_tokenize(generated_summary, language=language))
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        metrics['meteor'] = 0.0

    # === BERTScore ===
    try:
        if language == "english": 
            _model_type = "microsoft/deberta-xlarge-mnli"
        else: 
            _model_type = "distilbert-base-multilingual-cased"

        P, R, F1 = bert_score_calc([generated_summary], [reference_summary], lang=language, model_type=_model_type, verbose=False)
        metrics['bert_score_f1'] = F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}. Retrying with 'multi'...")
        try:
            P, R, F1 = bert_score_calc([generated_summary], [reference_summary], lang='multi', verbose=False)
            metrics['bert_score_f1'] = F1.mean().item()
        except Exception as e2:
            print(f"Fatal error in BERTScore: {e2}")
            metrics['bert_score_f1'] = 0.0

    return metrics

def evaluate_rag_csv(input_csv_path: str, output_csv_path: str, language: str = "english"):
    """
    Reads the CSV dataset, computes metrics, and saves the averages as new columns.
    """
    # Load the CSV
    print(f"Reading file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Lists to store the average results for each row
    avg_rouge_1 = []
    avg_rouge_2 = []
    avg_rouge_l = []
    avg_bleu = []
    avg_meteor = []
    avg_bert_f1 = []

    print("Starting line-by-line metric calculation...")

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Retrieve texts using the exact column names from the dataset
        gold = row['gold_awnser']
        ans1 = row['retrived_awnser_1']
        ans2 = row['retrived_awnser_2']
        ans3 = row['retrived_awnser_3']

        # Calculate individual metrics for the 3 retrieved answers
        m1 = calculate_metrics(ans1, gold, language=language)
        m2 = calculate_metrics(ans2, gold, language=language)
        m3 = calculate_metrics(ans3, gold, language=language)

        # Compute the arithmetic mean of each metric for this row
        avg_rouge_1.append((m1['rouge_1'] + m2['rouge_1'] + m3['rouge_1']) / 3.0)
        avg_rouge_2.append((m1['rouge_2'] + m2['rouge_2'] + m3['rouge_2']) / 3.0)
        avg_rouge_l.append((m1['rouge_l'] + m2['rouge_l'] + m3['rouge_l']) / 3.0)
        avg_bleu.append((m1['bleu'] + m2['bleu'] + m3['bleu']) / 3.0)
        avg_meteor.append((m1['meteor'] + m2['meteor'] + m3['meteor']) / 3.0)
        avg_bert_f1.append((m1['bert_score_f1'] + m2['bert_score_f1'] + m3['bert_score_f1']) / 3.0)

        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} rows...")

    # Add new columns with the averages to the original DataFrame
    df['avg_rouge_1'] = avg_rouge_1
    df['df_avg_rouge_2'] = avg_rouge_2
    df['avg_rouge_l'] = avg_rouge_l
    df['avg_bleu'] = avg_bleu
    df['avg_meteor'] = avg_meteor
    df['avg_bert_score_f1'] = avg_bert_f1

    # Save the final output file
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"\nEvaluation completed successfully! Output file saved at: {output_csv_path}")
    
evaluate_rag_csv("Respostas dataset - pos rollback.csv", "avalRollback.csv", language="english")