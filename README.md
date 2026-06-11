# Resilient RAGOps

Evaluation framework for a Retrieval-Augmented Generation (RAG) pipeline under different data quality scenarios. The project measures how well RAG-generated answers match gold-standard references using a suite of NLP metrics.

## Project Structure

```
Resilient_RAGOps/
├── metrics.py                  # Metric computation and evaluation script
├── rag_answers_dataset.xlsx    # Dataset with RAG answers across all scenarios
└── Reference Dataset/          # Source reference videos
```

## Dataset

`rag_answers_dataset.xlsx` contains RAG responses and their evaluations across multiple scenarios:

| Sheet | Description |
|---|---|
| Kaggle | Source dataset from Kaggle |
| Original | RAG responses on clean data |
| Evaluation - Original | Metrics for the original scenario |
| Contamination | RAG responses on contaminated data |
| Dirty | RAG responses on dirty/noisy data |
| Evaluation - Dirty | Metrics for the dirty data scenario |
| RollBack | RAG responses after a data rollback |
| Evaluation - RollBack | Metrics for the rollback scenario |

Each evaluation sheet expects the following columns in the input CSV:

| Column | Description |
|---|---|
| `gold_awnser` | Gold-standard reference answer |
| `retrived_awnser_1` | First retrieved answer from the RAG pipeline |
| `retrived_awnser_2` | Second retrieved answer |
| `retrived_awnser_3` | Third retrieved answer |

## Metrics

For each row, metrics are computed individually for all 3 retrieved answers against the gold reference, then averaged:

| Metric | Description |
|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | N-gram and longest common subsequence overlap |
| BLEU | 4-gram precision with brevity penalty |
| METEOR | Alignment-based metric with stemming and synonym matching |
| BERTScore F1 | Semantic similarity using contextual embeddings |

**Models used for BERTScore:**
- English: `microsoft/deberta-xlarge-mnli`
- Other languages: `distilbert-base-multilingual-cased`

## Usage

```python
from metrics import evaluate_rag_csv

evaluate_rag_csv(
    input_csv_path="your_input.csv",
    output_csv_path="your_output.csv",
    language="english"
)
```

The output CSV will contain the original columns plus the averaged metric columns:
`avg_rouge_1`, `avg_rouge_2`, `avg_rouge_l`, `avg_bleu`, `avg_meteor`, `avg_bert_score_f1`

## Dependencies

```
pandas
nltk
rouge-score
bert-score
```

Install with:

```bash
pip install pandas nltk rouge-score bert-score
```

After installing, download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
