# Abstractive Text Summarization System

This repository contains the code for a custom-built abstractive summarization system, developed as part of the L90 - Overview of Natural Language Processing practical. The system utilizes a Transformer-based model for generating summaries of news articles.

For a detailed overview of the project, methodology, and results, please see the [project report (l90_report.pdf)](l90_report.pdf).

## Dataset

The model is designed to be trained and evaluated on a subset of the CNN/Daily Mail dataset. The data is expected in JSON format:

```json
{
    "article" : "The article to be summarized.",
    "summary" : "The desired summary."
}
```
(The original dataset also included `"greedy_n_best_indices"` for extractive summarization, which is not the primary focus of this abstractive system.)

**Note on Data Files:** The original large data files (e.g., `data/train.json`, `data/validation.json`, `data/test.json`) are not included in this repository due to GitHub's file size limitations (files exceed 25MB). These files were submitted separately as per the course requirements.
A small example of the input data format can be found in [data/custom_example.json](data/custom_example.json).

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    All required Python packages are listed in [requirements.txt](requirements.txt). Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```
    This includes PyTorch, Transformers, NLTK, ROUGE-metric, and other necessary libraries.

## Usage

The primary scripts for interacting with the abstractive summarization system are [run_abstractive_summarizer.py](run_abstractive_summarizer.py) for training and prediction, and [eval.py](eval.py) for evaluation.

### Abstractive Summarizer ([models/abstractive_summarizer.py](models/abstractive_summarizer.py))

The core of the system is the [`AbstractiveSummarizer`](models/abstractive_summarizer.py) class, which implements a Transformer model ([`Transformer`](models/transformer.py)) for sequence-to-sequence learning.

**1. Training the Model:**
To train the model, you need to provide paths to your training and validation JSON data files.
```sh
python run_abstractive_summarizer.py \
    --train_data path/to/your/train.json \
    --validation_data path/to/your/validation.json
```
-   You can also use `--load_preprocessed_training_data path/to/preprocessed.pt` if you have already preprocessed your training data using the model's internal preprocessing steps.

**2. Generating Summaries (Prediction):**
To generate summaries for new articles using a trained model, provide the model and vocabulary paths, and the evaluation data. The output summaries will be printed to standard output in JSON format.
```sh
python run_abstractive_summarizer.py \
    --load_model_and_vocab path/to/your_model.pt path/to/your_vocab.pt \
    --eval_data path/to/your/eval_articles.json > prediction_file.json
```
-   `path/to/your_model.pt` is the saved PyTorch model state dictionary.
-   `path/to/your_vocab.pt` is the saved vocabulary (if not using BERT tokenizer).
-   `prediction_file.json` will contain the articles and their generated summaries.

### Evaluating Summaries

The [eval.py](eval.py) script evaluates generated summaries against reference summaries using ROUGE scores. It utilizes the [`RougeEvaluator`](evaluation/rouge_evaluator.py) class.
```sh
python eval.py \
    --pred_data path/to/your/prediction_file.json \
    --eval_data path/to/your/reference_summaries.json
```
-   `prediction_file.json` should be the output from the summarizer.
-   `reference_summaries.json` should contain the ground-truth summaries in the same JSON format.

The script will print ROUGE-1, ROUGE-2, and ROUGE-L F1, precision, and recall scores.

### Extractive Summarizer (Scaffolding)

The repository also includes a script `run_extractive_summarizer.py` as a starting point for an extractive summarizer.
```sh
python run_extractive_summarizer.py --eval_data dataset_to_predict_for.json > prediction_file.json
```
The implementation details for `models/extractive_summarizer.py` are not part of this abstractive summarization focused project.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
