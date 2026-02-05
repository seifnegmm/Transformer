# Next-Word Prediction: Scratch vs Fine-Tuned (Generalization Experiment)

This project compares two approaches to Language Modeling:
1.  **Scratch Model**: A custom Decoder-only Transformer trained from zero.
2.  **Fine-tuned Model**: A pre-trained GPT-2 (124M) adapted to our data.

The experiment evaluates **Generalization** by training on a combined dataset (Reuters Finance + WikiText) and testing on each domain separately.

You can find the new trained model weights here: https://drive.google.com/drive/folders/1zpYNBruT4bdv5QxPyRBuh4WYyIOo9lQb?usp=sharing
---

## Quick Start

### 1. Setup Environment
We use a Python virtual environment to manage dependencies.
```bash
# Create environment (if not exists)
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies (Torch, Transformers, Plotly, etc.)
pip install -r requirements.txt
```

### 2. Run the Full Experiment (One-Click)
We have an automated script that trains both models and runs the dual-domain evaluation.
```bash
bash run_training.sh
```
*   **Duration**: ~10-15 minutes on Apple M4 (MPS). Slower on CPU.
*   **What it does**:
    *   Trains Scratch Model (5 epochs).
    *   Fine-tunes GPT-2 (3 epochs).
    *   Runs `run_comparison.py` to evaluate Perplexity/Accuracy on Finance vs. General text.

### 3. Visualize Results
Open the Jupyter Notebook to generate charts for the report.
```bash
# Start Jupyter
jupyter notebook visualization.ipynb
```
*   **Charts Generated**:
    *   Training Loss Curves (Did it converge?).
    *   Perplexity Bar Chart (Reuters vs WikiText).
    *   Top-K Confidence Heatmap (Does the model predict "be" or "plummet"?).

---

## Manual Execution (Advanced)

If you want to run steps individually:

### Train Scratch Model (Combined Data)
```bash
python3 src/train.py \
    --model_type scratch \
    --dataset reuters,wikitext \
    --epochs 20 \
    --batch_size 16
```

### Fine-tune GPT-2 (Combined Data)
```bash
python3 src/train.py \
    --model_type finetune \
    --dataset reuters,wikitext \
    --epochs 4 \
    --batch_size 8
```

### Run Evaluation
```bash
python3 src/run_comparison.py
```

---
