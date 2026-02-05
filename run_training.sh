#!/bin/bash
# Training Launch Script for Exercise 3.2.3
# Run this script to train both models sequentially

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Adjust paths to relative execution from exercise_3 folder
# Assuming user is running from exercise_3/ or root
if [ -d "exercise_3" ]; then
    cd exercise_3
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Exercise 3.2.3: Next-Word Prediction${NC}"
echo -e "${BLUE}Training Pipeline (Combined Dataset)${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if venv exists
if [ -d "../venv" ]; then
    echo -e "${GREEN}[1/4] Activating virtual environment...${NC}"
    source ../venv/bin/activate
    echo -e "✓ Virtual environment activated\n"
elif [ -d "venv" ]; then
    echo -e "${GREEN}[1/4] Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "✓ Virtual environment activated\n"
else
    echo -e "${RED}Warning: Virtual environment not found. Using system python.${NC}\n"
fi

# Train Scratch Model Small (Combined)
echo -e "${GREEN}[2/5] Training Scratch Model (Small: d=256)...${NC}"
python3 src/train.py \
    --model_type scratch \
    --dataset reuters,wikitext \
    --d_model 256 \
    --num_layers 6 \
    --num_heads 4 \
    --epochs 20 \
    --batch_size 32 \
    --val_steps 1000 \
    --checkpoint_steps 5000

if [ $? -eq 0 ]; then
    echo -e "✓ Scratch-Small model training complete\n"
else
    echo -e "${RED}✗ Scratch-Small model training failed${NC}"
    exit 1
fi

# Train Scratch Model Medium (Combined)
echo -e "${GREEN}[3/5] Training Scratch Model (Medium: d=512)...${NC}"
python3 src/train.py \
    --model_type scratch \
    --dataset reuters,wikitext \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --epochs 8 \
    --batch_size 32 \
    --val_steps 1000 \
    --checkpoint_steps 5000

if [ $? -eq 0 ]; then
    echo -e "✓ Scratch-Medium model training complete\n"
else
    echo -e "${RED}✗ Scratch-Medium model training failed${NC}"
    exit 1
fi

# Train Fine-tuned GPT-2 (Combined)
echo -e "${GREEN}[4/5] Fine-tuning GPT-2 (Full Scale)...${NC}"
python3 src/train.py \
    --model_type finetune \
    --dataset reuters,wikitext \
    --epochs 3 \
    --batch_size 16 \
    --val_steps 1000 \
    --checkpoint_steps 5000

if [ $? -eq 0 ]; then
    echo -e "✓ GPT-2 fine-tuning complete\n"
else
    echo -e "${RED}✗ GPT-2 fine-tuning failed${NC}"
    exit 1
fi

# Run Comparison
echo -e "${GREEN}[5/5] Running model comparison...${NC}"
echo -e "Evaluating both models on Reuters (Finance) vs WikiText (General)..."
python3 src/run_comparison.py

if [ $? -eq 0 ]; then
    echo -e "✓ Comparison complete\n"
else
    echo -e "${RED}✗ Comparison failed${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All training and evaluation complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\nNext Steps:"
echo -e "1. Open 'visualization.ipynb' in VS Code."
echo -e "2. Copy the final Perplexity numbers from above into the notebook."
echo -e "3. Run the notebook to generate your report charts.\n"
