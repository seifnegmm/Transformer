import argparse
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
import os
import json
from tqdm import tqdm

from dataset import get_dataloader
from model_scratch import ScratchTransformer
from evaluate import Evaluator


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    save_dir=None,
    epoch=0,
    checkpoint_steps=5000,
):
    """
    Standard PyTorch training loop for one epoch.

    Args:
        model: Either ScratchTransformer or GPT2LMHeadModel.
        loader: DataLoader provided by dataset.py.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler (optional, used for fine-tuning).
        device: 'cuda', 'mps', or 'cpu'.
        save_dir: Directory to save checkpoints.
        epoch: Current epoch number.
        checkpoint_steps: Save model every N steps.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    step = 0

    for x, y in progress_bar:
        step += 1
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Handle difference in forward pass signature
        if isinstance(model, ScratchTransformer):
            _, loss = model(x, targets=y)
        else:
            # HuggingFace model returns a NamedTuple output
            # CRITICAL FIX: HF models shift labels internally!
            # If we pass 'y' (which is already shifted), the model shifts it AGAIN.
            # This causes a mismatch (predicting token t+2 from token t).
            # So for HF models, we pass 'x' as both input and labels.
            outputs = model(x, labels=x)
            loss = outputs.loss

        loss.backward()

        # Gradient Clipping: Prevents exploding gradients, crucial for Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # CHECKPOINTING logic
        if save_dir and checkpoint_steps > 0 and step % checkpoint_steps == 0:
            ckpt_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            # print(f"Saved checkpoint to {ckpt_path}") # Optional: reduce clutter

    return total_loss / len(loader)


def get_default_device():
    """
    Detects the best available hardware accelerator.
    Returns:
        'cuda': NVIDIA GPUs (Colab/Linux/Windows)
        'mps': Apple Silicon GPUs (M1/M2/M3/M4)
        'cpu': Fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    """
    Main entry point for training.

    CLI Arguments:
        --model_type: 'scratch' (Train NanoGPT from zero) or 'finetune' (Adapt GPT-2).
        --epochs: Number of passes over the dataset.
        --batch_size: Samples per batch (lower this if you run out of memory).
        --lr: Learning rate (usually lower for fine-tuning).
    """
    parser = argparse.ArgumentParser(description="Train Next-Word Prediction Model")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["scratch", "finetune"],
        help="Choose 'scratch' for custom Transformer or 'finetune' for GPT-2",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reuters",
        help="Dataset to train on (e.g. 'reuters', 'wikitext', or 'reuters,wikitext')",
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, default=5000, help="Save model every N steps"
    )
    parser.add_argument(
        "--val_steps", type=int, default=1000, help="Limit validation to N batches"
    )

    # Architecture Arguments (for Scratch Model)
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of Transformer blocks"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of Attention heads"
    )

    args = parser.parse_args()

    # Set default LR based on model type if not provided
    if args.lr is None:
        if args.model_type == "finetune":
            args.lr = (
                5e-5  # Lower LR for fine-tuning to prevent catastrophic forgetting
            )
        else:
            args.lr = 5e-4  # Higher LR for training from scratch

    print(f"Initializing {args.model_type} training on {args.device}...")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data Loading
    train_loader = get_dataloader(
        split="train", batch_size=args.batch_size, dataset_name=args.dataset
    )
    test_loader = get_dataloader(
        split="test", batch_size=args.batch_size, dataset_name=args.dataset
    )
    tokenizer = train_loader.dataset.tokenizer
    vocab_size = tokenizer.vocab_size

    # Model Initialization
    if args.model_type == "scratch":
        print(
            f"Creating Scratch Transformer: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}"
        )
        model = ScratchTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    else:
        print("Loading Pre-trained GPT-2...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.resize_token_embeddings(len(tokenizer))  # In case of special tokens

    model.to(args.device)

    # Optimizer
    # Add weight decay for regularization (important for Scratch model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = None
    if args.model_type == "finetune":
        # Linear warmup for fine-tuning
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=total_steps
        )

    # Training Loop
    evaluator = Evaluator(model, tokenizer, device=args.device, load_glove=True)
    history = {
        "train_loss": [],
        "val_perplexity": [],
        "val_accuracy": [],
        "val_semantic_sim": [],
    }

    # Initialize Early Stopping
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.device,
            save_dir=args.save_dir,
            epoch=epoch,
            checkpoint_steps=args.checkpoint_steps,
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
        history["train_loss"].append(float(avg_loss))

        # Evaluate Perplexity
        # Use val_steps to limit evaluation time
        metrics = evaluator.calculate_metrics(test_loader, max_batches=args.val_steps)
        print(
            f"Epoch {epoch + 1} - Validation Perplexity: {metrics['perplexity']:.2f} | Accuracy: {metrics['accuracy']:.2%} | Semantic Sim: {metrics['semantic_similarity']:.4f}"
        )
        history["val_perplexity"].append(float(metrics["perplexity"]))
        history["val_accuracy"].append(float(metrics["accuracy"]))
        history["val_semantic_sim"].append(float(metrics["semantic_similarity"]))

        # Check Early Stopping
        early_stopper(metrics["perplexity"])
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

        # Qualitative Check
        prompts = ["The company said", "Oil prices"]
        samples = evaluator.evaluate_next_word(prompts)
        for s in samples:
            print(f"Prompt: '{s['prompt']}' -> Top-1: '{s['top_k_words'][0]}'")

    # Save Model
    save_path = os.path.join(
        args.save_dir, f"{args.model_type}_{args.d_model}_{args.dataset}_model.pth"
    )
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save History
    history_path = os.path.join(
        args.save_dir, f"{args.model_type}_{args.d_model}_{args.dataset}_history.json"
    )

    # Convert numpy/tensor values to python floats for JSON serialization
    serializable_history = {}
    for k, v in history.items():
        serializable_history[k] = [float(x) for x in v]

    with open(history_path, "w") as f:
        json.dump(serializable_history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
