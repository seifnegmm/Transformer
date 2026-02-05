import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from dataset import get_dataloader
from model_scratch import ScratchTransformer
from evaluate import Evaluator
import os


def load_scratch_model(path, device, tokenizer, d_model=256, num_layers=6, num_heads=4):
    """
    Load the custom 'From-Scratch' Transformer.

    IMPORTANT: The architecture arguments (d_model, num_layers, etc.)
    MUST match exactly what was used in training.
    """
    vocab_size = tokenizer.vocab_size
    model = ScratchTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_finetune_model(path, device):
    """
    Load the Fine-tuned GPT-2 model.
    Uses HuggingFace's built-in loading mechanism.
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def get_default_device():
    """Detects best available hardware (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    """
    Main Execution Script for Comparative Analysis.

    Workflow:
    1. Detect hardware.
    2. Load the test dataset (Reuters 'test' split).
    3. Load both trained models (Scratch vs Finetune).
    4. Run Quantitative Eval (Perplexity).
    5. Run Qualitative Eval (Next-Word Prediction & Text Generation).
    """
    device = get_default_device()
    print(f"Running comparison on {device}...")

    # 1. Setup Data - Combined and Separate
    print("Loading test datasets...")
    reuters_loader = get_dataloader(split="test", batch_size=8, dataset_name="reuters")
    wiki_loader = get_dataloader(split="test", batch_size=8, dataset_name="wikitext")
    tokenizer = reuters_loader.dataset.tokenizer

    # 2. Load Models
    models = {}

    # 1. Scratch Small (256)
    path_256 = "./models/scratch_256_reuters,wikitext_model.pth"
    if os.path.exists(path_256):
        print(f"Loading Scratch-256 (Small) from {path_256}...")
        models["Scratch-256"] = load_scratch_model(path_256, device, tokenizer, d_model=256, num_layers=6, num_heads=4)

    # 2. Scratch Medium (512)
    path_512 = "./models/scratch_512_reuters,wikitext_model.pth"
    if os.path.exists(path_512):
        print(f"Loading Scratch-512 (Medium) from {path_512}...")
        models["Scratch-512"] = load_scratch_model(path_512, device, tokenizer, d_model=512, num_layers=8, num_heads=8)

    # 3. Fine-tuned
    finetune_path = "./models/finetune_reuters,wikitext_model.pth"
    if os.path.exists(finetune_path):
        print(f"Loading Fine-tuned Model from {finetune_path}...")
        models["Fine-tuned"] = load_finetune_model(finetune_path, device)

    if not models:
        print("No models loaded!")
        return

    # 3. Quantitative Evaluation (Perplexity) - Separated by Domain
    print("\n--- Quantitative Evaluation (Perplexity & Accuracy) ---")

    test_sets = {"Reuters (Finance)": reuters_loader, "WikiText (General)": wiki_loader}

    # Store results for visualization
    comparison_results = {}

    for name, model in models.items():
        print(f"\nEvaluating Model: {name}")
        evaluator = Evaluator(model, tokenizer, device=device, load_glove=True)

        comparison_results[name] = {}

        for domain, loader in test_sets.items():
            print(f"  Domain: {domain}")
            metrics = evaluator.calculate_metrics(loader)
            print(f"Perplexity: {metrics['perplexity']:.2f} | Accuracy: {metrics['accuracy']:.2%} | Semantic Sim: {metrics['semantic_similarity']:.4f}")
            # Store metrics (convert to float for JSON)
            comparison_results[name][domain] = {
                "perplexity": float(metrics["perplexity"]),
                "accuracy": float(metrics["accuracy"]),
                "semantic_sim": float(metrics["semantic_similarity"]),
            }

    # Save results to JSON
    import json

    results_path = "./models/comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\nComparison results saved to {results_path}")

    # 4. Qualitative Evaluation (Prompts) - Separated by Domain
    print("\n--- Qualitative Evaluation (Generation) ---")

    finance_prompts = [
        "Apple shares rose after the company reported earnings that beat expectations. Analysts called the results",
        "Oil prices rose on Monday due to tensions in the Middle East. Market sentiment was",
    ]

    wiki_prompts = [
        "The history of the Roman Empire is characterized by",
        "Quantum mechanics is a fundamental theory in physics that describes",
    ]

    all_prompts = [("Finance", p) for p in finance_prompts] + [("Wiki", p) for p in wiki_prompts]

    for name, model in models.items():
        print(f"\nModel: {name}")
        evaluator = Evaluator(model, tokenizer, device=device)

        # 1. Next-Word Stats
        print("  [Next-Word Stats]")
        results = evaluator.evaluate_next_word([p for _, p in all_prompts], k=5)

        for (domain, _), res in zip(all_prompts, results):
            print(f"[{domain}] Prompt: '{res['prompt']}'")
            print(f"Top-1: {res['top_k_words'][0]} ({res['top_k_probs'][0]:.2f})")
            print(f"Top-5: {res['top_k_words']}")

        # 2. Full Text Generation
        print("\n  [Text Generation (100 tokens)]")
        for domain, prompt in all_prompts:
            generated = evaluator.generate_text(prompt, max_new_tokens=100, temperature=0.4)
            print(f"[{domain}] Prompt: '{prompt}'")
            print(f"Output: ...{generated[len(prompt) :]}")
            print("-" * 20)


if __name__ == "__main__":
    main()
