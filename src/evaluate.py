import torch
import torch.nn.functional as F
import numpy as np

try:
    import gensim.downloader as api
except ImportError:
    api = None


class Evaluator:
    """
    Comparison Toolkit for Next-Word Prediction Models.

    This class handles:
    1. Calculating Perplexity (Mathematical certainty).
    2. Generating Top-K next-word candidates (Qualitative check).
    3. Auto-regressive text generation (Long-form fluency check).
    4. Semantic Similarity (via GloVe embeddings, if available).
    """

    def __init__(self, model, tokenizer, device="cpu", load_glove=False):
        """
        Initialize the Evaluator.

        Args:
            model: The PyTorch model to evaluate.
            tokenizer: The HuggingFace tokenizer (shared between models).
            device (str): 'cuda', 'mps', or 'cpu'.
            load_glove (bool): Whether to download/load GloVe embeddings for semantic scoring.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.glove = None

        if load_glove:
            if api is None:
                print("Warning: Gensim not found. Semantic metrics will be skipped.")
            else:
                print(
                    "Loading GloVe embeddings for Semantic Similarity (this may take a while)..."
                )
                try:
                    # Load a small 50d model for demonstration/speed
                    self.glove = api.load("glove-wiki-gigaword-50")
                except Exception as e:
                    print(
                        f"Warning: Could not load GloVe: {e}. Semantic metrics will be skipped."
                    )

    def calculate_metrics(self, data_loader, max_batches=None):
        """
        Calculate comprehensive metrics:
        1. Perplexity (Uncertainty)
        2. Accuracy (Exact Match)
        3. Semantic Similarity (Average Cosine Similarity of Top-1 prediction vs Target)

        Args:
            data_loader: DataLoader containing (x, y) pairs.
            max_batches (int, optional): Limit evaluation to first N batches (for speed).
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        correct_predictions = 0
        total_semantic_sim = 0
        semantic_count = 0

        # Import ScratchTransformer here to avoid circular imports at top-level
        from model_scratch import ScratchTransformer

        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if max_batches is not None and i >= max_batches:
                    break

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                if isinstance(self.model, ScratchTransformer):
                    # Our scratch model returns (logits, loss)
                    logits, loss = self.model(x, targets=y)
                else:
                    outputs = self.model(x, labels=x)
                    loss = outputs.loss
                    logits = outputs.logits

                # --- 1. Perplexity ---
                if not isinstance(loss, torch.Tensor):
                    continue
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

                # --- 2. Accuracy & 3. Semantic Similarity ---
                # Get predictions for the next word
                # Note on Shifting:
                # Scratch: Logits are aligned such that logits[:, t] predicts y[:, t]
                # HF: Logits are aligned such that logits[:, t] predicts x[:, t+1] (which corresponds to y[:, t])

                # We need to extract the relevant logits for comparison
                if isinstance(self.model, ScratchTransformer):
                    preds = torch.argmax(logits, dim=-1)  # (B, T)
                    targets = y
                else:
                    # HF logits[:, :-1] predicts x[:, 1:] which is effectively y
                    # We slice to match the shapes
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = x[..., 1:].contiguous()
                    preds = torch.argmax(shift_logits, dim=-1)
                    targets = shift_labels

                # Flatten for calculation
                preds_flat = preds.view(-1)
                targets_flat = targets.view(-1)

                # Accuracy
                correct_predictions += (preds_flat == targets_flat).sum().item()

                # Semantic Similarity
                # Calculating for EVERY token is very slow due to string decoding.
                # We'll calculate it for a maximum of 500 tokens total per evaluation run.
                if self.glove and np.random.rand() < 0.05:
                    for p_idx, t_idx in zip(
                        preds_flat[:50], targets_flat[:50]
                    ):  # Limit to 50 per batch
                        pred_word = self.tokenizer.decode([p_idx])
                        target_word = self.tokenizer.decode([t_idx])
                        sim = self.semantic_similarity(pred_word, target_word)
                        total_semantic_sim += sim
                        semantic_count += 1

        mean_loss = total_loss / total_tokens
        perplexity = np.exp(mean_loss)
        accuracy = correct_predictions / total_tokens
        avg_semantic_sim = (
            (total_semantic_sim / semantic_count) if semantic_count > 0 else 0.0
        )

        return {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "semantic_similarity": avg_semantic_sim,
        }

    def evaluate_next_word(self, prompts, k=5):
        """
        Qualitative and Top-K evaluation on a list of prompts.

        Args:
            prompts (list[str]): List of starting phrases (e.g. ["The stock market", "Oil prices"])
            k (int): Number of top candidates to retrieve.

        Returns:
            list[dict]: List of results containing the prompt, top-k words, and their probabilities.
        """
        self.model.eval()
        results = []
        from model_scratch import ScratchTransformer

        with torch.no_grad():
            for prompt in prompts:
                # Encode
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )

                # Get logits for the LAST token
                if isinstance(self.model, ScratchTransformer):
                    logits, _ = self.model(input_ids)
                else:
                    outputs = self.model(input_ids)
                    logits = outputs.logits

                next_token_logits = logits[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)

                # Get Top-K
                top_k_probs, top_k_indices = torch.topk(probs, k)
                top_k_words = [self.tokenizer.decode([idx]) for idx in top_k_indices]

                results.append(
                    {
                        "prompt": prompt,
                        "top_k_words": top_k_words,
                        "top_k_probs": top_k_probs.cpu().numpy(),
                    }
                )
        return results

    def generate_text(self, prompt, max_new_tokens=20, temperature=1.0):
        """
        Autoregressively generate text from a prompt.

        Args:
            prompt (str): The starting text.
            max_new_tokens (int): How many tokens to generate.
            temperature (float): Controls randomness.
                                 < 1.0 = more deterministic/repetitive.
                                 > 1.0 = more creative/chaotic.
        """
        self.model.eval()
        # Encode initial context
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        from model_scratch import ScratchTransformer

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if it gets too long (for Scratch model with limited context window)
                if input_ids.size(1) > 1000:
                    input_ids = input_ids[:, -1000:]

                # Forward pass
                if isinstance(self.model, ScratchTransformer):
                    logits, _ = self.model(input_ids)
                else:
                    outputs = self.model(input_ids)
                    logits = outputs.logits

                # Get last token logits
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)

                # Sample (or use greedy if temperature -> 0, here we use multinomial sampling)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat((input_ids, next_token), dim=1)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def semantic_similarity(self, predicted_word, target_word):
        """
        Calculate cosine similarity between two words using GloVe embeddings.

        Used to address the 'Ambiguity' of text: Even if the predicted word isn't the
        exact target, is it semantically close? (e.g., 'profit' vs 'revenue').

        Returns:
            float: Similarity score (-1.0 to 1.0). Returns 0.0 if word not in vocabulary.
        """
        if self.glove is None:
            return 0.0

        # Clean words (lowercase, strip spaces, remove GPT-2 special characters)
        pred = predicted_word.strip().lower()
        targ = target_word.strip().lower()

        if pred in self.glove and targ in self.glove:
            return self.glove.similarity(pred, targ)
        return 0.0


if __name__ == "__main__":
    # Test stub
    print("Evaluator module ready.")
