from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import nltk
from nltk.corpus import reuters
from datasets import load_dataset

# Ensure NLTK data is downloaded (for Reuters)
try:
    nltk.data.find("corpora/reuters")
except LookupError:
    nltk.download("reuters")
    nltk.download("punkt")


class TextDataset(Dataset):
    """
    Unified Dataset class for Next-Word Prediction.
    Supports:
    1. 'reuters': Financial news (NLTK)
    2. 'wikitext': Wikipedia articles (HuggingFace)
    """

    def __init__(
        self, split="train", tokenizer=None, max_length=128, dataset_name="reuters"
    ):
        """
        Args:
            split (str): 'train' or 'test'.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            max_length (int): Block size for training.
            dataset_name (str or list): 'reuters', 'wikitext', or list ['reuters', 'wikitext'].
        """
        self.tokenizer = (
            tokenizer if tokenizer else GPT2Tokenizer.from_pretrained("gpt2")
        )
        # GPT-2 does not have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

        # Handle single string or list of datasets
        if isinstance(dataset_name, str):
            if "," in dataset_name:
                self.dataset_names = dataset_name.split(",")
            else:
                self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name

        self.inputs = []
        self.targets = []

        print(f"Loading datasets: {self.dataset_names} ({split} split)...")

        all_raw_texts = []
        for name in self.dataset_names:
            all_raw_texts.extend(self._load_raw_texts(name.strip(), split))

        self._process_texts(all_raw_texts, split)

    def _load_raw_texts(self, dataset_name, split):
        """Load raw strings based on dataset name."""
        texts = []

        if dataset_name == "reuters":
            # Reuters Financial News (HuggingFace: danidanou/Reuters_Financial_News)
            # Replaces the small NLTK corpus with a 100k+ doc dataset
            print(
                f"Loading Reuters Financial News (danidanou/Reuters_Financial_News)..."
            )
            try:
                # Load the dataset
                # This dataset usually only has a 'train' split, so we simulate a test split
                hf_dataset = load_dataset(
                    "danidanou/Reuters_Financial_News", split="train"
                )

                # Manual 90/10 split
                total_size = len(hf_dataset)
                split_idx = int(0.9 * total_size)

                if split == "train":
                    # First 90%
                    data_slice = hf_dataset.select(range(split_idx))
                else:
                    # Last 10%
                    data_slice = hf_dataset.select(range(split_idx, total_size))

                # Extract 'Article' column and filter
                for item in data_slice:
                    article = item.get("Article", "").strip()
                    # Filter out empty or very short articles
                    if len(article) > 100:
                        texts.append(article)

                print(
                    f"Loaded {len(texts)} documents for Reuters/{split} (from {total_size} total)."
                )

            except Exception as e:
                print(f"Error loading HuggingFace Reuters: {e}")
                print("Fallback: Using NLTK Reuters (Small)...")
                # Fallback to NLTK if HF fails or no internet
                prefix = "training/" if split == "train" else "test/"
                file_ids = [fid for fid in reuters.fileids() if fid.startswith(prefix)]
                for fid in file_ids:
                    texts.append(reuters.raw(fid))
                print(f"Loaded {len(texts)} documents from NLTK Reuters.")

        elif dataset_name == "wikitext":
            # WikiText-103 (Raw) - SIGNIFICANTLY LARGER than WikiText-2
            # split mapping: 'train' -> 'train', 'test' -> 'test'
            config_name = "wikitext-103-raw-v1"
            print(f"Loading {config_name}...")

            # Use correct namespace 'Salesforce/wikitext' instead of 'wikitext'
            hf_dataset = load_dataset("Salesforce/wikitext", config_name, split=split)

            # WikiText is a list of lines/paragraphs
            # Filter out empty lines/headers
            for item in hf_dataset:
                text = item["text"].strip()
                if len(text) > 50:  # Only keep meaningful chunks
                    texts.append(text)

            # BALANCING: Subsample WikiText to match Reuters size (~100k docs)
            # Otherwise the model will be 90% Wikipedia and only 10% Finance.
            if len(texts) > 100000 and split == "train":
                print(
                    f"Subsampling WikiText from {len(texts)} to 100,000 to balance with Reuters..."
                )
                texts = texts[:100000]

            print(f"Found {len(texts)} documents for {dataset_name}/{split}.")

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return texts

    def _process_texts(self, raw_texts, split="unknown"):
        """Tokenize and create sliding windows."""
        # FULL SCALE: No artificial limit on document count.
        # We rely on the machine's RAM (Colab High-RAM) to handle 100k+ docs.
        print(f"Processing {len(raw_texts)} texts for {split}...")

        for text in raw_texts:
            # Cleaning: remove excessive whitespace/newlines
            clean_text = " ".join(text.split())

            # Tokenize
            # Truncate large documents to avoid OOM during processing
            tokenized = self.tokenizer(
                clean_text, truncation=True, max_length=1024, return_tensors="pt"
            )["input_ids"][0]

            # Sliding Window
            # We want: Input=[A, B, ...], Target=[B, C, ...]
            # Total sequence length needed: max_length + 1
            # Stride: How much we move the window (e.g. 64 for overlap)
            stride = self.max_length

            for i in range(0, len(tokenized) - self.max_length, stride):
                # chunk: length max_length + 1
                chunk = tokenized[i : i + self.max_length + 1]

                if len(chunk) == self.max_length + 1:
                    x = chunk[:-1]  # 0 to 127
                    y = chunk[1:]  # 1 to 128
                    self.inputs.append(x)
                    self.targets.append(y)

        print(
            f"Generated {len(self.inputs)} sequence samples for {self.dataset_names}/{split}."
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_dataloader(split="train", batch_size=16, dataset_name="reuters"):
    dataset = TextDataset(split=split, dataset_name=dataset_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


if __name__ == "__main__":
    # Sanity check
    train_loader = get_dataloader(split="train", batch_size=4)
    x, y = next(iter(train_loader))
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    print("Sample Input:", train_loader.dataset.tokenizer.decode(x[0]))
    print("Sample Target:", train_loader.dataset.tokenizer.decode(y[0]))
