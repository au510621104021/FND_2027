"""
Multimodal Fake News Dataset
==============================
Unified dataset class supporting multiple benchmark datasets:
    - Weibo (Chinese social media)
    - Twitter MediaEval
    - FakeNewsNet (GossipCop / PolitiFact)
    - Fakeddit (Reddit-based)

Each dataset adapter normalizes the data into a common format:
    (text, image_path, label)  where label: 0=real, 1=fake
"""

import os
import json
import csv
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

from .preprocessing import TextPreprocessor, ImagePreprocessor


# =============================================================================
# Dataset Adapters (normalize different dataset formats to a common schema)
# =============================================================================

class DatasetAdapter:
    """Base class for dataset-specific loading logic."""

    def load(self, data_dir: str) -> list:
        """
        Returns list of dicts: [{'text': str, 'image_path': str, 'label': int}, ...]
        Label: 0 = real, 1 = fake
        """
        raise NotImplementedError


def _normalize_binary_label(value) -> int:
    """Normalize various binary label encodings to {0: real, 1: fake}."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not pd.isna(value):
        return 1 if int(value) == 1 else 0

    text = str(value).strip().lower()
    if text in {"1", "fake", "rumor", "false", "f", "yes", "y", "misleading"}:
        return 1
    if text in {"0", "real", "true", "nonrumor", "non-rumor", "r", "no", "n", "credible"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


class WeiboAdapter(DatasetAdapter):
    """
    Weibo dataset adapter.
    Expected structure:
        data_dir/
            rumor_images/
            nonrumor_images/
            posts/
                rumor.txt
                nonrumor.txt
    """

    def load(self, data_dir: str) -> list:
        samples = []
        for label_name, label_id in [("rumor", 1), ("nonrumor", 0)]:
            text_file = os.path.join(data_dir, "posts", f"{label_name}.txt")
            image_dir = os.path.join(data_dir, f"{label_name}_images")

            if not os.path.exists(text_file):
                print(f"[WARNING] Weibo text file not found: {text_file}")
                continue

            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        post_id = parts[0].strip()
                        text = parts[1].strip() if len(parts) > 1 else ""

                        # Try to find corresponding image
                        image_path = None
                        for ext in [".jpg", ".jpeg", ".png", ".gif"]:
                            candidate = os.path.join(image_dir, post_id + ext)
                            if os.path.exists(candidate):
                                image_path = candidate
                                break

                        if image_path and text:
                            samples.append({
                                "text": text,
                                "image_path": image_path,
                                "label": label_id,
                            })

        print(f"[Weibo] Loaded {len(samples)} samples")
        return samples


class TwitterMediaEvalAdapter(DatasetAdapter):
    """
    Twitter MediaEval dataset adapter.
    Expected structure:
        data_dir/
            posts.csv  (columns: post_id, text, image_id, label)
            images/
    """

    def load(self, data_dir: str) -> list:
        samples = []
        posts_file = os.path.join(data_dir, "posts.csv")

        if not os.path.exists(posts_file):
            print(f"[WARNING] Twitter MediaEval posts file not found: {posts_file}")
            return samples

        df = pd.read_csv(posts_file)
        image_dir = os.path.join(data_dir, "images")

        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            image_id = str(row.get("image_id", row.get("post_id", "")))
            label = int(row.get("label", 0))

            # Find image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(image_dir, image_id + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if image_path and text:
                samples.append({
                    "text": text,
                    "image_path": image_path,
                    "label": label,
                })

        print(f"[Twitter MediaEval] Loaded {len(samples)} samples")
        return samples


class FakeNewsNetAdapter(DatasetAdapter):
    """
    FakeNewsNet dataset adapter (GossipCop / PolitiFact).
    Expected structure:
        data_dir/
            gossipcop/ or politifact/
                real/
                    gossipcop-XXX/
                        news content.json
                        top_img.png
                fake/
                    ...
    """

    def __init__(self, subset: str = "gossipcop"):
        self.subset = subset  # gossipcop or politifact

    def load(self, data_dir: str) -> list:
        samples = []
        subset_dir = os.path.join(data_dir, self.subset)

        if not os.path.exists(subset_dir):
            print(f"[WARNING] FakeNewsNet subset not found: {subset_dir}")
            return samples

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(subset_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            for article_dir in os.listdir(label_dir):
                article_path = os.path.join(label_dir, article_dir)
                if not os.path.isdir(article_path):
                    continue

                # Load text
                text = ""
                json_file = os.path.join(article_path, "news content.json")
                if os.path.exists(json_file):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            text = data.get("text", data.get("title", ""))
                    except Exception:
                        continue

                # Find image
                image_path = None
                for img_name in ["top_img.png", "top_img.jpg", "image.png", "image.jpg"]:
                    candidate = os.path.join(article_path, img_name)
                    if os.path.exists(candidate):
                        image_path = candidate
                        break

                if image_path and text:
                    samples.append({
                        "text": text,
                        "image_path": image_path,
                        "label": label_id,
                    })

        print(f"[FakeNewsNet/{self.subset}] Loaded {len(samples)} samples")
        return samples


class FakedditAdapter(DatasetAdapter):
    """
    Fakeddit dataset adapter (Reddit).
    Expected structure:
        data_dir/
            train.tsv / test.tsv
            images/
    """

    def load(self, data_dir: str) -> list:
        samples = []

        for split in ["train", "test", "validate"]:
            tsv_file = os.path.join(data_dir, f"{split}.tsv")
            if not os.path.exists(tsv_file):
                continue

            df = pd.read_csv(tsv_file, sep="\t")
            image_dir = os.path.join(data_dir, "images")

            for _, row in df.iterrows():
                text = str(row.get("clean_title", row.get("title", "")))
                image_id = str(row.get("id", ""))

                # Binary label (2_way_label or 3_way_label → binarize)
                if "2_way_label" in row:
                    label = int(row["2_way_label"])
                elif "label" in row:
                    label = 0 if int(row["label"]) == 0 else 1
                else:
                    continue

                image_path = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    candidate = os.path.join(image_dir, image_id + ext)
                    if os.path.exists(candidate):
                        image_path = candidate
                        break

                if image_path and text:
                    samples.append({
                        "text": text,
                        "image_path": image_path,
                        "label": label,
                    })

        print(f"[Fakeddit] Loaded {len(samples)} samples")
        return samples


class ISOTAdapter(DatasetAdapter):
    """
    ISOT Fake News dataset adapter.
    Expected structure:
        data_dir/
            Fake.csv
            True.csv

    Output schema:
        text from title + text, image_path=None, label (0=real, 1=fake)
    """

    def load(self, data_dir: str) -> list:
        samples = []
        fake_path = os.path.join(data_dir, "Fake.csv")
        true_path = os.path.join(data_dir, "True.csv")

        if not (os.path.exists(fake_path) and os.path.exists(true_path)):
            print(f"[WARNING] ISOT files not found in {data_dir}. Expected Fake.csv and True.csv")
            return samples

        for csv_path, label in [(fake_path, 1), (true_path, 0)]:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                body = str(row.get("text", "")).strip()
                text = f"{title}. {body}".strip(". ").strip()
                if not text:
                    continue
                samples.append({
                    "text": text,
                    "image_path": None,
                    "label": label,
                })

        print(f"[ISOT] Loaded {len(samples)} samples")
        return samples


class GenericCSVAdapter(DatasetAdapter):
    """
    Generic CSV/TSV adapter for custom datasets.
    Expected columns: text, image_path (or image), label
    """

    def __init__(self, separator: str = ","):
        self.separator = separator

    @staticmethod
    def _pick_first_column(columns: list, candidates: list):
        lowered = {c.lower(): c for c in columns}
        for name in candidates:
            if name in lowered:
                return lowered[name]
        return None

    @staticmethod
    def _compose_text(row, primary_col: str, fallback_col: str = None) -> str:
        primary = str(row.get(primary_col, "")).strip() if primary_col else ""
        fallback = str(row.get(fallback_col, "")).strip() if fallback_col else ""
        if fallback and fallback.lower() != "nan":
            return f"{primary}. {fallback}".strip(". ").strip()
        return primary

    @staticmethod
    def _candidate_files() -> list:
        return [
            "dataset.csv",
            "data.csv",
            "train.csv",
            "test.csv",
            "val.csv",
            "valid.csv",
            "validation.csv",
            "data.tsv",
            "train.tsv",
            "test.tsv",
            "val.tsv",
        ]

    @staticmethod
    def _split_file_map() -> dict:
        return {
            "train": ["train.csv", "train.tsv"],
            "val": ["val.csv", "val.tsv", "valid.csv", "validation.csv"],
            "test": ["test.csv", "test.tsv"],
            "all": ["dataset.csv", "dataset.tsv", "data.csv", "data.tsv"],
        }

    def _load_dataframe(self, filepath: str) -> pd.DataFrame:
        sep = "\t" if filepath.lower().endswith(".tsv") else self.separator
        df = pd.read_csv(filepath, sep=sep)

        # Some datasets are semicolon-separated but saved as .csv.
        if len(df.columns) == 1 and ";" in str(df.columns[0]) and sep == ",":
            try:
                df = pd.read_csv(filepath, sep=";")
            except Exception:
                pass
        return df

    def _parse_dataframe(self, df: pd.DataFrame, data_dir: str) -> list:
        samples = []
        text_col = self._pick_first_column(
            list(df.columns),
            ["text", "content", "statement", "headline", "title", "body"],
        )
        # Optional second text field to concatenate (common in news datasets)
        secondary_text_col = self._pick_first_column(
            list(df.columns),
            ["text", "content", "body", "article"],
        )
        if secondary_text_col == text_col:
            secondary_text_col = None

        image_col = self._pick_first_column(
            list(df.columns),
            ["image_path", "image", "img_path", "img", "image_url"],
        )
        label_col = self._pick_first_column(
            list(df.columns),
            ["label", "target", "class", "fake", "category", "verdict", "truth"],
        )

        if not text_col or not label_col:
            return samples

        for _, row in df.iterrows():
            text = self._compose_text(row, text_col, secondary_text_col)
            if not text or text.lower() == "nan":
                continue

            try:
                label = _normalize_binary_label(row[label_col])
            except Exception:
                continue

            image_path = None
            if image_col and pd.notna(row.get(image_col)):
                img = str(row[image_col])
                if os.path.isabs(img):
                    image_path = img
                else:
                    image_path = os.path.join(data_dir, img)
                if not os.path.exists(image_path):
                    image_path = None

            samples.append({
                "text": text,
                "image_path": image_path,
                "label": label,
            })

        return samples

    def _load_file(self, filepath: str, data_dir: str) -> list:
        df = self._load_dataframe(filepath)
        return self._parse_dataframe(df, data_dir)

    def load_splits(self, data_dir: str) -> dict:
        split_map = {key: [] for key in self._split_file_map()}
        data_root = Path(data_dir)
        found_paths = {}

        if not data_root.exists():
            return split_map

        for split_name, filenames in self._split_file_map().items():
            for filename in filenames:
                direct_path = data_root / filename
                if direct_path.exists():
                    found_paths[split_name] = str(direct_path)
                    break
            if split_name in found_paths:
                continue
            for candidate in data_root.rglob("*"):
                if candidate.is_file() and candidate.name.lower() in filenames:
                    found_paths[split_name] = str(candidate)
                    break

        for split_name, filepath in found_paths.items():
            loaded = self._load_file(filepath, data_dir)
            split_map[split_name].extend(loaded)
            print(f"[GenericCSV] Loaded {len(loaded)} {split_name} samples from {os.path.basename(filepath)}")

        return split_map

    def load(self, data_dir: str) -> list:
        samples = []
        candidate_files = self._candidate_files()

        found_files = []
        for filename in candidate_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                found_files.append(filepath)

        # If files are not at the top level, scan subdirectories (common after unzip).
        if not found_files and os.path.exists(data_dir):
            data_root = Path(data_dir)
            recursive_hits = []
            for p in data_root.rglob("*"):
                if not p.is_file():
                    continue
                name_l = p.name.lower()
                if name_l in candidate_files:
                    recursive_hits.append(str(p))

            if recursive_hits:
                found_files.extend(sorted(recursive_hits))

        if not found_files:
            print(f"[GenericCSV] No compatible CSV/TSV files found under: {data_dir}")
            return samples

        for filepath in found_files:
            loaded = self._load_file(filepath, data_dir)
            if not loaded:
                print(f"[GenericCSV] Skipping {os.path.basename(filepath)}: missing text/label columns")
                continue
            samples.extend(loaded)
            print(f"[GenericCSV] Loaded {len(loaded)} samples from {os.path.basename(filepath)}")

        print(f"[GenericCSV] Total loaded samples: {len(samples)}")

        return samples


# =============================================================================
# Dataset Adapter Registry
# =============================================================================

DATASET_ADAPTERS = {
    "weibo": WeiboAdapter,
    "twitter_mediaeval": TwitterMediaEvalAdapter,
    "gossipcop": lambda: FakeNewsNetAdapter("gossipcop"),
    "politifact": lambda: FakeNewsNetAdapter("politifact"),
    "fakeddit": FakedditAdapter,
    "isot": ISOTAdapter,
    "generic": GenericCSVAdapter,
}
def get_adapter(dataset_name: str) -> DatasetAdapter:
    """Get the appropriate dataset adapter."""
    if dataset_name not in DATASET_ADAPTERS:
        print(f"[WARNING] Unknown dataset '{dataset_name}', falling back to generic CSV adapter")
        return GenericCSVAdapter()
    adapter = DATASET_ADAPTERS[dataset_name]
    return adapter() if callable(adapter) else adapter


def _ensure_list(value, default=None) -> list:
    """Normalize a scalar or sequence config value into a list."""
    if value is None:
        return [] if default is None else [default]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_data_sources(data_dir, dataset_name) -> list[tuple[str, str]]:
    """
    Build a normalized list of (data_dir, dataset_name) pairs.

    Supports:
        - single data_dir + single dataset_name
        - multiple data_dirs + one dataset_name (dataset_name repeated)
        - multiple data_dirs + matching dataset_names
    """
    data_dirs = _ensure_list(data_dir)
    dataset_names = _ensure_list(dataset_name or "generic", default="generic")

    if not data_dirs:
        raise ValueError("At least one data directory must be provided.")

    if len(dataset_names) == 1 and len(data_dirs) > 1:
        dataset_names = dataset_names * len(data_dirs)
    elif len(data_dirs) == 1 and len(dataset_names) > 1:
        data_dirs = data_dirs * len(dataset_names)

    if len(data_dirs) != len(dataset_names):
        raise ValueError(
            "Number of data directories must match number of dataset names, "
            "or provide a single dataset name to reuse across all directories."
        )

    return [(str(curr_dir), str(curr_name)) for curr_dir, curr_name in zip(data_dirs, dataset_names)]


def load_dataset_splits(data_dir, dataset_name="generic") -> tuple[dict, list]:
    """
    Load samples from one or more sources while preserving official split files
    when available.

    Returns:
        split_samples: dict with train/val/test/all sample lists
        source_stats: list of dicts with per-source counts
    """
    split_samples = {"train": [], "val": [], "test": [], "all": []}
    source_stats = []

    for current_dir, current_name in _normalize_data_sources(data_dir, dataset_name):
        adapter = get_adapter(current_name)

        if current_name == "generic" and isinstance(adapter, GenericCSVAdapter):
            current_split_samples = adapter.load_splits(current_dir)
            has_official_split = bool(
                current_split_samples["train"] or current_split_samples["val"] or current_split_samples["test"]
            )

            if has_official_split:
                for split_name in split_samples:
                    split_samples[split_name].extend(current_split_samples.get(split_name, []))
                num_samples = (
                    len(current_split_samples["all"])
                    if current_split_samples["all"]
                    else len(current_split_samples["train"]) + len(current_split_samples["val"]) + len(current_split_samples["test"])
                )
            else:
                loaded_samples = adapter.load(current_dir)
                split_samples["all"].extend(loaded_samples)
                num_samples = len(loaded_samples)
        else:
            loaded_samples = adapter.load(current_dir)
            split_samples["all"].extend(loaded_samples)
            num_samples = len(loaded_samples)

        source_stats.append({
            "dataset_name": current_name,
            "data_dir": current_dir,
            "num_samples": num_samples,
        })

    return split_samples, source_stats


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MultimodalFakeNewsDataset(Dataset):
    """
    Unified PyTorch dataset for multimodal fake news detection.

    Handles text tokenization, image preprocessing, and label encoding
    for any supported benchmark dataset.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "generic",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        image_size: int = 224,
        train: bool = True,
        samples: list = None,
    ):
        self.data_dir = data_dir
        self.train = train
        self.max_length = max_length

        # Text preprocessing & tokenization
        self.text_preprocessor = TextPreprocessor()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Image preprocessing
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)

        # Load samples (or use pre-split samples)
        if samples is not None:
            self.samples = samples
        else:
            adapter = get_adapter(dataset_name)
            self.samples = adapter.load(data_dir)

        if len(self.samples) == 0:
            print("[WARNING] Dataset is empty! Check data_dir and dataset_name.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # --- Text processing ---
        raw_text = sample["text"]
        clean_text = self.text_preprocessor(raw_text)

        encoded = self.tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        # --- Image processing ---
        image_path = sample.get("image_path")
        if image_path and os.path.exists(image_path):
            image = self.image_preprocessor.load_image(image_path)
            pixel_values = self.image_preprocessor(image, train=self.train)
        else:
            pixel_values = self.image_preprocessor.get_blank_tensor()

        # --- Label ---
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "pixel_values": pixel_values,
            "label": label,
            "raw_text": raw_text,
            "image_path": image_path or "",
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_dataloader(
    data_dir,
    dataset_name: str = "generic",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256,
    image_size: int = 224,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> dict:
    """
    Create train/val/test DataLoaders from a dataset.

    Returns:
        dict with 'train', 'val', 'test' DataLoader objects and 'dataset_size' info.
    """
    data_dirs = _ensure_list(data_dir)
    base_data_dir = str(data_dirs[0])
    primary_dataset_name = str(_ensure_list(dataset_name or "generic", default="generic")[0])
    split_samples, source_stats = load_dataset_splits(data_dir, dataset_name=dataset_name)

    has_official_split = bool(split_samples["train"] or split_samples["val"] or split_samples["test"])

    def _build_dataset(samples: list, train_flag: bool):
        return MultimodalFakeNewsDataset(
            data_dir=base_data_dir,
            dataset_name=primary_dataset_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            image_size=image_size,
            train=train_flag,
            samples=samples,
        )

    if has_official_split:
        train_samples = list(split_samples["train"])
        val_samples = list(split_samples["val"])
        test_samples = list(split_samples["test"])

        if len(source_stats) > 1:
            print(f"\n{'='*50}")
            print("Combined Dataset Sources")
            print(f"{'='*50}")
            for idx, source in enumerate(source_stats, start=1):
                print(
                    f"  {idx}. {source['dataset_name']} | "
                    f"{source['num_samples']} samples | {source['data_dir']}"
                )
            print(f"{'='*50}\n")

        if not train_samples and split_samples["all"]:
            train_samples = list(split_samples["all"])

        if not val_samples and len(train_samples) > 1 and val_split > 0:
            derived_val_size = int(len(train_samples) * val_split)
            derived_val_size = max(derived_val_size, 1)
            if len(train_samples) - derived_val_size < 1:
                derived_val_size = max(len(train_samples) - 1, 0)
            if derived_val_size > 0:
                pool_samples = list(train_samples)
                generator = torch.Generator().manual_seed(seed)
                train_subset, val_subset = random_split(
                    pool_samples,
                    [len(pool_samples) - derived_val_size, derived_val_size],
                    generator=generator,
                )
                train_samples = [pool_samples[i] for i in train_subset.indices]
                val_samples = [pool_samples[i] for i in val_subset.indices]

        if not test_samples and len(train_samples) > 2 and test_split > 0:
            derived_test_size = int(len(train_samples) * test_split)
            derived_test_size = max(derived_test_size, 1)
            if len(train_samples) - derived_test_size < 1:
                derived_test_size = max(len(train_samples) - 1, 0)
            if derived_test_size > 0:
                pool_samples = list(train_samples)
                generator = torch.Generator().manual_seed(seed)
                reduced_train_subset, test_subset = random_split(
                    pool_samples,
                    [len(pool_samples) - derived_test_size, derived_test_size],
                    generator=generator,
                )
                train_samples = [pool_samples[i] for i in reduced_train_subset.indices]
                test_samples = [pool_samples[i] for i in test_subset.indices]

        total = len(train_samples) + len(val_samples) + len(test_samples)
        train_size = len(train_samples)
        val_size = len(val_samples)
        test_size = len(test_samples)
        if total == 0:
            raise ValueError(
                f"No samples found for dataset_name='{dataset_name}' in data_dir='{data_dir}'. "
                "For generic datasets, place train/test/dataset CSV files with text and label columns."
            )

        train_dataset = _build_dataset(train_samples, True)
        val_dataset = _build_dataset(val_samples, False)
        test_dataset = _build_dataset(test_samples, False)
    else:
        full_samples = list(split_samples["all"])
        full_dataset = _build_dataset(full_samples, True)
        total = len(full_dataset)
        if total == 0:
            raise ValueError(
                f"No samples found for dataset_name='{dataset_name}' in data_dir='{data_dir}'. "
                "For generic datasets, place train/test/dataset CSV files with text and label columns."
            )

        if len(source_stats) > 1:
            print(f"\n{'='*50}")
            print("Combined Dataset Sources")
            print(f"{'='*50}")
            for idx, source in enumerate(source_stats, start=1):
                print(
                    f"  {idx}. {source['dataset_name']} | "
                    f"{source['num_samples']} samples | {source['data_dir']}"
                )
            print(f"{'='*50}\n")

        test_size = int(total * test_split)
        val_size = int(total * val_split)
        train_size = total - val_size - test_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    if train_size + val_size + test_size == 0:
        raise ValueError(
            f"No samples found for dataset_name='{dataset_name}' in data_dir='{data_dir}'. "
            "For generic datasets, place train/test/dataset CSV files with text and label columns."
        )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    print(f"\n{'='*50}")
    print(f"Dataset Split Summary")
    print(f"{'='*50}")
    print(f"  Total samples : {total}")
    print(f"  Train         : {train_size}")
    print(f"  Validation    : {val_size}")
    print(f"  Test          : {test_size}")
    print(f"{'='*50}\n")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "dataset_size": {
            "total": total,
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "sources": source_stats,
    }


def _collate_fn(batch: list) -> dict:
    """Custom collate function to handle mixed tensor/string fields."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_text": [b["raw_text"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }
