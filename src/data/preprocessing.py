"""
Text and Image Preprocessing Pipelines
========================================
Handles cleaning, normalization, augmentation, and tokenization
of raw text and image data before feeding into the model.
"""

import re
import string
import torch
from PIL import Image
from torchvision import transforms


class TextPreprocessor:
    """
    Cleans and normalizes raw text from social media posts.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove @mentions and #hashtags (optional)
        5. Remove excessive whitespace
        6. Remove special characters (optional)
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 3,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length

    def __call__(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = re.sub(r"http\S+|www\.\S+", "", text)

        if self.remove_html:
            text = re.sub(r"<[^>]+>", "", text)

        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)

        if self.remove_special_chars:
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Return empty string if too short
        if len(text.split()) < self.min_length:
            return text  # Keep short text but flag it

        return text


class ImagePreprocessor:
    """
    Handles image loading, validation, and transformation.

    Supports separate train (with augmentation) and eval pipelines.
    Gracefully handles corrupt/missing images by returning a blank tensor.
    """

    def __init__(
        self,
        image_size: int = 224,
        augment: bool = True,
        mean: list = None,
        std: list = None,
    ):
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # Training augmentation pipeline
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(p=0.1),
        ])

        # Evaluation pipeline (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image(self, path: str) -> Image.Image:
        """Load and convert image to RGB, handling errors gracefully."""
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print(f"[WARNING] Failed to load image {path}: {e}")
            # Return a blank RGB image as fallback
            return Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

    def __call__(self, image: Image.Image, train: bool = True) -> torch.Tensor:
        """Apply the appropriate transform pipeline."""
        transform = self.train_transform if train else self.eval_transform
        return transform(image)

    def get_blank_tensor(self) -> torch.Tensor:
        """Return a normalized blank image tensor for missing images."""
        blank = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        return self.eval_transform(blank)
