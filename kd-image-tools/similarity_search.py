import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from torch.nn import CosineSimilarity
import numpy as np
from tqdm import tqdm
import pandas as pd


class ImageSimilaritySearch:
    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.cosine_similarity = CosineSimilarity(dim=1)
        self.image_features = {}

    def extract_features(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(image)
                features = features.squeeze()
                features = features / features.norm()

            return features
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def index_images(self, image_directory: str) -> None:
        image_files = [
            f
            for f in os.listdir(image_directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]

        print(f"Indexing {len(image_files)} images...")
        for filename in tqdm(image_files):
            image_path = os.path.join(image_directory, filename)
            features = self.extract_features(image_path)
            if features is not None:
                self.image_features[image_path] = features

    def find_similar_images_batch(
        self, image_directory: str, top_images, threshold
    ) -> pd.DataFrame:
        self.index_images(image_directory)

        if len(self.image_features) < 2:
            return pd.DataFrame(
                {"Error": ["Not enough valid images found for comparison"]}
            )

        all_results = []

        print("\nFinding similar images for each image...")
        for query_path in tqdm(self.image_features.keys()):
            query_features = self.image_features[query_path]
            similarities = []

            for path, features in self.image_features.items():
                if path != query_path:
                    similarity = self.cosine_similarity(
                        query_features.unsqueeze(0), features.unsqueeze(0)
                    ).item()
                    similarities.append((path, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_images]

            for rank, (match_path, similarity) in enumerate(top_matches, 1):
                if similarity > threshold:
                    all_results.append(
                        {
                            "Query Image": os.path.basename(query_path),
                            "Similar Image": os.path.basename(match_path),
                            "Similarity Score": f"{similarity:.4f}",
                            "Rank": rank,
                        }
                    )

        df = pd.DataFrame(all_results)
        return df
