from model_training.imports import os, random, Dataset, Image

class SOCOFingDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        # Modified to ensure only valid image paths are considered
        self.image_paths = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.endswith('.BMP') and os.path.isfile(os.path.join(image_folder, fname))
        ]
        # Check if image_paths is empty and handle it
        if not self.image_paths:
            raise ValueError(f"No .BMP image files found in the directory: {image_folder}")

        self.label_map = self._build_label_map()

    def _build_label_map(self):
        labels = set()
        for path in self.image_paths:
            person_id = int(os.path.basename(path).split('__')[0])
            labels.add(person_id)
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = int(os.path.basename(anchor_path).split('__')[0])
        anchor_image = Image.open(anchor_path).convert('L')

        positive_path = random.choice([p for p in self.image_paths if f"{anchor_label}__" in p and p != anchor_path])
        negative_path = random.choice([p for p in self.image_paths if f"{anchor_label}__" not in p])

        positive_image = Image.open(positive_path).convert('L')
        negative_image = Image.open(negative_path).convert('L')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
