from model_training.imports import Image, os, torch
from model_training.fingerprint_net import FingerprintNet
import numpy as np

class EmbeddingDBBuilder:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def build_embedding_db(self, image_folder):
        self.model.eval()
        embeddings, labels = [], []
        paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.BMP')]

        with torch.no_grad():
            for path in paths:
                img = Image.open(path).convert('L')
                img_tensor = self.transform(img).unsqueeze(0).cuda()
                embedding = self.model(img_tensor).cpu().numpy().flatten()
                embeddings.append(embedding)
                labels.append(path)

        return embeddings, labels
