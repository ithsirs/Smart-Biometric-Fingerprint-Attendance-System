from model_training.imports import Image, np, torch
from model_training.imports import cosine_similarity

class IdentityPredictor:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def predict_identity(self, query_image_path, db_embeddings, db_labels):
        self.model.eval()
        img = Image.open(query_image_path).convert('L')
        img_tensor = self.transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            query_embedding = self.model(img_tensor).cpu().numpy().flatten()

        sims = cosine_similarity([query_embedding], db_embeddings)[0]
        best_match_idx = np.argmax(sims)
        return db_labels[best_match_idx], sims[best_match_idx]
