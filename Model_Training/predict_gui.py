import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
from model_training.fingerprint_net import FingerprintNet
from model_training.identity_predictor import IdentityPredictor
from model_training.embedding_db_builder import EmbeddingDBBuilder
from model_training.imports import transforms
import os

class PredictGUI:
    def __init__(self, master, data_dir):
        self.master = master
        self.master.title("Fingerprint Identity Predictor")

        self.model_path = "fingerprint.pth"
        self.data_dir = data_dir

        self.load_model_and_db()

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(master, text="", wraplength=400)
        self.result_label.pack(pady=10)

    def load_model_and_db(self):
        # Load model
        self.model = FingerprintNet()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # Build embedding DB
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        db_builder = EmbeddingDBBuilder(self.model, transform)
        self.db_embeddings, self.db_labels = db_builder.build_embedding_db(self.data_dir)

        self.transform = transform
        self.predictor = IdentityPredictor(self.model, self.transform)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP Images", "*.BMP"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            matched_file, confidence = self.predictor.predict_identity(file_path, self.db_embeddings, self.db_labels)
            result_text = f"Matched file: {matched_file}\nSimilarity score: {confidence:.4f}"
            self.result_label.config(text=result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_gui.py <data_dir>")
        return
    data_dir = sys.argv[1]

    root = tk.Tk()
    app = PredictGUI(root, data_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
