from model_training.imports import transforms, DataLoader, torch
from model_training.socofing_dataset import SOCOFingDataset
from model_training.fingerprint_net import FingerprintNet
from model_training.triplet_loss import TripletLoss

class ModelTrainer:
    def __init__(self, data_dir, model_save_path="fingerprint_model.pth", epochs=10):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.epochs = epochs

    def train_model(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = SOCOFingDataset(self.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = FingerprintNet().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = TripletLoss()

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for anchor, positive, negative in dataloader:
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                loss = criterion(anchor_out, positive_out, negative_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), self.model_save_path)
        return model, transform
