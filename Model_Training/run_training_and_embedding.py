from model_training.model_trainer import ModelTrainer
from model_training.embedding_db_builder import EmbeddingDBBuilder

def main(data_dir):
    # Train the model
    trainer = ModelTrainer(data_dir, epochs=15)
    model, transform = trainer.train_model()

    # Build embedding database
    db_builder = EmbeddingDBBuilder(model, transform)
    embeddings, labels = db_builder.build_embedding_db(data_dir)

    print(f"Built embedding database with {len(embeddings)} embeddings.")
    return model, transform, embeddings, labels

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_training_and_embedding.py <data_dir>")
    else:
        data_dir = sys.argv[1]
        main(data_dir)
