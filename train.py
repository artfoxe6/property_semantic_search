import os
import pandas as pd
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation, SentencesDataset
from torch.utils.data import DataLoader
from datetime import datetime
import logging


def load_triplet_data(csv_path):
    df = pd.read_csv(csv_path)
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[row['query'], row['positive'], row['negative']]))
    return examples


def load_dev_pairs(dev_path):
    df = pd.read_csv(dev_path)
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[row['query'], row['positive']], label=1.0))
    return examples


def build_model(pretrained_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    word_embedding_model = models.Transformer(pretrained_model)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], trust_remote_code=True)

    return model


def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )


def train_model(csv_path='./train_data.csv', output_path='./train_model', batch_size=32, num_epochs=3,
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                dev_path='./train_data_dev.csv'):
    print(f"Training model with {model_name}...")
    # 日志初始化
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, 'train.log')
    setup_logger(log_file)

    model = build_model(pretrained_model=model_name)

    logging.info("Loading training data...")
    train_samples = load_triplet_data(csv_path)
    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    logging.info("Building model...")

    train_loss = losses.TripletLoss(model=model)

    evaluator = None
    if dev_path and os.path.exists(dev_path):
        logging.info("Loading dev set...")
        dev_samples = load_dev_pairs(dev_path)
        dev_queries = [e.texts[0] for e in dev_samples]
        dev_positives = [e.texts[1] for e in dev_samples]
        dev_scores = [1.0 for _ in dev_samples]

        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            dev_queries,
            dev_positives,
            dev_scores,
            name="dev-eval"
        )

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    logging.info(f"Starting training for {num_epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=1000 if evaluator else None,
        output_path=output_path,
        show_progress_bar=True
    )

    logging.info("Training completed.")
    logging.info(f"Model saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='train_data.csv', help='Path to training CSV file')
    parser.add_argument('--dev_path', type=str, default='dev_data.csv', help='Path to dev CSV file (optional)')
    parser.add_argument('--output_path', type=str, default='./output', help='Output model directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    args = parser.parse_args()

    train_model(
        csv_path=args.csv_path,
        dev_path=args.dev_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        model_name=args.model_name
    )
