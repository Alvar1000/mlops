import subprocess
import git
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from matching.dataset import VariantPairDataset
from matching.model import PairwiseBinaryClassifier
from matching.utils import normalize_vector


def download_data(cfg: DictConfig) -> None:
    data_dir = Path(cfg.dvc.data_dir)
    print(f"Текущая рабочая директория: {Path.cwd()}")
    print(f"Путь к данным: {data_dir.absolute()}")
    print(f"Существует ли папка data: {data_dir.exists()}")
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "competition.zip"

    if not zip_path.exists():
        url = "https://disk.yandex.ru/d/bSNKwDpGHTTvAg"
        print(f"Загрузите дата-сет с {url} и поместите в {data_dir}")
    else:
        print(f"Data already exists at {zip_path}")


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_dir = Path(cfg.dvc.data_dir)
    result = subprocess.run(["dvc", "pull"], cwd=Path.cwd(), capture_output=True, text=True)
    try:
        import polars as pl

        images_embeds_df = pl.read_parquet(
            data_dir / cfg.data.images_embeds_path.split("/")[-1]
        ).to_pandas()
        texts_embeds_df = (
            pl.read_parquet(data_dir / cfg.data.texts_embeds_path.split("/")[-1])
            .select(["variantid", "name_bert_64"])
            .to_pandas()
        )
        train_df = pl.read_parquet(data_dir / cfg.data.train_path.split("/")[-1]).to_pandas()
    except ImportError:
        images_embeds_df = pd.read_parquet(data_dir / cfg.data.images_embeds_path.split("/")[-1])
        texts_embeds_df = pd.read_parquet(data_dir / cfg.data.texts_embeds_path.split("/")[-1])[
            ["variantid", "name_bert_64"]
        ]
        train_df = pd.read_parquet(data_dir / cfg.data.train_path.split("/")[-1])

    images_embeds_df["main_pic_embeddings_resnet_v1"] = images_embeds_df[
        "main_pic_embeddings_resnet_v1"
    ].apply(lambda x: normalize_vector(x[0]))

    texts_embeds_df["name_bert_64"] = texts_embeds_df["name_bert_64"].apply(normalize_vector)

    embeddings_df = texts_embeds_df.merge(images_embeds_df, on="variantid")
    del images_embeds_df, texts_embeds_df
    train_df, test_df = train_test_split(
        train_df, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    embed_dict = embeddings_df.set_index("variantid").to_dict()

    return train_df, test_df, embed_dict


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module
) -> tuple[list[float], list[float], list[float], float]:
    model.eval()
    all_targets = []
    all_predictions = []
    all_probas = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            text_emb1 = batch["text_emb1"]
            img_emb1 = batch["img_emb1"]
            text_emb2 = batch["text_emb2"]
            img_emb2 = batch["img_emb2"]
            targets = batch["target"]

            outputs = model(text_emb1, img_emb1, text_emb2, img_emb2)
            loss = criterion(outputs, batch["target"].view(-1, 1))
            predictions = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
            all_probas.extend(outputs.squeeze().cpu().numpy().tolist())

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return all_targets, all_predictions, all_probas, avg_loss


def get_git_commit_id() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown"


def train(cfg: DictConfig) -> None:
    """Функция обучения"""
    download_data(cfg)

    # Load data
    train_df, test_df, embed_dict = load_data(cfg)

    # Create datasets
    train_dataset = VariantPairDataset(
        train_df[["variantid1", "variantid2"]].values,
        embed_dict["name_bert_64"],
        embed_dict["main_pic_embeddings_resnet_v1"],
        train_df["target"].values,
    )

    test_dataset = VariantPairDataset(
        test_df[["variantid1", "variantid2"]].values,
        embed_dict["name_bert_64"],
        embed_dict["main_pic_embeddings_resnet_v1"],
        test_df["target"].values,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.trainer.batch_size, shuffle=cfg.trainer.shuffle_train
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.trainer.batch_size, shuffle=cfg.trainer.shuffle_test
    )

    # Create model
    model = hydra.utils.instantiate(cfg.model)
    criterion = nn.BCELoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    commit_id = get_git_commit_id()

    print("\nHyperparameters:")
    print(f"  text_emb_size: {cfg.model.text_emb_size}")
    print(f"  img_emb_size: {cfg.model.img_emb_size}")
    print(f"  hidden_size: {cfg.model.hidden_size}")
    print(f"  nlayers: {cfg.model.nlayers}")
    print(f"  learning_rate: {cfg.optimizer.lr}")
    print(f"  batch_size: {cfg.trainer.batch_size}")
    print(f"  num_epochs: {cfg.trainer.num_epochs}")
    print(f"  git_commit_id: {commit_id}")
    print()

    num_epochs = cfg.trainer.num_epochs
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            outputs = model(
                batch["text_emb1"], batch["img_emb1"], batch["text_emb2"], batch["img_emb2"]
            )
            loss = criterion(outputs, batch["target"].view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        _, _, _, eval_loss = evaluate_model(model, test_dataloader, criterion)
        eval_losses.append(eval_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}"
        )

    real, preds, probas, final_loss = evaluate_model(model, test_dataloader, criterion)

    accuracy = accuracy_score(real, preds)
    precision = precision_score(real, preds)
    recall = recall_score(real, preds)
    prauc_precision, prauc_recall, _ = precision_recall_curve(real, probas)
    prauc = auc(prauc_recall, prauc_precision)
    f1 = f1_score(real, preds)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), eval_losses, label="Eval Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    print("Saved loss_curves.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(prauc_recall, prauc_precision, label=f"PR-AUC = {prauc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("pr_curve.png")
    print("Saved pr_curve.png")
    plt.close()

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": prauc,
    }
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_dict.keys(), metrics_dict.values())
    plt.ylabel("Score")
    plt.title("Final Evaluation Metrics")
    plt.ylim([0, 1])
    plt.grid(True, axis="y")
    plt.savefig("metrics.png")
    print("Saved metrics.png")
    plt.close()

    print(f"\nFinal Metrics:")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"PR-AUC: {prauc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    model_path = "pairwise_binary_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
