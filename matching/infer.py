import os
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from matching.dataset import VariantPairDataset
from matching.model import PairwiseBinaryClassifier
from matching.utils import normalize_vector


def load_data_for_inference(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    data_dir = Path(cfg.dvc.data_dir)

    try:
        result = subprocess.run(["dvc", "pull"], cwd=Path.cwd(), capture_output=True, text=True)
        if result.returncode == 0:
            print("Успешно")
        else:
            print(f"Ошибка: {result.stderr}")
    except Exception as e:
        print(f"Ошибка: {e}.")
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
    except ImportError:
        images_embeds_df = pd.read_parquet(data_dir / cfg.data.images_embeds_path.split("/")[-1])
        texts_embeds_df = pd.read_parquet(data_dir / cfg.data.texts_embeds_path.split("/")[-1])[
            ["variantid", "name_bert_64"]
        ]

    images_embeds_df["main_pic_embeddings_resnet_v1"] = images_embeds_df[
        "main_pic_embeddings_resnet_v1"
    ].apply(lambda x: normalize_vector(x[0]))

    texts_embeds_df["name_bert_64"] = texts_embeds_df["name_bert_64"].apply(normalize_vector)

    embeddings_df = texts_embeds_df.merge(images_embeds_df, on="variantid")
    del images_embeds_df, texts_embeds_df

    embed_dict = embeddings_df.set_index("variantid").to_dict()
    inference_df = pl.read_parquet(data_dir / cfg.data.inference_path.split("/")[-1]).to_pandas()

    return inference_df, embed_dict


@hydra.main(version_base=None, config_path="../config", config_name="config")
def infer(cfg: DictConfig) -> None:
    inference_df, embed_dict = load_data_for_inference(cfg)

    inference_dataset = VariantPairDataset(
        inference_df[["variantid1", "variantid2"]].values,
        embed_dict["name_bert_64"],
        embed_dict["main_pic_embeddings_resnet_v1"],
        inference_df["target"].values
        if "target" in inference_df.columns
        else np.zeros(len(inference_df)),
    )

    inference_dataloader = DataLoader(
        inference_dataset, batch_size=cfg.trainer.batch_size, shuffle=False
    )

    model = hydra.utils.instantiate(cfg.model)
    model_path = getattr(cfg, "model_path", "pairwise_binary_classifier.pth")
    model.eval()

    all_predictions = []
    all_probas = []

    with torch.no_grad():
        for batch in inference_dataloader:
            outputs = model(
                batch["text_emb1"], batch["img_emb1"], batch["text_emb2"], batch["img_emb2"]
            )
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
            all_probas.extend(outputs.squeeze().cpu().numpy().tolist())

    inference_df["prediction"] = all_predictions
    inference_df["probability"] = all_probas

    output_path = getattr(cfg, "output_path", "predictions.csv")
    inference_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    infer()
