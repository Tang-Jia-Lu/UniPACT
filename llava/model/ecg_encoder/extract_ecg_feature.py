"""
ECG Feature Extraction Module

Provides the ECG encoder (M3AE wrapper) and a thin dataset that loads
12-lead ECG signals from MIMIC-IV-ECG WFDB records.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import wfdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cmelt import M3AEModel


class ECGEncoder(nn.Module):
    """
    ECG Encoder class that inherits from nn.Module

    This class loads a pre-trained M3AE model and provides methods to extract
    features from ECG signals.
    """

    def __init__(self):
        """Initialize ECG Encoder. Builds the architecture from the bundled
        config; pretrained weights are expected to be loaded externally
        (training/inference drivers pass `--ecg_encoder_dir` and call
        `load_state_dict` themselves)."""
        super().__init__()

        self.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs", "config0.json",
        )

        self.model = None
        self.pooler = None
        self.proj = None
        self.class_embedding = None

        self._build_model()

    def _build_model(self):
        """Construct M3AE submodules from config; weights stay random until the
        caller loads a checkpoint into `ecg_tower`."""
        if self.model is not None:
            return

        with open(self.config_path, "r") as json_file:
            cfg = json.load(json_file)
        cfg = SimpleNamespace(**cfg["model"])

        model = M3AEModel(cfg)

        self.model = model.ecg_encoder
        self.pooler = model.unimodal_ecg_pooler
        self.proj = model.multi_modal_ecg_proj
        self.class_embedding = model.class_embedding


    def forward(self, ecgs):
        """
        Forward pass through the ECG encoder

        Args:
            ecgs: ECG tensor data

        Returns:
            Extracted ECG features
        """
        return self.extract_features(ecgs)

    def extract_features(self, ecgs, datasets="ptbxl"):
        """
        Extract features from ECG signals

        Args:
            ecgs: Input ECG tensor
            datasets: Dataset type (default: "ptbxl")

        Returns:
            ECG features tensor
        """
        ecg_batch = ecgs

        if ecg_batch.shape[1] == 5000:
            ecg_batch = ecg_batch.permute(0, 2, 1)

        with torch.no_grad():
            uni_modal_ecg_feats, ecg_padding_mask = (
                self.model.get_embeddings(ecg_batch, padding_mask=None)
            )

            cls_emb = self.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
            uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)

            uni_modal_ecg_feats = self.model.get_output(uni_modal_ecg_feats, ecg_padding_mask)
            out = self.proj(uni_modal_ecg_feats)
            ecg_features = self.pooler(out)

        return ecg_features

    def extract_features_batch(self, ecgs, batch_size=16, datasets="ptbxl"):
        """
        Extract features from ECG signals in batches

        Args:
            ecgs: List of ECG arrays
            batch_size: Size of each batch (default: 16)
            datasets: Dataset type (default: "ptbxl")

        Returns:
            Concatenated numpy array of features
        """
        features = []
        num_ecgs = len(ecgs)

        for start_idx in range(0, num_ecgs, batch_size):
            end_idx = min(start_idx + batch_size, num_ecgs)
            ecg_batch = torch.tensor(
                ecgs[start_idx:end_idx],
                dtype=torch.float32,
                device=next(self.parameters()).device
            )

            if datasets != "ptbxl":
                ecg_batch = ecg_batch.permute(0, 2, 1)

            with torch.no_grad():
                uni_modal_ecg_feats, ecg_padding_mask = (
                    self.model.get_embeddings(ecg_batch, padding_mask=None)
                )

                cls_emb = self.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
                uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)

                uni_modal_ecg_feats = self.model.get_output(uni_modal_ecg_feats, ecg_padding_mask)
                out = self.proj(uni_modal_ecg_feats)
                ecg_features = self.pooler(out)
                features.append(ecg_features.cpu().numpy())

        return np.concatenate(features, axis=0)


class ECGDataset:
    """Load 12-lead ECG signals from a local copy of MIMIC-IV-ECG.

    The training annotations reference each sample by a path of the form
    ``.../<patient_id>/<study_id>/...`` (e.g.
    ``.../p10000032/s40689238/40689238``); we resolve those to the matching
    PhysioNet WFDB record under `mimic_iv_ecg_root` and read the signal
    with `wfdb.rdsamp`.

    Expected layout under `mimic_iv_ecg_root` (PhysioNet ``mimic-iv-ecg``):

        <root>/<patient_prefix>/<patient_id>/<study_id>/<study_id_no_s>.{dat,hea}

    where ``patient_prefix`` is the first three characters of ``patient_id``
    (e.g. ``p10`` for ``p10000032``). Adjust the prefix slicing below if your
    download uses a different grouping.
    """

    def __init__(self, mimic_iv_ecg_root):
        if not mimic_iv_ecg_root or not os.path.isdir(mimic_iv_ecg_root):
            raise FileNotFoundError(
                f"MIMIC-IV-ECG root not found: {mimic_iv_ecg_root!r}"
            )
        self.root = mimic_iv_ecg_root

    def load_ecg_tensor(self, file_path):
        """Resolve `file_path` to a WFDB record under `self.root` and return
        the 12 x 5000 signal as a float32 tensor."""
        path_parts = file_path.strip("/").split("/")
        if len(path_parts) < 4:
            raise ValueError(f"Invalid path format: {file_path!r}")
        patient_id = path_parts[-3]
        study_id = path_parts[-2]

        record_path = os.path.join(
            self.root,
            patient_id[:3],
            patient_id,
            study_id,
            study_id[1:],
        )
        signals, _ = wfdb.rdsamp(record_path)
        return torch.tensor(signals.T, dtype=torch.float32)


if __name__ == "__main__":
    ecg_tower = ECGEncoder()
    ecg_tower.to(dtype=torch.float32, device='cuda:0')

    ecg_tensor = torch.randn(1, 12, 5000, device='cuda:0', dtype=torch.float32)

    features = ecg_tower.extract_features(ecg_tensor)
    print("Extracted features shape:", features.shape)
