"""Cora dataset loading and GCN adjacency normalization.

This module is intentionally independent of TensorFlow so that the data
preparation logic (downloading, parsing and adjacency normalization) can be
tested without a TensorFlow installation.
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
CORA_FEATURE_DIM = 1433


def download_cora(data_dir: str | Path = "data/cora") -> Path:
    """Download and extract the Cora dataset if it is not already present.

    Args:
        data_dir: Directory that will contain ``cora.content`` and
            ``cora.cites``.

    Returns:
        The ``data_dir`` path.

    Raises:
        RuntimeError: If the download or extraction fails.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    content_path = data_dir / "cora.content"
    cites_path = data_dir / "cora.cites"
    if content_path.exists() and cites_path.exists():
        return data_dir

    tgz_path = data_dir / "cora.tgz"
    try:
        if not tgz_path.exists():
            print(f"Downloading Cora dataset to {tgz_path} ...")
            urllib.request.urlretrieve(CORA_URL, tgz_path)

        with tarfile.open(tgz_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith("cora.content"):
                    with tar.extractfile(member) as src, open(content_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                elif name.endswith("cora.cites"):
                    with tar.extractfile(member) as src, open(cites_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    except Exception as exc:  # noqa: BLE001 - re-raised with a clearer message.
        raise RuntimeError(
            f"Failed to download/extract Cora. Download it manually from {CORA_URL} "
            f"and place cora.content and cora.cites in {data_dir}."
        ) from exc

    if not content_path.exists() or not cites_path.exists():
        raise RuntimeError(
            f"Cora files are missing in {data_dir}. Download {CORA_URL} manually and "
            "place cora.content and cora.cites there."
        )

    return data_dir


def parse_cora(data_dir: str | Path = "data/cora") -> dict:
    """Parse the raw Cora files into features, labels and an edge index.

    Args:
        data_dir: Directory containing ``cora.content`` and ``cora.cites``.

    Returns:
        A dict with keys ``features``, ``labels``, ``edge_index``,
        ``num_nodes``, ``num_features``, ``num_classes``, ``class_names`` and
        ``num_edges``.
    """
    data_dir = Path(data_dir)
    content_path = data_dir / "cora.content"
    cites_path = data_dir / "cora.cites"

    if not content_path.exists() or not cites_path.exists():
        raise FileNotFoundError(
            f"Cora files not found in {data_dir}. Run download_cora() first."
        )

    paper_ids: list[str] = []
    feature_rows: list[list[float]] = []
    raw_labels: list[str] = []

    with open(content_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < CORA_FEATURE_DIM + 2:
                raise ValueError(
                    f"Malformed cora.content line (expected >= {CORA_FEATURE_DIM + 2} "
                    f"columns, got {len(parts)}): {line[:80]}"
                )

            paper_ids.append(parts[0])
            feature_rows.append([float(v) for v in parts[1 : 1 + CORA_FEATURE_DIM]])
            raw_labels.append(parts[-1])

    num_nodes = len(paper_ids)
    paper_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

    class_names = sorted(set(raw_labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    features = np.asarray(feature_rows, dtype=np.float32)
    labels = np.asarray([class_to_idx[name] for name in raw_labels], dtype=np.int64)

    # Cora citations are directed in the file; treat the graph as undirected.
    edges: list[tuple[int, int]] = []
    with open(cites_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src = paper_to_idx.get(parts[0])
            dst = paper_to_idx.get(parts[1])
            if src is not None and dst is not None:
                edges.append((src, dst))
                edges.append((dst, src))

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T  # shape [2, 2 * |E|]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    num_edges = edge_index.shape[1] // 2

    return {
        "features": features,
        "labels": labels,
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "num_features": CORA_FEATURE_DIM,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_edges": num_edges,
    }


def normalize_adjacency(edge_index: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """Build the symmetric-normalized GCN adjacency matrix.

    Computes ``A_hat = D^{-1/2} (A + I) D^{-1/2}`` where ``A`` is the
    unweighted adjacency matrix (symmetrized if needed), ``I`` adds self-loops
    and ``D`` is the degree matrix of ``A + I``.

    Args:
        edge_index: Int array of shape ``[2, num_edges]``. Directed edges are
            symmetrized automatically.
        num_nodes: Number of nodes in the graph.

    Returns:
        Sparse CSR matrix of shape ``[num_nodes, num_nodes]``.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    num_nodes = int(num_nodes)

    rows = np.concatenate([edge_index[0], np.arange(num_nodes)]).astype(np.int64)
    cols = np.concatenate([edge_index[1], np.arange(num_nodes)]).astype(np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)

    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency = adjacency.tocsr()

    # Symmetrize (binary adjacency) so undirected propagation is guaranteed.
    adjacency = (adjacency + adjacency.T).astype(bool).astype(np.float32).tocsr()

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    degree_matrix_inv_sqrt = sp.diags(degree_inv_sqrt)

    normalized = degree_matrix_inv_sqrt @ adjacency @ degree_matrix_inv_sqrt
    return normalized.tocsr().astype(np.float32)


def _split_masks(
    labels: np.ndarray,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the classic Cora train/val/test masks.

    20 training nodes per class (140 total), 500 validation nodes and 1000
    test nodes are sampled from the remaining nodes with a fixed random seed.
    """
    rng = np.random.RandomState(seed)
    num_nodes = labels.shape[0]

    train_indices: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_nodes = np.where(labels == class_id)[0]
        class_nodes = rng.permutation(class_nodes)
        train_indices.append(class_nodes[:20])

    train_idx = np.concatenate(train_indices) if train_indices else np.asarray([], dtype=np.int64)

    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True

    remaining_idx = rng.permutation(np.where(~train_mask)[0])
    val_idx = remaining_idx[:500]
    test_idx = remaining_idx[500:1500]

    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def load_cora(data_dir: str | Path = "data/cora", seed: int = 42) -> dict:
    """Load Cora and prepare features, labels, normalized adjacency and masks.

    The dataset is downloaded on first use. The returned adjacency matrix is a
    ``scipy.sparse.csr_matrix``; callers can convert it to
    ``tf.sparse.SparseTensor`` for training if desired.

    Args:
        data_dir: Directory containing (or receiving) ``cora.content`` and
            ``cora.cites``.
        seed: Random seed for the reproducible train/val/test split.

    Returns:
        A dict with ``features``, ``labels``, ``adjacency``, ``num_classes``,
        ``num_nodes``, ``num_features``, ``num_edges``, ``class_names`` and
        boolean ``train_mask``/``val_mask``/``test_mask`` arrays.
    """
    download_cora(data_dir)

    parsed = parse_cora(data_dir)
    adjacency = normalize_adjacency(parsed["edge_index"], parsed["num_nodes"])
    train_mask, val_mask, test_mask = _split_masks(
        parsed["labels"], parsed["num_classes"], seed
    )

    return {
        "features": parsed["features"],
        "labels": parsed["labels"],
        "adjacency": adjacency,
        "num_classes": parsed["num_classes"],
        "num_nodes": parsed["num_nodes"],
        "num_features": parsed["num_features"],
        "num_edges": parsed["num_edges"],
        "class_names": parsed["class_names"],
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
