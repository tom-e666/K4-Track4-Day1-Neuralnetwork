"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    if not os.path.exists(processed_dir):
        candidates = [
            Path(processed_dir),
            Path(__file__).resolve().parents[2] / "data" / "processed",
            Path(__file__).resolve().parents[1] / "data" / "processed",
            Path.cwd() / "data" / "processed",
        ]
        for candidate in candidates:
            if candidate.exists():
                processed_dir = str(candidate)
                break

    train_path = os.path.join(processed_dir, "train.npz")
    eval_path = os.path.join(processed_dir, "eval.npz")

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)
    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"]

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train_full shape invalid: {X_train_full.shape}"
    assert y_train_full.ndim == 1 and y_train_full.shape[0] == X_train_full.shape[0], f"y_train_full shape invalid: {y_train_full.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval shape invalid: {X_eval.shape}"
    assert y_eval.ndim == 1 and y_eval.shape[0] == X_eval.shape[0], f"y_eval shape invalid: {y_eval.shape}"
    assert len(eval_row_id) == len(X_eval), f"eval_row_id length mismatch: {len(eval_row_id)} vs {len(X_eval)}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    numeric_data = X_tr[:, :N_NUMERIC]
    mean = np.mean(numeric_data, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(numeric_data, axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id, mean, std
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64, device=device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.int64, device=device)
    X_eval_t = torch.as_tensor(X_eval, dtype=torch.float32, device=device)
    y_eval_t = torch.as_tensor(y_eval, dtype=torch.int64, device=device)

    classes, counts = np.unique(y_val, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    majority_acc = float(np.mean(y_val == majority_class))

    print(f"Data prepared on {device}:")
    print(f"  Train: X={X_tr_t.shape}, y={y_tr_t.shape}")
    print(f"  Val:   X={X_val_t.shape}, y={y_val_t.shape}")
    print(f"  Eval:  X={X_eval_t.shape}, y={y_eval_t.shape}")
    print(f"  Val majority class {majority_class} baseline accuracy: {majority_acc:.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    n_samples = len(X)
    if shuffle:
        perm = torch.randperm(n_samples, generator=generator, device=X.device)
    else:
        perm = torch.arange(n_samples, device=X.device)

    for i in range(0, n_samples, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
