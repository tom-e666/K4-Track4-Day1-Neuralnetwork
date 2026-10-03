import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Giá trị khuyến nghị khởi điểm
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    f1_scores = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds, dim=0)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        one_hot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, one_hot)
    raise ValueError(f"Loss không hỗ trợ: {loss_name}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    num_classes = 7

    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(xb)
        all_preds.append(logits.argmax(dim=-1))

    all_preds = torch.cat(all_preds, dim=0)
    avg_loss = total_loss / len(X)
    acc = float((all_preds == y).float().mean().item())

    y_cpu = y.cpu().numpy()
    pred_cpu = all_preds.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_cpu, pred_cpu):
        cm[t, p] += 1

    macro_f1 = macro_f1_from_confusion(cm)
    return {"loss": float(avg_loss), "acc": acc, "macro_f1": macro_f1}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg.get("seed", 42))
    device = data["X_tr"].device

    model = MLP(
        hidden=cfg["hidden"],
        dropout=cfg.get("dropout", 0.0),
        init=cfg.get("init", "he")
    ).to(device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])], "Số tham số không khớp quy định"

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9)
    )

    precision = cfg.get("precision", "fp32")
    use_scaler = (precision == "fp16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # Loss bước 0 trên val trước khi cập nhật
    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])["loss"]

    # Tập con 50.000 mẫu train cố định để tính train loss ở chế độ eval
    train_eval_X = data["X_tr"][:50000]
    train_eval_y = data["y_tr"][:50000]

    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_val_acc = 0.0
    best_val_f1 = 0.0
    diverged = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=cfg["batch"], shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision in ("fp16", "bf16") and device.type == "cuda":
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if use_scaler:
                scaler.scale(loss).backward()
                if cfg.get("clip_norm") is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                optimizer.step()

            batch_grad_norms.append(gn)

        if diverged:
            print(f"[{cfg['exp_id']}] Cảnh báo: Loss thành NaN/inf tại epoch {epoch}!")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        # Đánh giá cuối epoch ở chế độ eval()
        train_res = evaluate(model, train_eval_X, train_eval_y, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        avg_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_res["acc"]
            best_val_f1 = val_res["macro_f1"]

    peak_mem = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    mean_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "exp_id": cfg["exp_id"],
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_f1),
        "time_per_epoch_s": float(mean_epoch_time),
        "peak_mem_MB": float(peak_mem),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("row_id,pred\n")
        for r_id, p in zip(row_id, preds):
            f.write(f"{r_id},{int(p)}\n")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    model = MLP(hidden=cfg["hidden"], dropout=0.0, init=cfg.get("init", "he")).to(device)
    if "best_state" in result and result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
    else:
        print("best_state không có sẵn trong dict (do load từ JSON), đang huấn luyện lại mô hình tốt nhất...")
        fresh_res = run_experiment(cfg, data)
        model.load_state_dict(fresh_res["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Đã ghi dự đoán eval vào: {pred_path}")
