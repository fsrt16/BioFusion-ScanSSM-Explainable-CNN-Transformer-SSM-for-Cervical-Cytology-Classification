import os
import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# CHANGES FROM FCI → WSI
# ------------------------------------------------------------------------------
# 1. CLASS_NAMES updated to match WSI folder names:
#       im_Dyskeratotic, im_Koilocytotic, im_Metaplastic,
#       im_Parabasal, im_Superficial-Intermediate
# 2. load_flat_dataset now scans RECURSIVELY to handle the nested
#       im_X / im_X / *.bmp folder layout, and explicitly includes .bmp.
# 3. Augmentation (build_transforms) made more aggressive:
#       - Larger random crop scale (cluster images are larger & more complex)
#       - Extra augmentations: RandomPerspective, GaussianBlur, RandomGrayscale
#       - RandAugment added for data-efficient training on the small (~966 img) set
#       - RandomErasing scale increased to simulate out-of-focus cell regions
# 4. CHECKPOINT_DIR / MASTER_RESULTS_FILE renamed to *_wsi_*.
# 5. DATA_DIR updated with WSI-specific placeholder path.
# 6. FOLDS default changed 1 → 5 (essential: only ~766 training images).
# 7. EPOCHS default raised 15 → 30 (tiny dataset benefits from more passes).
# 8. get_dataset_stats: larger default batch size kept at 64; unchanged logic.
# ==============================================================================

# ==============================================================================
# 0. GLOBAL CONFIG & REPRODUCIBILITY
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "checkpoints_sipakmed_wsi")   # ← CHANGED (was _fci)
MASTER_RESULTS_FILE = "sipakmed_wsi_results.csv"           # ← CHANGED (was _fci_)

# ------------------------------------------------------------------------------
# CHANGE 1: CLASS_NAMES updated to WSI folder names.
# WSI folder names use "im_" prefix and hyphen in Superficial-Intermediate.
# torchvision / our loader will index them in CLASS_NAMES order.
# ------------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = [
    "abnormal_Dyskeratotic",
    "abnormal_Koilocytotic",
    "benign_Metaplastic",
    "normal_Parabasal",
    "normal_Superficial-Intermediate",
]

# ==============================================================================
# 1. LOSS — Focal Cross-Entropy  (unchanged)
# ==============================================================================
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ==============================================================================
# 2. PER-MODEL RESOLUTION LOOKUP  (unchanged)
# ==============================================================================
MODEL_IMG_SIZES = {
    "inception_v3":              299,
    "inception_resnet_v2":       299,
    "xception":                  299,
    "efficientnet_b0":           224,
    "tf_efficientnet_b3":        300,
    "tf_efficientnet_b7":        600,
    "swin_tiny_patch4_window7_224":        224,
    "swin_base_patch4_window7_224":        224,
    "swinv2_base_window12to16_192to256":   256,
    "vit_base_patch16_224":      224,
    "vit_large_patch16_224":     224,
    "vit_base_patch16_224_miil": 224,
    "deit_small_distilled_patch16_224": 224,
    "deit_base_distilled_patch16_224":  224,
    "beit_base_patch16_224":     224,
    "t2t_vit_14":                224,
    "convnext_base":             224,
    "convmixer_768_32":          224,
    "resnet18":                  224,
    "resnet50":                  224,
    "resnet101":                 224,
    "vgg16":                     224,
    "vgg19":                     224,
    "seresnet50":                224,
    "resnext101_32x8d":          224,
    "densenet121":               224,
    "densenet201":               224,
    "mobilenetv2_100":           224,
    "nasnet_mobile":             224,
    "squeezenet1_0":             224,
    "mobilevit_s":               256,
    "regnety_008":               224,
    "resnetv2_50x1_bit":         224,
    "hrnet_w32":                 224,
    "pvt_v2_b2":                 224,
    "cvt_13":                    224,
    "coat_lite_small":           224,
    "twins_svt_small":           224,
    "poolformer_s12":            224,
    "efficientformer_l1":        224,
}


def get_img_size(model_name: str) -> int:
    if model_name in MODEL_IMG_SIZES:
        return MODEL_IMG_SIZES[model_name]
    print(f"  [get_img_size] '{model_name}' not in table — querying timm config...")
    try:
        cfg = timm.get_pretrained_cfg(model_name)
        size = cfg.input_size[1]
        print(f"  [get_img_size] timm reports {size}px for '{model_name}'")
        return size
    except Exception:
        print(f"  [get_img_size] Could not resolve size for '{model_name}', defaulting to 224")
        return 224


# ==============================================================================
# 3. DATA UTILS & STATS
# ==============================================================================
_STATS_CACHE: dict = {}


# ------------------------------------------------------------------------------
# CHANGE 2: load_flat_dataset now scans RECURSIVELY.
#
# WSI disk layout (original SIPaKMeD distribution & Kaggle mirror):
#   DATA_DIR/
#     im_Dyskeratotic/
#       im_Dyskeratotic/          ← same name, one level deeper
#         001.bmp
#         002.bmp
#         ...
#         CROPPED/                ← single-cell crops; we SKIP this folder
#     im_Koilocytotic/ ...
#
# Strategy: for each class folder, use os.walk() to find ALL image files
# in any subdirectory, but explicitly SKIP any path containing "CROPPED"
# (we want WSI cluster images only, not the single-cell crops).
# The .bmp extension is included; jpg/png kept for flexibility.
# ------------------------------------------------------------------------------
def load_flat_dataset(data_dir: str):
    """
    Recursively scans each class folder for image files.
    Skips any path whose components include 'CROPPED' (FCI single-cell crops
    that live inside the WSI folder tree — we do NOT want those here).

    Returns:
        all_samples : list of (abs_path, label_int)
        all_targets : list of label_int
        class_to_idx: dict {folder_name: int}
    """
    from collections import Counter

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    missing = [n for n in CLASS_NAMES
               if not os.path.isdir(os.path.join(data_dir, n))]
    if missing:
        raise FileNotFoundError(
            f"Expected class folders not found in {data_dir}:\n"
            + "\n".join(f"  MISSING: {m}" for m in missing)
            + "\nCheck DATA_DIR and folder names match CLASS_NAMES exactly."
        )

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}  # .bmp added explicitly

    all_samples, all_targets = [], []
    for cls_name in CLASS_NAMES:
        label    = class_to_idx[cls_name]
        cls_root = os.path.join(data_dir, cls_name)
        found    = []

        for dirpath, dirnames, filenames in os.walk(cls_root):
            # Skip CROPPED sub-folders (FCI single-cell crops)
            # Also prune them from os.walk so we don't descend into them
            dirnames[:] = [d for d in dirnames
                           if d.upper() != "CROPPED"]
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    found.append(os.path.join(dirpath, fname))

        # Deduplicate (case-sensitive FS on Linux; extra safety on Windows)
        found = list(dict.fromkeys(found))
        for path in found:
            all_samples.append((path, label))
            all_targets.append(label)

    print(f"  Class → index: {class_to_idx}")
    print(f"  Total images : {len(all_samples)}")
    dist = Counter(all_targets)
    for cls_name in CLASS_NAMES:
        print(f"    [{class_to_idx[cls_name]}] {cls_name}: "
              f"{dist[class_to_idx[cls_name]]} images")

    if len(all_samples) == 0:
        raise RuntimeError(
            f"No images found under {data_dir}.\n"
            "Ensure the WSI .bmp files are inside the class sub-folders "
            "(e.g. im_Dyskeratotic/im_Dyskeratotic/*.bmp)."
        )
    return all_samples, all_targets, class_to_idx


def stratified_train_test_split(all_samples, all_targets,
                                test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split as _tts
    idx = list(range(len(all_samples)))
    train_idx, test_idx = _tts(
        idx, test_size=test_size, stratify=all_targets, random_state=seed
    )
    train_samples = [all_samples[i] for i in train_idx]
    train_targets = [all_targets[i] for i in train_idx]
    test_samples  = [all_samples[i] for i in test_idx]
    test_targets  = [all_targets[i] for i in test_idx]

    print(f"  Train: {len(train_samples)} | Test: {len(test_samples)} "
          f"(stratified {int((1-test_size)*100)}/{int(test_size*100)} split, seed={seed})")
    return train_samples, train_targets, test_samples, test_targets


def get_dataset_stats(train_samples: list, target_img_size: int = 224):
    if target_img_size in _STATS_CACHE:
        mean, std = _STATS_CACHE[target_img_size]
        print(f"--- Stats cache hit ({target_img_size}px) | "
              f"Mean: {[f'{m:.4f}' for m in mean]} | Std: {[f'{s:.4f}' for s in std]} ---")
        return mean, std

    print(f"--- Computing Pixel-Level Stats ({target_img_size}x{target_img_size}) ---")
    trans = v2.Compose([
        v2.Resize((target_img_size, target_img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    ds     = SafeDiskDataset(train_samples, [s[1] for s in train_samples], trans)
    loader = DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    sum_, sum_sq, n_pixels = torch.zeros(3), torch.zeros(3), 0
    for imgs, _ in tqdm(loader, leave=False, desc="Computing stats"):
        b, c, h, w = imgs.shape
        n_pixels += b * h * w
        for i in range(3):
            sum_[i]    += torch.sum(imgs[:, i, :, :])
            sum_sq[i]  += torch.sum(imgs[:, i, :, :] ** 2)

    mean = (sum_ / n_pixels).tolist()
    std  = torch.sqrt((sum_sq / n_pixels) - (torch.tensor(mean) ** 2)).tolist()
    print(f"    Mean: {[f'{m:.4f}' for m in mean]} | Std: {[f'{s:.4f}' for s in std]}")

    _STATS_CACHE[target_img_size] = (mean, std)
    return mean, std


class SafeDiskDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples   = samples
        self.targets   = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ==============================================================================
# 3. TRANSFORMS — updated for WSI cluster images
# ------------------------------------------------------------------------------
# CHANGE 3: Augmentation strengthened for the tiny WSI dataset (~966 images).
#
# WHY these additions vs FCI:
#   • WSI images contain clusters of multiple cells, so spatial context is
#     larger and more varied — RandomPerspective distorts the cluster geometry
#     realistically (slide tilts/smears).
#   • GaussianBlur mimics microscope focus drift across a cluster.
#   • RandAugment (magnitude=9, num_ops=2) provides diverse policy-based
#     augmentation, proven effective for small medical datasets.
#   • RandomGrayscale (p=0.05) helps learn stain-invariant features.
#   • RandomErasing scale raised (0.02–0.20) to occlude larger cell groups,
#     forcing the model not to rely on a single cluster corner.
#   • Resize scale factor raised 1.1→1.15 to give RandomCrop more variance.
# ==============================================================================
def build_transforms(img_size, mean, std, mode='train'):
    if mode == 'train':
        return v2.Compose([
            # Slightly larger resize so RandomCrop sees different sub-regions
            v2.Resize((int(img_size * 1.15), int(img_size * 1.15))),   # ← 1.1→1.15
            v2.RandomCrop((img_size, img_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=360),
            # NEW: Perspective warp — realistic for slide mounting variation
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            # NEW: Focus blur common in WSI acquisition
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            # NEW: RandAugment — strong policy augmentation for tiny datasets
            v2.RandAugment(num_ops=2, magnitude=9),
            # NEW: Stain-invariance training signal
            v2.RandomGrayscale(p=0.05),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            # Larger erasing scale to occlude cell groups (was 0.02–0.10)
            v2.RandomErasing(p=0.30, scale=(0.02, 0.20), ratio=(0.3, 3.3))  # ← scale raised
        ])
    else:
        return v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])


# ==============================================================================
# 4. TTA — unchanged; rotation-invariance is equally valid for WSI clusters
# ==============================================================================
def tta_predict(model, imgs):
    views = [
        imgs,
        torch.rot90(imgs, 1, [2, 3]),
        torch.rot90(imgs, 2, [2, 3]),
        torch.rot90(imgs, 3, [2, 3]),
        v2.functional.hflip(imgs),
        torch.rot90(v2.functional.hflip(imgs), 1, [2, 3]),
        torch.rot90(v2.functional.hflip(imgs), 2, [2, 3]),
        torch.rot90(v2.functional.hflip(imgs), 3, [2, 3]),
    ]
    logits_sum = None
    for v_img in views:
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(v_img)
        probs = torch.softmax(logits, dim=1)
        logits_sum = probs if logits_sum is None else logits_sum + probs
    return logits_sum / len(views)


# ==============================================================================
# 5. EVALUATION  (unchanged)
# ==============================================================================
def evaluate_set(model, loader, use_tta=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            if use_tta:
                probs = tta_predict(model, imgs)
                preds = probs.argmax(dim=1).cpu()
            else:
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                    preds = model(imgs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)

    pv = torch.cat(all_preds).numpy()
    lv = torch.cat(all_labels).numpy()
    return {
        'acc':       accuracy_score(lv, pv),
        'f1':        f1_score(lv, pv, average='macro', zero_division=0),
        'recall':    recall_score(lv, pv, average='macro', zero_division=0),
        'precision': precision_score(lv, pv, average='macro', zero_division=0),
    }


def print_confusion_matrix(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            preds = tta_predict(model, imgs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)
    pv = torch.cat(all_preds).numpy()
    lv = torch.cat(all_labels).numpy()
    cm = confusion_matrix(lv, pv)
    print("\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = "          " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:10]:<10}", " ".join(f"{v:>7}" for v in row))


# ==============================================================================
# 6. MODEL MAP  (unchanged — same architectures, same timm names)
# ==============================================================================
MODELS_MAP = {
    "RESNET18":           "resnet18",
    "RESNET50":           "resnet50",
    "RESNET101":          "resnet101",
    "VGG16":              "vgg16",
    "VGG19":              "vgg19",
    "SE_RESNET50":        "seresnet50",
    "RESNEXT101":         "resnext101_32x8d",
    "DENSENET121":        "densenet121",
    "DENSENET201":        "densenet201",
    "INCEPTIONV3":        "inception_v3",
    "INCEPTIONRESNETV2":  "inception_resnet_v2",
    "XCEPTION":           "xception",
    "EFFICIENTNETB0":     "efficientnet_b0",
    "EFFICIENTNETB3":     "tf_efficientnet_b3",
    "EFFICIENTNETB7":     "tf_efficientnet_b7",
    "MOBILENETV2":        "mobilenetv2_100",
    "NASNETMOBILE":       "nasnet_mobile",
    "SQUEEZENET":         "squeezenet1_0",
    "MOBILEVIT":          "mobilevit_s",
    "PROPOSED_GCSWIN":    "swin_tiny_patch4_window7_224",
    "SWIN":               "swin_base_patch4_window7_224",
    "SWIN_V2":            "swinv2_base_window12to16_192to256",
    "VIT_BASE":           "vit_base_patch16_224",
    "VIT_LARGE":          "vit_large_patch16_224",
    "DEIT_SMALL":         "deit_small_distilled_patch16_224",
    "DEIT_BASE":          "deit_base_distilled_patch16_224",
    "BEIT":               "beit_base_patch16_224",
    "T2T_VIT":            "t2t_vit_14",
    "CONVNEXT":           "convnext_base",
    "CONVMIXER":          "convmixer_768_32",
    "REGNETY":            "regnety_008",
    "BIT_R50":            "resnetv2_50x1_bit",
    "HRNET":              "hrnet_w32",
    "PVT":                "pvt_v2_b2",
    "CVT":                "cvt_13",
    "COAT_LITE":          "coat_lite_small",
    "TWINS":              "twins_svt_small",
    "POOLFORMER":         "poolformer_s12",
    "EFFICIENTFORMER":    "efficientformer_l1",
    "VITAEV2":            "vit_base_patch16_224_miil",
}


def get_model_kwargs(model_name):
    kwargs = {}
    if "inception" in model_name:
        kwargs['aux_logits'] = False
    if "swinv2" in model_name:
        kwargs['img_size'] = 256
    return kwargs


def build_classifier(model_name, kwargs):
    backbone = timm.create_model(
        model_name, pretrained=True, num_classes=0, **kwargs
    )
    head = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(backbone.num_features, NUM_CLASSES)
    )
    return nn.Sequential(backbone, head)


# ==============================================================================
# 7. EXPERIMENT ENGINE  (fold/epoch defaults adjusted for WSI size)
# ==============================================================================
def run_experiment(model_name, train_samples, train_targets,
                   test_samples, test_targets,
                   epochs=30, folds=5, patience=7):   # ← epochs 20→30, patience 5→7
    # Resume guard
    if os.path.exists(MASTER_RESULTS_FILE):
        if model_name in pd.read_csv(MASTER_RESULTS_FILE)['model_name'].values:
            print(f"--- Skipping {model_name} (already completed) ---")
            return

    img_size = get_img_size(model_name)
    stats    = get_dataset_stats(train_samples, target_img_size=img_size)
    print(f"  → img_size={img_size}px  "
          f"mean={[f'{m:.4f}' for m in stats[0]]}  "
          f"std={[f'{s:.4f}' for s in stats[1]]}")

    fold_metrics = []
    safe_name    = model_name.replace("/", "_").replace(":", "_")
    kwargs       = get_model_kwargs(model_name)

    if folds == 1:
        from sklearn.model_selection import train_test_split as _tts
        all_idx = list(range(len(train_samples)))
        tr_idx, val_idx = _tts(
            all_idx, test_size=0.2, stratify=train_targets, random_state=42
        )
        fold_splits = [(tr_idx, val_idx)]
    else:
        skf         = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fold_splits = list(skf.split(train_samples, train_targets))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        model = None
        try:
            print(f"\n--- {model_name} | Fold {fold+1}/{folds} ---")

            fold_train_samples = [train_samples[i] for i in tr_idx]
            fold_train_targets = [train_targets[i] for i in tr_idx]
            fold_val_samples   = [train_samples[i] for i in val_idx]
            fold_val_targets   = [train_targets[i] for i in val_idx]

            fold_class_counts = np.bincount(fold_train_targets, minlength=NUM_CLASSES)
            sample_weights = (1.0 / (torch.tensor(fold_class_counts, dtype=torch.float) + 1e-6))[fold_train_targets]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

            model = build_classifier(model_name, kwargs).to(DEVICE)

            train_trans = build_transforms(img_size, stats[0], stats[1], mode='train')
            val_trans   = build_transforms(img_size, stats[0], stats[1], mode='val')

            train_loader = DataLoader(
                SafeDiskDataset(fold_train_samples, fold_train_targets, train_trans),
                batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                SafeDiskDataset(fold_val_samples, fold_val_targets, val_trans),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )

            criterion = FocalCrossEntropyLoss(gamma=2.0, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
            warmup    = LinearLR(optimizer, start_factor=0.1, total_iters=3)
            cosine    = CosineAnnealingLR(optimizer, T_max=epochs - 3, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
            scaler    = torch.amp.GradScaler(device="cuda", enabled=(DEVICE == "cuda"))

            best_f1    = -1.0
            best_m     = {}
            no_improve = 0

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for imgs, lbls in tqdm(train_loader, desc=f"F{fold+1} E{epoch+1}", leave=False):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                        loss = criterion(model(imgs), lbls)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()

                scheduler.step()

                m = evaluate_set(model, val_loader, use_tta=False)
                avg_loss = running_loss / len(train_loader)
                print(f"  E{epoch+1} | Loss: {avg_loss:.4f} | "
                      f"Acc: {m['acc']:.4f} | F1: {m['f1']:.4f} | "
                      f"Recall: {m['recall']:.4f}")

                if m['f1'] > best_f1:
                    best_f1, best_m = m['f1'], m
                    no_improve = 0
                    torch.save(
                        model.state_dict(),
                        f"{CHECKPOINT_DIR}/{safe_name}_f{fold+1}.pth"
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                        break

            fold_metrics.append(best_m)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"--- OOM on Fold {fold+1} for {model_name} — skipping fold ---")
                continue
            raise
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    if not fold_metrics:
        print(f"--- All folds failed for {model_name} — skipping ---")
        return

    # -----------------------------------------------------------------------
    # TEST SET: ENSEMBLE SOFTMAX AVERAGING ACROSS ALL FOLD CHECKPOINTS
    # -----------------------------------------------------------------------
    try:
        test_trans = build_transforms(img_size, stats[0], stats[1], mode='test')
        test_loader = DataLoader(
            SafeDiskDataset(test_samples, test_targets, test_trans),
            batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        ensemble_probs = None
        loaded_folds   = 0

        for fold_idx in range(len(fold_metrics)):
            ckpt_path = f"{CHECKPOINT_DIR}/{safe_name}_f{fold_idx+1}.pth"
            if not os.path.exists(ckpt_path):
                print(f"  Warning: checkpoint missing for fold {fold_idx+1}, skipping")
                continue

            fold_model = build_classifier(model_name, kwargs).to(DEVICE)
            fold_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            fold_model.eval()

            fold_probs = []
            with torch.no_grad():
                for imgs, _ in tqdm(test_loader, desc=f"Ensemble fold {fold_idx+1}", leave=False):
                    imgs = imgs.to(DEVICE)
                    fold_probs.append(tta_predict(fold_model, imgs).cpu())

            fold_probs_cat = torch.cat(fold_probs)
            ensemble_probs = fold_probs_cat if ensemble_probs is None \
                             else ensemble_probs + fold_probs_cat
            loaded_folds += 1

            del fold_model
            gc.collect()
            torch.cuda.empty_cache()

        ensemble_probs /= loaded_folds
        pred_ens = ensemble_probs.argmax(dim=1).numpy()
        lv       = np.array(test_targets)

        test_results = {
            'acc':       accuracy_score(lv, pred_ens),
            'f1':        f1_score(lv, pred_ens, average='macro', zero_division=0),
            'recall':    recall_score(lv, pred_ens, average='macro', zero_division=0),
            'precision': precision_score(lv, pred_ens, average='macro', zero_division=0),
        }

        per_class_f1 = f1_score(lv, pred_ens, average=None, zero_division=0)
        for i, cls in enumerate(CLASS_NAMES):
            test_results[f'f1_{cls.replace("-", "_")}'] = per_class_f1[i]

        fold_df   = pd.DataFrame(fold_metrics)
        final_row = {'model_name': model_name, 'img_size': img_size}
        for col in fold_df.columns:
            final_row[f'cv_{col}_mean'] = fold_df[col].mean()
            final_row[f'cv_{col}_std']  = fold_df[col].std()
        for k, v in test_results.items():
            final_row[f'test_{k}'] = v

        pd.DataFrame([final_row]).to_csv(
            MASTER_RESULTS_FILE, mode='a',
            header=not os.path.exists(MASTER_RESULTS_FILE), index=False
        )

        print(f"\n✅ {model_name}")
        print(f"   Test → Acc: {test_results['acc']:.4f} | F1: {test_results['f1']:.4f} | "
              f"Recall: {test_results['recall']:.4f}")
        print(f"   CV   → F1: {fold_df['f1'].mean():.4f} ± {fold_df['f1'].std():.4f}")
        print(f"   Per-class F1: { {c: f'{per_class_f1[i]:.3f}' for i, c in enumerate(CLASS_NAMES)} }")

    except Exception as e:
        print(f"--- Ensemble Eval Error for {model_name}: {e} ---")


# ==============================================================================
# 8. MAIN
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    DATA_DIR = r"C:\Users\Yash\Downloads\wsi\DATA_DIR"   # ← UPDATE THIS PATH

    print(f"\nRunning on: {DEVICE}")

    all_samples, all_targets, class_to_idx = load_flat_dataset(DATA_DIR)
    train_samples, train_targets, test_samples, test_targets = \
        stratified_train_test_split(all_samples, all_targets, test_size=0.2, seed=42)

    target_models = [
        "resnet18",
        "resnet101",
        "vgg16",
        "vgg19",
        "seresnet50",
        "resnext101_32x8d",
        "densenet121",
        "inception_resnet_v2",
        "xception",
        "tf_efficientnet_b7",
        "mobilenetv2_100",
        "nasnet_mobile",
        "squeezenet1_0",
        "vit_large_patch16_224",
        "deit_small_distilled_patch16_224",
        "deit_base_distilled_patch16_224",
        "beit_base_patch16_224",
        "t2t_vit_14",
        "convmixer_768_32",
        "cvt_13",
        "coat_lite_small",
        "twins_svt_small",
    ]
    print(f"Total target models: {len(target_models)}\n")

    EPOCHS   = 15
    FOLDS    = 1
    PATIENCE = 7

    for m in target_models:
        try:
            run_experiment(
                m,
                train_samples, train_targets,
                test_samples,  test_targets,
                epochs=EPOCHS, folds=FOLDS, patience=PATIENCE
            )
        except Exception as e:
            print(f"Critical failure on {m}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print("\n✅ Master experiment complete. Results in:", MASTER_RESULTS_FILE)