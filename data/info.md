# Dataset Structure

```text
data/
├── rgb/                 # Original RGB images
├── masked/              # Ground Truth (GT) mask images
├── augmented/           # Augmented dataset
│   ├── images/          # Input RGB images
│   ├── GT/              # Ground-truth binary masks
│   └── edges/           # Sobel edge/gradient maps
├── info.md
└── raw.dvc

scripts/
└── convert_gt.py        # Python script for GT mask conversion
```

## DVC Tracking

```bash
dvc add data/raw
```

**Or, if you want to track the RGB folder directly:**

```bash
dvc add data/rgb
```

---

# Training Sample Structure

During training, each dataset item loaded into **DGNet** consists of three corresponding components:

1. **RGB image**

   * 3-channel RGB image.
   * During training, this can be either the original RGB image or an augmented variant from `data/augmented/images/`.
   * Used as the primary input to the model.

2. **Ground-truth mask**

   * Single-channel binary mask with shape `(1, H, W)`.
   * Defines the target object/foreground region.
   * Used to compute segmentation loss, such as Binary Cross Entropy or Structure Loss, against the model's predicted mask.

3. **Edge/gradient map**

   * Single-channel Sobel boundary/gradient map with shape `(1, H, W)`.
   * Stored under `data/augmented/edges/`.
   * DGNet uses **Deep Gradient Learning**, so the edge map is required to supervise the model's high-frequency texture and boundary-extraction branch.

---

# Do the Images Need to Be Arranged Correctly?

**Yes.**

The corresponding RGB image, ground-truth mask, and edge map must be correctly paired. They are loaded together as **one training sample** during each batch iteration.

For example:

```text
data/augmented/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── 000003.jpg
│
├── GT/
│   ├── 000001.png
│   ├── 000002.png
│   └── 000003.png
│
└── edges/
    ├── 000001.png
    ├── 000002.png
    └── 000003.png
```

Therefore:

```text
images/000001.jpg  ↔  GT/000001.png  ↔  edges/000001.png
images/000002.jpg  ↔  GT/000002.png  ↔  edges/000002.png
images/000003.jpg  ↔  GT/000003.png  ↔  edges/000003.png
```

The filename or another deterministic identifier should be used to ensure that the three components correspond to the **same original image**.

---

# Training Data Loading

The dataset pipeline should load the corresponding components together.

Conceptually, each training sample is:

```text
RGB Image
    +
Ground-Truth Mask
    +
Edge/Gradient Map
    ↓
DGNet
    ↓
Predicted Segmentation + Gradient/Boundary Features
    ↓
Loss Computation
```

For a batch:

```text
Batch
├── RGB images
├── GT masks
└── Edge maps
```

All three tensors must maintain the same sample ordering within the batch.

---

# Issue 1.0 — Create `src/utils/dataset.py`

Create:

```text
src/utils/dataset.py
```

Place the dataset pipeline in this file.

The pipeline should use **`tf.data.Dataset`** and read the paired training data from:

```text
data/augmented/images/
data/augmented/GT/
data/augmented/edges/
```

The dataset pipeline should:

1. Discover the available RGB images.
2. Match each RGB image with its corresponding GT mask.
3. Match each RGB image with its corresponding edge map.
4. Load the three files together.
5. Decode and preprocess the images.
6. Convert them into the required tensor shapes and data types.
7. Apply any required augmentation.
8. Create batches using `tf.data.Dataset`.
9. Return the RGB image, GT mask, and edge map together for DGNet training.

Conceptually:

```python
image, gt_mask, edge_map = dataset_batch
```

The important point is that **the three components must remain correctly paired throughout preprocessing, shuffling, augmentation, and batching.**

---

# Inference Pipeline

Create/use:

```text
src/inference/predict.py
```

This module uses the trained DGNet weights to run predictions on **new, unseen RGB images**.

Inference flow:

```text
New RGB Image
      ↓
Preprocessing
      ↓
Trained DGNet
      ↓
Predicted Mask
      ↓
Final Output Mask
```

During inference, the model does **not** require the ground-truth mask or edge map as input because these are unavailable for unseen test images.

---

# Important: Do Not Remove Edge Maps

**Do not remove the edge maps if you are training DGNet.**

DGNet's core innovation relies on **Deep Gradient Learning**, which requires explicit edge/gradient maps during training.

The edge maps provide supervision for the model's:

* Boundary extraction
* High-frequency feature learning
* Gradient/edge representation
* Object boundary localization

Therefore, the training dataset should contain all three components:

```text
RGB Image
     +
GT Mask
     +
Edge/Gradient Map
     ↓
   DGNet
```

The minimum recommended training structure is:

```text
data/augmented/
├── images/       # Input RGB images — Required
├── GT/           # Ground-truth binary masks — Required
└── edges/        # Edge/gradient maps — Required for DGNet
```

**Summary:** Keep the RGB images, GT masks, and edge maps properly aligned by filename/ID. During training, each corresponding triplet is loaded together as a single sample and then grouped into batches. The edge maps are essential for DGNet training and should not be removed.
