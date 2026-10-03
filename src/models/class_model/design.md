# Stage 2: Tactical Mask-Gated Target Classifier

## Aim & Objective

The primary aim of **Stage 2** is to verify and classify whether a candidate surveillance region contains a camouflaged entity or non-target noise.

Instead of processing the raw frame directly, Stage 2 takes the original **Input RGB Image ($X$)** alongside the spatial **Mask Image ($M$)** produced by Stage 1 (DGNet Segmentation). By masking and cropping tightly around the isolated Region of Interest (ROI), Stage 2 removes surrounding foliage and terrain noise to accurately predict whether a camouflaged target is present and identify its specific target category ($Y$).

---

## Data Flow Diagram 
Here is step-by-step how an input image with a military/army target passes through the pipeline, gets processed using the generated mask, and is evaluated by **EfficientNet-B0** to decide whether it is **Camouflaged Target** or **Non-Target (Background Noise)**.

---

### Step-by-Step Pipeline Walkthrough
```mermaid
graph TD
    A["Raw Input RGB Image (X)<br><i>(e.g., Army soldier hidden in forest)</i>"] --> B["Stage 1: DGNet Segmentor"]
    A --> C["Mask-Gated ROI Cropper"]
    
    B --> M["Generates Spatial Mask M"]
    M --> C
    
    C --> D["Cropped Target ROI (224x224)<br><i>(Foliage zeroed out)</i>"]
    
    D --> E["EfficientNet-B0 Classifier"]
    E --> F["Feature Extraction via MBConv + SE Blocks"]
    F --> G["Softmax Classification Head"]
    
    G --> H["Class Prediction (Y): Camouflaged Target vs. Non-Target"]

    classDef input fill:#1f2937,stroke:#6b7280,color:#fff;
    classDef stage fill:#1e3a8a,stroke:#3b82f6,color:#fff;
    classDef crop fill:#065f46,stroke:#10b981,color:#fff;
    classDef output fill:#4c1d95,stroke:#8b5cf6,color:#fff;

    class A input;
    class B,E,F,G stage;
    class C,D crop;
    class H output;
    ```

### 1. Step 1: Input RGB Frame ($X$) & DGNet Mask Generation ($M$)

* **The Raw Input ($X$):** A full surveillance image containing heavy background clutter (e.g., an army soldier in BDU camouflage standing among trees and foliage).
* **DGNet Inference:** The frame passes into **Stage 1 (DGNet)**, which outputs a single-channel **Binary Mask ($M$)** highlighting candidate pixel regions where a camouflage boundary is localized.

---

### 2. Step 2: Mask-Gated ROI Cropping (Filtering Out Noise)

Passing full raw images into a classifier often leads to false positives because the network gets confused by ambient tree shadows, leaves, or rocks.
To fix this, the pipeline performs **Mask-Gating**:

1. **Element-wise Multiplication:** $X_{\text{Gated}} = X_{\text{RGB}} \odot M_{\text{DGNet}}$. Every pixel outside the mask turns completely black ($0, 0, 0$), stripping away the foliage and terrain background.
2. **Bounding Box Crop:** Contour detection draws a bounding box around the remaining non-zero patch and crops it.
3. **Resizing:** The cropped patch is resized to $224 \times 224 \times 3$, creating a clean **Region of Interest (ROI)** patch focusing strictly on the potential army target.

---

### 3. Step 3: EfficientNet Processing & Feature Learning

The $224 \times 224$ cropped ROI patch enters **EfficientNet-B0**:

1. **MBConv Depthwise Convolutions:** The network scans spatial properties across channels efficiently using minimal parameters (~5.3M).
2. **Squeeze-and-Excitation (SE) Channel Attention:** Since camouflage is designed to blur edges, SE blocks dynamically scale important feature channels up and suppress irrelevant ones. They force the network to pay attention to subtle tactical indicators:
* **Unnatural geometric outlines:** Straight lines of gear, helmets, weapon barrels, or boot soles.
* **Textural irregularities:** Synthetic fabric weave vs. organic bark/leaf textures.


3. **Global Average Pooling & Dense Head:** Features are flattened into a 1280-dimensional embedding vector $\phi(\text{ROI})$ and mapped through a Linear + Softmax layer:

$$\hat{y} = \text{Softmax}(W \cdot \phi(\text{ROI}) + b)$$



---

### 4. Step 4: Final Grouping & Decision Output

The network outputs a final prediction confidence score ($0.0 - 1.0$) categorized into grouped classes:

* **Group A: Camouflaged Target Identified** (Confidence $> \text{Threshold}$)
* Sub-classes: *Camouflaged Personnel*, *Armored Vehicle*, *Equipment/Gear*.


* **Group B: Non-Target / False Alarm** (Confidence $< \text{Threshold}$)
* Rejects candidate patches caused by segmentor noise (e.g., strange leaf formations or rock shadows misidentified by Stage 1).

---

## Technical Details

### 1. Mathematical Formulation

* **Mask-Gated Multiplication:**

$$X_{\text{Gated}} = X_{\text{RGB}} \odot M_{\text{DGNet}}$$


* **ROI Extraction & Resizing:**

$$\text{ROI} = \text{Crop}(X_{\text{Gated}}, x_{\min}, y_{\min}, w, h) \longrightarrow \text{Resize to } 224 \times 224$$


* **Probability Output:**

$$\hat{y} = \text{Softmax}(W \cdot \phi(\text{ROI}) + b)$$



### 2. Why EfficientNet-B0?

* **Compound Scaling:** Uniformly balances depth, width, and resolution to maximize accuracy on small cropped target patches.
* **Squeeze-and-Excitation (SE) Attention:** Highlights subtle tactical indicators against organic background clutter.
* **Edge Compatibility:** At **~5.3M parameters**, it runs smoothly on mobile/edge runtimes alongside Stage 1.