--------------------- optimisation techiques ----------------------------------
we can do two type of the optimisation techniques 
   1. PTQ-post training quantisation
   2. QAT- quatisation aware training

PTQ:
it is the fastest staring point brcause it requires zero retraining time. work best when minor dorps in accuracy are acceptable or when  high -precious feature resolution is not serverly impacted.

QAT:
insert fake wuatisation nodes into the computation graph  during training r fine tuning. the model learns and adapys to rounding errors and precious lpss while adjusting its wrights.
When to use: Essential for fine-grained computer vision tasks (such as camouflage detection or subtle edge segmentation) where standard PTQ causes noticeable drop-offs in accuracy metrics ($S$-measure, $E$-measure, MAE).


conclusion: i came to know that qat is best comapre ptq

-------------------------------------------------------------------------------------------------

# comparative analysis:
paper: Optimizing Large Language Models through
Quantization: A Comparative Analysis of PTQ and
QAT Techniques

they introduced mixed-precision quatisation.

quatisation in neural network offers a promising solution to these chakkenge by reducing the precision oog model paramter and activation.

Typically, neural network weights
and activations are represented using 32-bit floating-point
(FP32) numbers, which provide high precision but consume
substantial memory and computational resources

fundamental of quantisation:
it has two categories  unifrom anf non-uniform quatisation. each it has own advantage and trade-off.


quatisation aware trainingL
qat integreates the quantization process into the training loop allowing the model to adapt its paramters to the lower precision representation(i think they are telling negative precision). durin each trainig itertin a batch B is sample from the training data D, and the current model parameter theta are quatized to L then compute using uatized paramters and the batch data.
 gradient G are caluclated with respect to the loss using a stright Through estimator 

 QAT typically results in better performance compared
to PTQ, as the model can learn to compensate for the
quantization-induced errors. However, it requires additional
computational resources and time for training, making it more
suitable for scenarios where maintaining high accuracy is
critical and retraining is feasible
                     ............................... Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques

-----------------------------------**Optimal bit allocation** -----------------------------------------------------
is a strategy used in model quantization to decide how many bits (precision) each layer of a neural network should get so you shrink the model as much as possible without breaking its performance.

Instead of giving every layer the exact same bit precision (like 8 bits everywhere), you give **more bits to high-impact layers** and **fewer bits to low-impact layers**.

---

### Key Idea

The document bases optimal bit allocation on two main factors:

1. **Layer Sensitivity ($\alpha_l$):** How much the model's accuracy drops if you mess with this layer.
2. **Weight Variance ($\sigma_l^2$):** How widely spread out or diverse the values in that layer are.

---

### Simple Example: The Cooking Recipe Analogy

Imagine you are baking a cake, and you need to reduce the quality or precision of a few ingredients to save money (your "bit budget"):

* **Layer 1: The Flour/Sugar (High Sensitivity + High Variance)**
* *Role:* Crucial structure and sweetness.
* *Allocation:* **High precision (e.g., 8-bit).** If you mess up the measurement even a little, the cake fails completely.


* **Layer 2: Vanilla Extract (High Sensitivity, Low Variance)**
* *Role:* Strongly affects flavor, but uses a tiny, consistent amount.
* *Allocation:* **Medium precision (e.g., 6-bit).**


* **Layer 3: Food Coloring (Low Sensitivity + Low Variance)**
* *Role:* Purely decorative; a slight change in shade won't change how tasty the cake is.
* *Allocation:* **Low precision (e.g., 2-bit or 4-bit).** You can round the measurement aggressively to save budget without ruining the cake.



---

### Real World Application in LLMs

When quantizing a Large Language Model under a constrained total bit budget:

* **Critical Layers** (e.g., attention layers handling complex relationships) get assigned **8-bit or 16-bit** parameters to protect accuracy.
* **Non-critical Layers** (e.g., certain redundant feed-forward layers with low weight variance) are squeezed down to **4-bit or 2-bit** precision.

q/a: among those two QAT AND PTQ which one is the mix precesion or optimal bit allocation  strategy

ampng those two quatisation aware traiiiniing is much better suited for mixed precsion and optimal bit allocation strategy.

Quantization-Aware Training (QAT) is much better suited for mixed-precision and optimal bit allocation strategies.

Why QAT Fits Mixed-Precision Better
QAT (Quantization-Aware Training): When you assign different bit-widths (like 2-bit, 4-bit, and 8-bit) across various layers, the model's weight dynamics change drastically. Because QAT retrains the model using a Straight-Through Estimator (STE), the network actively adjusts to the specific bit allocation for each layer, learning to compensate for high errors in low-bit layers.

PTQ (Post-Training Quantization): PTQ applies static scale factors to pre-trained weights without retraining. While you can mathematically calculate optimal bit allocation for PTQ, aggressive low-bit assignments (like 2-bit or 4-bit) in non-critical layers often lead to severe accumulated precision loss that PTQ cannot recover from without retraining.

# Numerical Stability

 # why we need scaling factor?