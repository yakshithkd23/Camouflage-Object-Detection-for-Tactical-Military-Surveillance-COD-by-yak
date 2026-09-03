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

