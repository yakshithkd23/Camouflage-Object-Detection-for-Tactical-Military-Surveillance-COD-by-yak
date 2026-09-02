--------------------- optimisation techiques ----------------------------------
we can do two type of the optimisation techniques 
   1. PTQ-post training quantisation
   2. QAT- quatisation aware training

PTQ:
it is the fastest staring point brcause it requires zero retraining time. work best when minor dorps in accuracy are acceptable or when  high -precious feature resolution is not serverly impacted.

QAT:
insert fake wuatisation nodes into the computation graph  during training r fine tuning. the model learns and adapys to rounding errors and precious lpss while adjusting its wrights.
When to use: Essential for fine-grained computer vision tasks (such as camouflage detection or subtle edge segmentation) where standard PTQ causes noticeable drop-offs in accuracy metrics ($S$-measure, $E$-measure, MAE).


