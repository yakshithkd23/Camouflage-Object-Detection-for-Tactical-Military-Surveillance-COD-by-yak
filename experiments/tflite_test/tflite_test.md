# Tflite model Test
 This file is used for to test .how actually that tflite model work in target device 

 ## method: google ai edge gallery
note: this one is under private review.

constraints:

physical single-board computers like the Raspberry Pi (ARM Cortex-A series) or NVIDIA Jetson Nano (ARM CPU + Jetson Maxwell GPU) aren't directly available in cloud device labs,

altered approach:
- approach is to target mobile hardware profiles that closely match their compute specs, RAM, and SoC architecture inside the Google AI Edge Portal.
- can measure FPS (frames per second), Inference Speed (latency in ms), and Peak Memory / Model Footprint using physical mobile devices in the Google AI Edge Portal.


| Target Physical Board | Recommended Mobile Device Pool | Hardware Mode (Config) | Why Select This Device? |
|---|---|---|---|
| **Raspberry Pi 4 / Pi 5**<br>*(ARM Cortex CPU focus)* | **Mid-Tier Devices**<br>• Pixel 7a / Pixel 8a<br>• Samsung Galaxy A54 / A55 | **CPU Delegate**<br>*(Set to 2 or 4 threads)* | Standard ARM Cortex CPUs with 4–8 GB RAM. Serves as a direct proxy for evaluating pure CPU inference speed and memory footprint. |
| **NVIDIA Jetson Nano**<br>*(CPU + Embedded GPU focus)* | **High-Tier Devices**<br>• Google Pixel 9 Pro XL<br>• Samsung Galaxy S24 Ultra / S25 Ultra | **GPU / OpenCL Delegate** | Features high-performance GPUs with OpenCL execution drivers. Excellent for checking if your layers run on GPU acceleration. |


