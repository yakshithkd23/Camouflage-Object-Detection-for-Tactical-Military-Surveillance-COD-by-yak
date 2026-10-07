from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from scripts.DGnet import DGNet


def run_ptq_int8(
    val_image_path: str = r"C:\Users\HP\Downloads\000005.jpg",
    output_path: str = r"E:\camoflage\Camouflage-Object-Detection-for-Tactical-Military-Surveillance-COD-by-yak\models\quantized\dgnet_int8.tflite",
):
    """Converts DGNet model to full INT8 TFLite format using sample validation image calibration."""

    print("--- Starting INT8 PTQ Conversion ---")

    # 1. Instantiate and build DGNet architecture
    model = model = DGNet()  # ✅ Fixed
    model.build((None, 384, 384, 3))

    print("DGNet model built successfully.")

    # 2. Representative Dataset Generator for Calibration
    def representative_dataset():
        img_path = Path(val_image_path)

        if not img_path.exists():
            raise FileNotFoundError(
                f"Calibration image not found at: {val_image_path}"
            )

        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(
                f"Unable to read calibration image: {val_image_path}"
            )

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to DGNet input size
        img = cv2.resize(img, (384, 384))

        # Convert uint8 -> float32
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Feed calibration image multiple times
        for _ in range(100):
            yield [img]

    # 3. Configure TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable Post-Training Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for activation calibration
    converter.representative_dataset = representative_dataset

    # Force all supported operations to INT8
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Force input and output tensors to INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 4. Convert Model
    print("Converting DGNet to INT8...")

    int8_model = converter.convert()

    # 5. Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Save INT8 TFLite model
    with open(output_path, "wb") as f:
        f.write(int8_model)

    print("==========================================")
    print("✅ INT8 TFLite model successfully created!")
    print(f"📁 Saved to: {output_path}")
    print("==========================================")


if __name__ == "__main__":
    run_ptq_int8()