------------------ dataset folder--------------------------
during trainign each datset item loadd into DGNET cnsists of an input image, a ground-truth mask and edge map

training sample components:
- rgb image - 3 channel RGB image .during trainign either the original rgb image  or augumented varient from data/agumented/images to improve model generalisation.
- ground truth mask: the single channel binary mask(1*h*w) defingin object boundaries. used to cm[ute segemetation loss(e.g binrary cross entropy /structure loss) against the models ouput predication
- edge/gradient map(data/agumented/edges): the single channel sobel boundary map(1*H*W). DGNet specifically uses deep gradient learning,rrequiring edge maps during training to supervise its high-frequency tecture and boundary extraction branch


question: do i need to pass this image in single pass it means image are strcutred such way that all are need arrange is correctly each folder
yes, during training,corresponding files(RGB images,GT mask..) are loaded together i  a single pass per bach step. 

data/augmented/
├── images/  --> Input RGB images (Required)
└── GT/      --> Ground-truth binary masks (Required for loss computation)

what need to do[issue 1.0]- create src/utils/dataset.py
Place your dataset pipeline here. It reads paired RGB images from data/augmented/images/ and ground-truth binary masks from data/augmented/GT/ using tf.data.Dataset

src/inference/predict.py: Uses the trained weights to run predictions on new unseen RGB images and output final predicted masks.