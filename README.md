# MCDC
In this folder, we provide multiple MCDC model variants along with the corresponding baseline implementations to ensure the authenticity and reproducibility of the results reported in the paper.


For various baseline models, **their environment configurations follow the original paper**. Specific parameters and code are provided in this folder.


For MCDC, its environment configuration and training method **are consistent with the baseline models**. The specific corresponding parameters are provided in the 'config' folder.


## Human pose estimation
### Environment



### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```


   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing
```
cd Pose
```
#### Testing on the MPII dataset.
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_64x64_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/your_pretrained.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_64x64_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset.
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_64x64_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/your_pretrained.pth \
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_64x64_adam_lr1e-3.yaml \
```



## Object Detection & Image Classification

### Installation

```
# Download the code
git clone https://github.com/LabSAINT/MCDC-Conv

# Create an environment
cd MCDC-Conv
conda create -n myenv python==3.7
conda activate myenv
pip3 install -r requirements.txt
```

MCDC-Conv is evaluated using two most representative computer vision tasks, object detection and image classification. Specifically, we construct YOLOv5-MCDC, ResNet18-MCDC and ResNet50-MCDC, and evaluate them on COCO-2017, Tiny ImageNet, and CIFAR-10 datasets in comparison with several state-of-the-art deep learning models.

### YOLOV5

```
cd YOLOv5
```










##### Evaluation

The script `val.py` can be used to evaluate the pre-trained models

```
  $ python val.py --weights './weights/nano_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  $ python val.py --weights './weights/small_best.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
```

##### Training 


```
python3 train.py --data coco.yaml --cfg ./models/space_depth_n.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project space_depth --name space_depth_n

python3 train.py --data coco.yaml --cfg ./models/space_depth_s.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project space_depth --name space_depth_s

```
 


### Tiny-ImageNet

ResNet18-MCDC and ConvNext-MCDC model is evaluated on the TinyImageNet dataset

```bash
cd Classification/tinyimagenet
```




##### Dataset

Tiny-ImageNet-200 dataset can be downloaded from this link [tiny-imagenet-200.zip](https://drive.google.com/file/d/1xLcRyy7-jLV-ywaGwCHxymX9D05X0g5i/view?usp=sharing)

##### Evaluation

```bash
$ python3 test.py -net resnet18_MCDC -weights ./weights/resnet18_MCDC.pt
```

##### Training

```bash
python3 train_tiny.py -net resnet18_MCDC -b 256 -lr 0.01793 -momentum 0.9447 -weight_decay 0.002113 -gpu -project MCDC -name resnet18_MCDC
```



### CIFAR-10

ResNet50-MCDC model is implemented on the CIFAR-10 dataset

```bash
cd ./Classification/cifar
```

##### Dataset

```bash
CIFAR-10 dataset will be downloaded automatically by the script
```




