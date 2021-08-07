# Low Light Object Detection using TFOD 2.0
This repo implements Object detection on Low Light/ Dark images
![Alt Text](https://github.com/ashwin-py/Low-Light-Object-Detection/blob/main/detections.gif)
## Dataset
For this task Exclusively-Dark-Image-Dataset was used.
Follow this link to download the dataset:  
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset  
Annotations here:  
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Groundtruth

Below preprocess steps were performed on the original dataset.
- Since some images were saved in PNG with RGBA channel, All images are converted to JPG with RGB channels.
- Annotations are in the txt format, So all Class info and Bounding box info is transferred to a single csv file.
- Dataset is split in Train and test with ratio of 0.75:0.25
- Dataset is converted to TFRecord format.
- Label_map with 12 classes 
```bash
└── data/  
    ├── annotations.csv  
    ├── train.tfrecord  
    ├── test.tfrecord  
    └── label_map.txt  
```
## Model
ssd_efficientdet_d1_640x640_coco17_tpu-8 Checkpoint is selected for Fine-tuning.  
Model Config changes in the default pipeline.config  
- num of classes: 12
- batch_size: 32
- num_steps: 15000
- bfloat: false(true only for tpu)
- finetune checkpoint path
- label_map path
- tfrecord path for both train and test  
[Prepared Data can be downloaded from here.](https://drive.google.com/file/d/1bYcrm5rWjhUpmJqY4zLdkTiVlqKign2m/view?usp=sharing)

## Training

Training performed on AWS EC2 Instance  
Instance details:  
- Deep Learning AMI (Ubuntu 18.04)  
- Type: p3.8xlarge (4xTeslaV100 16gb gpu)  
- Time for 15000 steps: 5:30hrs  
```bash
  └── checkpoint/  
      └── ssd_efficientdet_d1_640x640_coco17_tpu-8/  
          ├── eval/  
          ├── train/  
          ├── ssd_efficientdet_d1_640x640_coco17_tpu-8.config  
          ├── ckpt-xx-data-xxxx  
          └── ckpt-xx-index  
```
[Trained Checkpint can be downloaded from here](https://drive.google.com/file/d/1rNA6U2sYpP4peDc4DYedHBdlWD4mYYBz/view?usp=sharing)
## Evaluation

Evaluation done on default coco_detection_metrics.  

#### **DetectionBoxes_Precision/mAP: 0.368919**  

## Usage
* Download and save the checkpoint.
* Download some test images or download images uploaded here in test_images folder.
* Clone the TFOD repo.
* Run inference.py
    ```
    cd models/research
    
    python run_inference.py --images_folder_dir=test_images/ \
    --model_dir=checkpoint/ \
    --label_map_path=data/label_map.txt \
    --save_detection=True \
    --save_path=. \  ## if save_detection=True
    ```
## To Do
Train for more steps with existing checkpoint.  
Train a different model architecture.  

## References
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
https://tensorflow-object-detection-api-tutorial.readthedocs.io/
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md  

