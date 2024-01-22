Zero-shot connector anomaly detector
===================================
  Pytorch implementation of 'A Zero-shot connector anomaly detection approach based on similarity-contrast learning'

  
Requirements
-----------------------------------
  The code has been tested with python 3.7, pytorch 1.7.1 and Cuda 10.1.
  
  	conda create -n zsad python=3.7.13
  	conda activate zsad
  	pip install -r requirements.txt
		
Required Data 
-----------------------------------
  To evaluate/train zsda, you will need to download the required datasets.
  
1.[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)<br />
2.[DeepPCB](https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB)<br />
  
### Pretraining on VOC2012
    python ./train.py --bg_data path/to/VOC2012
        
### Evaluation on DeepPCB(This step only generates predicted bounding box results, and metric calculation needs to be done in [this](https://github.com/tangsanli5201/DeepPCB)<br />)
    python ./experiment_on_pcb.py --run_mode val --DeepPCB_path path/to/DeepPCB

### Finetune on DeepPCB(optional)  
    python ./experiment_on_pcb.py --run_mode train --DeepPCB_path path/to/DeepPCB

### Demo on image pair 
    python ./inference.py path/to/test_image path/to/template_image

### pretrained models
1.[pretrained on VOC2012](https://pan.baidu.com/s/1emCdEGXfzELTubF3xl8DnQ?pwd=attp)：attp<br />
2.[finetuned on DeepPCB](https://pan.baidu.com/s/1UVyODqdJvZSeowPvbPXiJQ?pwd=3we4)：3we4<br />
