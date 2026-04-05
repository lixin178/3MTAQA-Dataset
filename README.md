### Requirement
Python >= 3.6

Pytorch >=1.8.0

### Dataset Preparation

**1.3MTAQA dataset** 

If the article is accepted for publication, you can download our prepared 3MTAQA dataset demo from Google Drive. Then, please move the uncompressed data folder to `./data/frames`. We used the I3D backbone pretrained on Kinetics.


**2.MTL-AQA dataset**

### Training & Evaluation

LVFL, as a plug and play module, has good versatility and can be easily integrated into other models. In this study, we first select classic action quality assessment models R(2+1)D-34-WD, USDL, MUSDL, and DAE as baseline models, and then integrate LVFL into the above models, namely R(2+1)D-34-WD+LVFL,USDL+LVFL, MUSDL+LVFL and DAE+LVFL, respectively, to evaluate the action quality.Take **DAE+LVFL** as an example,To train and evaluate on 3MTAQA:

` python DAE+LVFL.py --log_info=DAE+LVFL --num_workers=8 --gpu=0 --train_batch_size=4 --test_batch_size=20 --num_epochs=100 `

If you use the 3MTAQA dataset, please cite this paper: A Language-Guided Visual Feature Learning Strategy for Multimodal Teacher Action Quality Assessment.
