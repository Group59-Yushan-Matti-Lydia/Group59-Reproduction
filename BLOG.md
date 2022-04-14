# A Reproduction of Tracking Objects as Points
Matti Lang - 5632935 m.a.lang@student.tudelft.nl
Lydia Mak - 5540453 h.y.l.mak@student.tudelft.nl
Yushan Liu - 5525829 y.liu-104@student.tudelft.nl
Group 59                                                                           April 14, 2022 – 8 min read

In this blog post, we reproduced results from the paper “Tracking Object as Points” [1]. Tracking objects is an essential component of computer vision and many different methods have been proposed over the years to optimise this task. In this paper, CenterTrack is presented, combining a point detection method from CenterNet and a tracking method.

Figure 1: Object modelled as the centre point of its bounding box (taken from [2])
## How does CenterTrack work?
### CenterNet
CenterTrack builds on the CenterNet detector presented in [2]. The CenterNet detector represents each object by a single point at the centre of its bounding box (see Figure 1). After representing objects as points, other properties (object size, dimension, etc) are regressed from the image features at the centre location. Object detection now becomes a keypoint estimation problem; a convolutional network takes the image as input and generates a heatmap where each peak represents the centre of the objects [2].  
### CenterTrack
The goal of CenterTrack is to detect objects and maintain a consistent id of this object throughout the different frames. The challenges in this is finding all the objects, including occluded ones in the frame and associating these objects through time [2]. To solve these challenges, two frames are provided to the network as input: the current frame and the prior frame. This can help to recover occluded objects at time t. Additionally, prior detections are given as additional input in a class-agnostic single-channel heatmap.  These two methods provide temporal coherence but an additional method is needed to link these detections across time. An offset prediction between two frames and a greedy matching algorithm provide this association. Figure 2, taken from  [1], represents the framework of CenterTrack showing inputs and outputs. 

![](readme/fig2.png)
Figure 2: Illustration of the CenterTrack framework (taken from [1])
## Reproduction
In terms of the reproducibility project we were tasked with reproducing Table 4 from the paper, specifically the results from the MOT17 dataset. The table shows the results of ablation studies on the proposed framework, in which the two main components of CenterTrack, offset prediction and the use of prior heatmap, are omitted separately and at the same time (detection only). The results are reported in terms of false positive rate (FP), false negative rate (FN), identity switches (IDSW) as well as the multi-object tracking accuracy (MOTA) [3].  

|    MOT17  |
|     |  MOTA     | FP    | FN    | IDSW    |
|--------------|-----------|--------|
|Detection only        | 63.6 %     |  3.5 %    |30.3 %|2.5 %|
|w/o offset       | 65.8 %     |  4.5 %    |28.4 %|1.3 %|
|w/o heatmap        |63.9 %     | 3.5 %   |30.3 %|2.3 %|
|Ours       | 66.1 %  |  4.5 %|28.4 %|1.0 %|

Table 1: Results from the paper which the group had to reproduce

The authors of the paper provide a detailed GitHub page about the paper, which can be found here: GitHub - xingyizhou/CenterTrack: Simultaneous object detection and tracking using center points. The GitHub page is well documented and we therefore used existing code to reproduce the results. All three members of the group used the Google Cloud Platform (GCP) to run the existing code on virtual machines. It is important to note here that because of our unfamiliarity with the GCP platform a lot of time was spent fixing some issues related to the virtual machine. Eventually, all three members of the group managed to successfully reproduce Table 1 with some small variations in the results. 
Using the pretrained model provided by the authors of the paper, we obtained the same results as the authors for the full framework (last row of Table 1) and without offset.  However, for detection only and without heatmap, we obtained slightly different results as shown in Table 2. In general, our results show a lower MOTA, FP and IDSW but a higher FN. This may be due to the use of different hyperparamters such as the tracking threshold or a different version of the model, which were not well-documented by the authors. However, our reproduced results show that although neglecting heatmap leads to lower false positive rate, it increases false negative rate and identity switches, giving a lower MOTA. This qualitative conclusion is consistent with that provided in the paper.



MOTA
FP
FN
IDSW
Detection only
63.1 %
2.4 %
32.4 %
1.8 %
w/o heatmap
63.4 %
2.4 %
32.4 %
1.8 %

Table 2: Reproduced results that differ from the paper
Hyperparameter Check
The authors of the paper trained their model using 4 GPUs and a batch size of 32. However, we only have one Nvidia Tesla K80 GPU and a 30 GB CPU, so to train our own model, we have to reduce the batch size to 8. But with reduced batch size and only one GPU, the training time increases significantly, taking around 30 minutes per epoch. It is thus impractical for us to train the model for 70 epochs as the authors did. Therefore, we would like to investigate how reducing the number of epochs may affect the performance of the trained model.
We train our first model with the same hyperparameters as the authors except from reducing the batch size to 8 and reducing the number of epochs to 30. The learning rate used for this training is 1.25e-4. The MOTA, FP, FN and IDSW on the validation set are shown in the table below. Although the FN and IDSW is similar to the model trained by the authors, the MOTA and FP are noticeably worse than their model. 
Since a smaller batch size means more noise in the gradient, it is preferable to use a smaller learning rate with a smaller batch size. Therefore, we train a second model with the same number of batch size and number of epochs, but with a reduced learning rate of 7.5e-5. With this model, we are able to achieve comparable and even slightly better MOTA, FP, FN and IDSW compared to the model trained by the authors even though our model is trained for less than half the number of epochs used by the authors.


MOTA
FP
FN
IDSW
lr = 1.25e-4
61.0 %
9.8 %
28.2 %
1.0 %
lr = 7.5e-5
66.4 %
4.6 %
28.1 %
1.0 %

Table 3: Hyperparamter check on MOT17 validation set
Ablation Study
As an ablation study we decided to train on the MOT17 dataset without any pre-training on external data. In the paper, the CrowdHuman dataset is used to pretrain the model on static images. The approach is to simulate the previous frame by randomly scaling and translating the current frame creating “hallucinated” motion. This method is so effective that a 52.2 % MOTA is reported in the paper when only training on static images! When training from scratch, a 60.7 % MOTA is documented which means there is almost a 6 percentage point difference when using the static images as pre-training (66.1 %). 
For the same reasons explained in the hyperparameter check section, the ablation study was done on a batch size of 8 and the number of epochs for training was reduced to 5 due to the long training time. This reduction in batch size and number of epochs makes this ablation study different from the one reported in the paper in the Annex. In order to compare the results, we initially trained with pre-training (but using batch size 8 and 5 epochs) and then without pre-training. The results from this study are shown in Table 4. 


MOTA
FP
FN
IDSW
with pre-training
63.9 %
4.9 %
30.1 %
1.0 %
scratch
46.2 %
11.8 %
40.3 %
1.7 %

Table 4: Ablation on MOT17 with and without pre-training
	From the results obtained we can see a significant drop in performance when not using the pre-trained model. This drop in performance is much bigger than the one reported in the Annex of the paper with a 17 % gap in the MOTA performance. We also note a significant increase in false positive rate and false negative rate. This ablation study demonstrates the importance of pretraining on the CrowdHuman dataset especially when the batch size and number of epochs are reduced. 
New Data
In addition to reproducing Table 4 from the paper and an additional ablation study on the MOT17 dataset, we also did some experiments on the KITTI dataset. More specifically, we did a controlled study on output threshold and rendering threshold using the KITTI dataset. 
In CenterTrack, prior detections are rendered as a heatmap, in which only the objects with a confidence score greater than a threshold τ are rendered. And the output threshold θ is essential in all tracking algorithms because MOTA does not consider the confidence of predictions. In the paper, a controlled study on output threshold θ and rendering threshold τ was performed to find the optimal values for the tracking algorithm of CenterTrack. Their results are shown in Table 11 from the paper. The results in Table 11 from the paper were obtained on the MOT17 validation set. To find out if MOT17 and KITTI would give the same choices on the thresholds, we also did the experiments on the KITTI dataset, and the results are shown in Table 5. 
θ
τ
MOTA
IDF1
MT
ML
FP
FN
IDSW
0.4
0.4
88.7%
95.3%
90.3%
2.2%
5.4%
5.8%
0.08%
0.4
0.6
88.1%
94.9%
87.8%
3.6%
3.5%
8.4%
0.08%
0.4
0.5
88.0%
94.9%
88.5%
3.2%
4.2%
7.7%
0.05%
0.3
0.5
88.3%
95.1%
91.0%
1.4%
5.6%
6.1%
0.05%
0.5
0.5
87.7%
94.6%
84.9%
4.7%
3.0%
9.3%
0.02%

Table 5: Controlled study on output threshold and rendering threshold using KITTI dataset. (IDSW values are rounded to two decimal places)
From the table, basically increasing both thresholds results in fewer outputs, thus increases the false negatives while decreases the false positives. We can find a good balance at θ = 0.4 and τ = 0.5, which is the same conclusion as in the paper based on the MOT17 validation set. 

Conclusion
In this project, we reproduced the results in the paper “Tracking Objects as Points”, showing how using heatmaps of previous timestep and offsets help improve the accuracy of object tracking. We also performed hyperparameter check and ablation study for training under limited resources and showed that similar accuracy can be achieved as the model trained by the authors and pretraining with CrowdHuman dataset is even more important. In addition, we also demonstrated the effect of optimising output and rendering thresholds for a different dataset. 

Contributions
Matti: reproducing results, ablation study
Lydia: reproducing results, hyperparameter check
Yushan: reproducing results, new data

References
[1] Zhou, X., Koltun, V., & Krähenbühl, P. (2020). Tracking Objects as Points. In Computer Vision – ECCV 2020 (pp. 474–490). Springer International Publishing. https://doi.org/10.1007/978-3-030-58548-8_28
[2] Zhou, X., Wang, D., Krahenb ¨ uhl, P.: Objects as points. arXiv:1904.07850 (2019)
[3] Leal-Taix´e, L., Milan, A., Schindler, K., Cremers, D., Reid, I., Roth, S.: Tracking the trackers: an analysis of the state of the art in multiple object tracking. arXiv:1704.02781 (2017)

Acknowledgements
We would like to thank our TA Xin Liu for guiding us in this project.
