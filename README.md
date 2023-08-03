# FlexiFed

Implementation of paper - [FlexiFed: Personalized Federated Learning for Edge Clients with Heterogeneous Model Architectures](https://dl.acm.org/doi/10.1145/3543507.3583347)

## Requirement

- Python 3.8 (ubuntu20.04)
- Pytorch 2.0.0
- Cuda 11.8
- GPU RTX 3080

## Structure

The structure of the project is shown below :

```bash

```

- src/model : 

  > - VGG.py : There are four models in VGG-family - VGG-11, VGG-13, VGG-16, VGG-19
  > - ResNet.py : There are four models in ResNet-family - ResNet-20, VGG-32, VGG-44, VGG-56
  > - CharCNN.py : There are four models in CharCNN-family - CharCNN-3, CharCNN-4, CharCNN-5, CharCNN-6
  > - VDCNN.py : There are four models in VDCNN-family - VDCNN-9, VDCNN-17, VDCNN-29, VDCNN-49

  

- src/utils :

  > - Dataset.py : There are some necessary functions to get the dataset (`get_dataset`)  and split the dataset by index (`get_idx_dict`) . The relevant dataset includes : CIFAR-10, CINIC-10, Speech-Commands, AG-News
  > - Visualization : There are functions to visualize the result of the FL schemes and two figures will be created : the convergence process of models in the same family with different version under the same aggregation strategy, the convergence process of models in the same family with same version under different aggregation strategy

- src/core:

  > - Clients.py : There is an important class to simulate the clients in edge, which has the ability to train and test locally
  > - FlexiFed.py : The FlexiFed frame named ParamserServer, which has a client-set to simulate the clients in the FL System, has three aggregation strategies (Basic-Common, Clustered-Common, Max-Common) and a function to train clients globally
  > - main.py : The main function to run the FL System, you can run different system by setting the parameters : num_clients (the number of the clients in FL System), family_name (the model family), dataset_name, communication_round (the rounds of global training), strategy (the aggregation strategy)
  
- result : There are the .csv files that record the convergence process, includes : VGG-CIFAR-10, VGG-CINIC-10, VGG-Speech-Commands, ResNet-CIFAR-10, ResNet-CINIC-10, ResNet-Speech-Commands, CharCNN-AG-NEWS, VDCNN-AG-NEWS

- final result : There are .docx and .pdf files recording the accuracy table of the final result.

- img : There are relevant images for this project.

 

## Results

I will show you the results of federated learning for models with architecture heterogeneity on different datasets. Each result includes three pictures ( the convergence process under different strategy and different model, the final accuracy ) and one table ( the final accuracy of the model trained by the FL system).

We assume that the number of clients in FL system is n, the accuracy of version v model is Acc_v, then there is :


$$
Acc_v=\frac4n\sum_{i=1}^{n/4}Acc_{v-i}
$$

- VGG-family on CIFAR-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CIFAR_10_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CIFAR_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6936 | 0.7480 | 0.7084 | 0.7364 |
  |   Clustered-FL   | 0.7848 | 0.8068 | 0.7960 | 0.8152 |
  |   Basic-Common   | 0.7136 | 0.7368 | 0.7184 | 0.7152 |
  | Clustered-Common | 0.7560 | 0.8000 | 0.7968 | 0.7892 |
  |    Max-Common    | 0.7680 | 0.8428 | 0.8640 | 0.8428 |
  
  <div align=center>
  <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CIFAR_10_3.png" width=654 height=234/>
  </div>
  
- VGG-family on CINIC-10 :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CINIC_10_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CINIC_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6319 | 0.6730 | 0.6434 | 0.6561 |
  |   Clustered-FL   | 0.7111 | 0.7404 | 0.7292 | 0.7331 |
  |   Basic-Common   | 0.6129 | 0.6619 | 0.6379 | 0.6225 |
  | Clustered-Common | 0.6895 | 0.7317 | 0.7420 | 0.7228 |
  |    Max-Common    | 0.6772 | 0.7772 | 0.7892 | 0.7874 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CINIC_10_3.png" width=654 height=234/>
  </div>

- VGG-family on Speech Commands :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_Speech_Commands_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_Speech_Commands_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.9207 | 0.9385 | 0.9077 | 0.9184 |
  |   Clustered-FL   | 0.9516 | 0.9468 | 0.9681 | 0.9575 |
  |   Basic-Common   | 0.9219 | 0.9385 | 0.9055 | 0.9126 |
  | Clustered-Common | 0.9373 | 0.9563 | 0.9587 | 0.9598 |
  |    Max-Common    | 0.9327 | 0.9787 | 0.9764 | 0.9764 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_Speech_Commands_3.png" width=654 height=234/>
  </div>

- ResNet-family on CIFAR-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CIFAR_10_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CIFAR_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6764 | 0.7044 | 0.6884 | 0.7052 |
  |   Clustered-FL   | 0.7824 | 0.7604 | 0.7796 | 0.7624 |
  |   Basic-Common   | 0.7156 | 0.7164 | 0.7000 | 0.6908 |
  | Clustered-Common | 0.7856 | 0.7672 | 0.7888 | 0.7752 |
  |    Max-Common    | 0.7828 | 0.7944 | 0.7944 | 0.7956 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CIFAR_10_3.png" width=654 height=234/>
  </div>

- ResNet-family on CINIC-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CINIC_10_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CINIC_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6190 | 0.6470 | 0.6402 | 0.6360 |
  |   Clustered-FL   | 0.7014 | 0.7067 | 0.7156 | 0.7162 |
  |   Basic-Common   | 0.6525 | 0.6385 | 0.6528 | 0.6539 |
  | Clustered-Common | 0.7066 | 0.7119 | 0.7300 | 0.7249 |
  |    Max-Common    | 0.7233 | 0.7252 | 0.7283 | 0.7281 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CINIC_10_3.png" width=654 height=234/>
  </div>

- ResNet-family on Speech Commands :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_Speech_Commands_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_Speech_Commands_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3    |   V4   |
  | :--------------: | :----: | :----: | :-----: | :----: |
  |    Standalone    | 0.9101 | 0.9160 | 0.8723  | 0.9031 |
  |   Clustered-FL   | 0.9326 | 0.9480 | 0.9350  | 0.9480 |
  |   Basic-Common   | 0.9031 | 0.9054 | 0.9243  | 0.9291 |
  | Clustered-Common | 0.9551 | 0.9445 | 0.94325 | 0.9303 |
  |    Max-Common    | 0.9374 | 0.9563 | 0.9610  | 0.9433 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_Speech_Commands_3.png" width=654 height=234/>
  </div>

- CharCNN-family on AG News :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/CharCNN_AG_NEWS_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/CharCNN_AG_NEWS_2.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/CharCNN_AG_NEWS_3.png" width=654 height=234/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.2589 | 0.6937 | 0.7095 | 0.7589 |
  |   Clustered-FL   | 0.2500 | 0.8253 | 0.7400 | 0.7490 |
  |   Basic-Common   | 0.7990 | 0.8579 | 0.8532 | 0.8326 |
  | Clustered-Common | 0.8021 | 0.8711 | 0.8232 | 0.8342 |
  |    Max-Common    | 0.6568 | 0.8705 | 0.8547 | 0.8590 |

- VDCNN-family on AG News :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VDCNN_AG_NEWS_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VDCNN_AG_NEWS_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.8215 | 0.8205 | 0.7968 | 0.8279 |
  |   Clustered-FL   | 0.8584 | 0.8384 | 0.8395 | 0.8332 |
  |   Basic-Common   | 0.8305 | 0.8063 | 0.7284 | 0.8347 |
  | Clustered-Common | 0.8500 | 0.8258 | 0.8621 | 0.8269 |
  |    Max-Common    | 0.8558 | 0.8416 | 0.8216 | 0.8679 |

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VDCNN_AG_NEWS_3.png" width=654 height=234/>
  </div>

## Comparation

I will compare my results with the result in Table1 of the paper and analyze the similarities and differences :

<div align=center>
  <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/Table.png" width=853 height=555/>
</div>

- Similarities :

  Although the differences between the various aggregation strategies are not obvious in the convergence process figures, the value relationship for accuracy is almost the same as that in Table1, which satisfies :

  
$$
Basic-Common < Clustered-Common < Max-Common
$$

$$
Standalone\approx Basic-Common
$$

$$
Clustered-FL\approx Clustered-Common
$$

  This demonstrates the usefulness of FlexiFed. The idea "aggragate the model in FL system to the maximum extent" does work and the gains from Max-Common strategy are significant.

- Differences :

  - The accuracy of ResNet family on CIFAR-10 is slightly lower than Table1's : 

    I think this is due to the lack of communication rounds. When training to 70 rounds, the accuracy is just stable, we can see that there is still a small upward trend. ResNet family needs more communication rounds than VGG family to get higher accuracy.

  - The accuracy of models on Speech Commands under the Standalone, Clustered-FL, Basic-Common strategy is much higher than Table1's : 

    I think this is due to the data augmentation before local training. I use the STFT to transform the 1d audio to 2d matrix, so that we don't have to adjust the model and this operation contribute much to the higher accuracy.

  - The accuracy of CharCNN family on AG NEWS is slightly lower than Table1's : 

    CharCNN can easily fall into the local minima ( 25%, 50%, 75% ) during the convergence process especially the CharCNN-3 under the Standalone strategy. I think this is due to the presence of xor-like operations during training, which will bring the difficulty for models with simple architechiture to converge normally. But I found that the aggregation strategy of FlexiFed will help the model to escape the local minimum when there is a  well convergent model in the FL system, which confirms the usefulness of federated learning and FlexiFed again.

  - The convergence process of VDCNN is oscillating :

    I think that's inherent in the model, because the convergence process of the model is oscillating even under the standalone strategy. Although the higher the degree of aggregation of the model, the more obvious the oscillation, it does not affect the final high accuracy of the model.

## References

Here are some repositories and articles that I refer to during the implementation of the project :

- Project Structure : 

  [fio1982/FlexiFed (github.com)](https://github.com/fio1982/FlexiFed)

- The architecture of ResNet : 

  [pytorch_resnet_cifar10: Proper implementation of ResNet-s for CIFAR10/100 in pytorch that matches description of the original paper.](https://github.com/akamaster/pytorch_resnet_cifar10)

- Loading of Speech Commands dataset : 

  [pytorch-speech-commands: Speech commands recognition with PyTorch | Kaggle 10th place solution in TensorFlow Speech Recognition Challenge ](https://github.com/tugstugi/pytorch-speech-commands)

- The architecture of CharCNN : 

  [charcnn-classification: Character-level Convolutional Networks for Text Classification in Pytorch](https://github.com/Sandeep42/charcnn-classification)

- The architecture of VDCNN : 

  [Very-deep-cnn-pytorch: Very deep CNN for text classification](https://github.com/uvipen/Very-deep-cnn-pytorch)

- Loading of AG News dataset : 

  [pytorch-char-cnn-text-classification: A pytorch implementation of the paper "Character-level Convolutional Networks for Text Classification"](https://github.com/cswangjiawei/pytorch-char-cnn-text-classification)

