## Introduction

Deep Learning (DL) algorithms are increasingly used in products and services that affect people’s lives. Examples of such algorithms can be found within recruitment, bank loans, and court decisions. As the implementation of Deep Learning increases, the requirements for fair and unbiased algorithms become ever more relevant. The emerging field of algorith- mic fairness investigates such issues by providing metrics to measure if an algorithm is discriminating or not. Unfairness is rooted in the data used for model training, which again can be rooted in the way the data was generated, gathered or labeled. Assessing fairness thus requires a thorough examination of the input data as the problem is not specifically emerging from the Deep Learning methods themselves. Furthermore, data is often collected by humans which pose an even bigger and more fundamental problem, namely, whether peoples’ inherent biases are included in these potentially high impact algorithms. 

## Code
#### Folders
The _Data_ folder holds all necessary datasets to reconstruct the training of the models, except for the pictures itself. _train.csv, valid.csv_ and _test.csv_ are used to train and evaluate the model on specific attributes. These are constructed from the two txt-files, _list_attr_celeba.txt_ and _list_eval_partition.txt_. The pictures can be found on CelebA’s official Google Drive folder under Aligned Images: [CelebA dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?usp=sharing)

The _data_utils_ folder holds important utility functions used when training the model. _data_utils_celeba_pytorch5.py_ is the dataloader used to construct the batches on which the model is training. The _network_tuning_valid.py_ holds utility functions for creating the model architecture, training the model, running validation, etc. These functions are called when training the model. _network_tuning_valid_copy.py_ is the script used when training the model with a weighted loss function to alleviate the identified biases. 

_models_ contains all the models which has been trained on Amazon Web Services (AWS) with dataframes for each run containing all results for every model included in the run. This means, that results for all eight models saved in _run2_ is summarized in the dataframe called _run2_df.pkl_. 

_tuning_ comprise all the scripts that has been used in the different training rounds, i.e. the ones contained in the _models_ folder under _aws_models_. Basically, each script runs the training rounds for the different models in each round, and prints all results for each model into a dataframe, which is also saved as a pickle file in the end. 


#### Scripts

_Project_ calls the dataloader function and splits the dataset into train,validation and test. Then it defines the model and trains the model (happens on AWS). The trained model is loaded and the model is tested for bias against the remanining 39 attributes. 

_mean_accuracy_in_40_attributes_ calculates the mean accuracy over all the 40 attributes in the dataset. Each variables is used as a target variable.

_training_with_weigthed_loss implements a new loss function which adds a weight to the loss function in the training resulting in higher punishment for wrong predictions of under represented groups

_Upsampling_ contains the second bias alleviation method, upsampling, where pictures from underepresented groups are upsampled to the most presented group. Furthermore, data augmentation are added to the pictures. 

## Contributors
* Martin Johnsen ([mar10nJohns1](https://github.com/mar10nJohns1))
* Charlotte Theisen 
* Aleksander Pratt ([AleksanderPratt](https://github.com/AleksanderPratt))
