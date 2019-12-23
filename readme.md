## Introduction

Deep Learning (DL) algorithms are increasingly used in products and services that affect people’s lives. Examples of such algorithms can be found within recruitment, bank loans, and court decisions. As the implementation of Deep Learning increases, the requirements for fair and unbiased algorithms become ever more relevant. The emerging field of algorith- mic fairness investigates such issues by providing metrics to measure if an algorithm is discriminating or not. Unfairness is rooted in the data used for model training, which again can be rooted in the way the data was generated, gathered or labeled. Assessing fairness thus requires a thorough examination of the input data as the problem is not specifically emerging from the Deep Learning methods themselves. Furthermore, data is often collected by humans which pose an even bigger and more fundamental problem, namely, whether peoples’ inherent biases are included in these potentially high impact algorithms. 

## Code
#### Folders
The Data folder holds all necessary datasets to reconstruct the training of the models, except for the pictures itself. train.csv, valid.csv and test.csv are used to train and evaluate the model on specific attributes. These are constructed from the two txt-files, list_attr_celeba.txt and list_eval_partition.txt. The pictures can be found on CelebA’s official Google Drive folder under Aligned Images: (link)

The data_utils folder holds important utility functions used when training the model. data_utils_celeba_pytorch5.py is the dataloader used to construct the batches on which the model is training. The network_tuning_valid.py holds utility functions for creating the model architecture, training the model, running validation, etc. These functions are called when training the model. network_tuning_valid_copy.py is the script used when training the model with a weighted loss function to alleviate the identified biases. 

models contains all the models which has been trained on Amazon Web Services (AWS) with dataframes for each run containing all results for every model included in the run. This means, that results for all eight models saved in run2 is summarized in the dataframe called run2_df.pkl. 

tuning comprise all the scripts that has been used in the different training rounds, i.e. the ones contained in the models folder under aws_models. Basically, each script runs the training rounds for the different models in each round, and prints all results for each model into a dataframe, which is also saved as a pickle file in the end. 


#### Scripts

## Contributors
* Martin Johnsen ([mar10nJohns1](https://github.com/mar10nJohns1))
* Charlotte Theisen 
* Aleksander Pratt ([AleksanderPratt](https://github.com/AleksanderPratt))
