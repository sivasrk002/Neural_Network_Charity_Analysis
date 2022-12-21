# Neural Network Charity Analysis

## Overview

### Introduction

Knowing if a donation recipient will prove a successful investment or not is invaluable to a not-for-profit organization like Alphabet Soup. Using a dataset of previous investments by Alphabet Soup, this repository designs and trains a neural network to help predict the outcomes of future potential recipients. With this tool in hand, not-for-profit organizations like Alphabet Soup can make more informed decisions in the future.

### Resources

This repository employs the following Python libraries:

* TensorFlow (neural networks)
* pandas (dataframes)
* sk-learn (machine learning models and preprocessing)


## Results 

### Data Preprocessing

* The target variable in this model is the "IS_SUCCESSFUL". This is a binary categorical feature (represented as either 0 or 1), that determines whether Alphabet Soup found their investment in that company was well used.

* The feature variables in this model are "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "ORGANIZATION", "INCOME_AMT", "ASK_AMT", "STATUS", "SPECIAL_CONSIDERATIONS" and "USE_CASE".

* "EIN" and "NAME" are considered identification variables that are neither targets nor features.


### Compiling, Training, and Evaluating the Model

![model summary](https://github.com/juberr/Neural_Network_Charity_Analysis/blob/main/pics/model_summ.png?raw=true)

* The model designed here consists of four layers:

    * The first layer takes in 35 input nodes(the number of feature columns after encoding ), has 105 hidden nodes, and uses the "reLU" activation function. The model is currently set up to have 3 times more nodes than the input layer, as this is a good rule of thumb for neural network design. The "reLu" activation function has a proven track record against the vanishing gradient problem that the sigmoid and tanh activation functions suffer from.

    * The second layer has 40 hidden nodes and uses the "reLU" activation function. To avoid over fitting the model, the number of nodes is significantly reduced per layer, this ensures the model won't get caught up in small patterns it may have noticed, and keep its sights on the larger picture.  

    * The third layer has 20 hidden nodes and uses the "softmax" activation function. When experimenting with different activation functions it was found that having the second last layer use the "softmax" activation function consistently yields a higher average accuracy rating on the test data. The "softmax" activation feature helps generalize the data from the "reLU" functions into more logistic friendly values.

    * The final and fourth layer has 1 output node and uses the "tanh" activation function. This is the function that returns the likely hood that any given input is considered to be successful or not. This is the appropriate function since the question at hand is one of binary classification.

* The model did not achieve the target model of performance of 75%, the highest achieved hovers around 73%.

* The steps taken to try and increase the model's performance are as follows:

    * Removed noisy or mute variables that did not offer good information to the neural network. These variables include:
        * "STATUS" - The status variable consisted of mostly 1s (which mean the organization is currently active), which does not tell the neural network a lot about the organization.

        * "SPECIAL_CONSIDERATIONS" - A organization given special considerations, and what those special considerations were, did not have a high correlation with the target variable of "IS_SUCCESSFUL".

        * "USE_CASE" - This variable did not give the neural network much information because there was a lack of variation in the data set. A vast majority of the responses for this variable were for "preservation".

    ![affil density](https://github.com/juberr/Neural_Network_Charity_Analysis/blob/main/pics/density.png?raw=true)

    * Much like "CLASSIFICATION" and "APPLICATION_TYPE" binning was performed on the "AFFILIATION" column, since the vast majority of responses in that column were either "Independent" or "CompanySponsored".

    * The number of neurons per layer were increased.

    * The number of layers were increased.

    * The types of activation functions were changed for the second last and output layers ("softmax" and "tanh").

    # Summary

    ![model accuracy](https://github.com/juberr/Neural_Network_Charity_Analysis/blob/main/pics/model_acc.png?raw=true)

    The deep learning model achieved an accuracy of 73% on the test set of data. Should Alphabet Soup want to continue with a neural network to predict the success of potential investments, it is recommended that more data be collected. The dataset used here did not provide any particularly high correlations to the target variable.

    More features should be collected such as how long the organization has been in operation for, and how many previous donations they've received in the past (from Alphabet Soup or otherwise).

    # Recommendation

    ![rfc accuracy](https://github.com/juberr/Neural_Network_Charity_Analysis/blob/main/pics/rfc.png?raw=true)

    Alternatively, Alphabet Soup could employ the use of other machine learning models. For example, the Random Forest Classifier (RFC) from sklearn can achieve a similar accuracy without being as computationally expensive. The RFC model displayed here only uses 128 estimators with a max depth of 9. This is much smaller than the total 8,961 parameters performing calculations in the neural network!
