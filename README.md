[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/58HShPQN)
# Comprehensive exploration of Wine Dataset and Fruit Classification
## Group Members

1. Parthavi Vaghani  [@Parthvi579](https://github.com/Parthvi579)
2. Vikas Chaudhary [@VikasChaudhary123](https://github.com/VikasChaudhary123)
3. Kaushik Lathiya [@Kaushik0704](https://github.com/Kaushik0704)
4. Augustin Lobjoie [@Augustinlobjoie](https://github.com/Augustinlobjoie) H00450696

## Initial Project Proposal
This project aims to use two datasets as mentioned below. 
1. **Wine dataset**
   - To proprocess, explore, get insights, show visualizations.
   - Clustering including all the features, some specific features. Verification using Elbow method and Silhouette score.
   - Decision trees to predict wine type and wine quality
   - Results after applying these techniques can be used by wine makers. 
3. **Fruit dataset of images**
     - Neural network and Convolutional neural network to predict fruit type from images.
     - Applicaiton of image classification is in the field of agriculture. There are hundreds of varities of apple and other fruits, making it impossible to remember for a human being. Applications can be build to identify species of fruits, same technique can be applied to detect diseases in fruits or vegetables. 

## Project Planning
We'll make use of GitHub's collaborative features, such as issue tracking, version control, and a dedicated wiki, to organise project tasks and documentation in a way that will facilitate effective development and teamwork.
- **Click to view -** [Wiki](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki)
  
### Research objectives
The objectives of this project are:
> Apply clustering techniques on wine data based on quality and chemical properties.
> Identify key features contributing to the wine type and best wine quality using decision tree algorithms.
> Develop effective image classification models for fruit recognition using neural networks and CNNs.
> Assess model performance using metrics like accuracy, precision, recall, and F-score.

### Milestones
> Milestone of this project.

| Task                                      | Start Date | End Date   | Duration | Week |
|-------------------------------------------|------------|------------|----------|---------|
| 1. Define Project Topic | 18/09/2023 | 24/09/2023 | 7 days   | [Week-2](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week2.md) |
| 2. Find Dataset related to the topic                      | 25/09/2023 | 01/10/2023 | 7 days   | [Week-3](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week3.md) |
| 3. Data Visualisation and Analysis                | 02/10/2023 | 08/10/2023 | 7 days   | [Week-4](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week4.md)
| 4. Clustering               | 09/10/2023 | 15/10/2023 | 7 days   | [week-5](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week5.md) |
| 5. Conclusion of Clustering                    | 16/10/2023 | 22/10/2023 | 7 days   | [Week-6](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week6.md) |
| 6. Decision Tree                        | 23/10/2023 | 29/10/2023 | 7 days   | [Week-7](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week7.md) |
| 7. Conclusion of Decision Tree           | 30/10/2023 | 05/11/2023 | 7 days   | [Week-8](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week8.md) |
| 8. Neural Network               | 06/11/2023 | 12/11/2023 | 7 days   | [Week-9](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week9.md) |
| 9. Conclusion of Neural Network                          | 13/11/2023 | 19/11/2023 | 7 days   | [Week-10](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week10.md) |
| 10. Final Documentation and Presentation                        | 20/11/2023 | 26/11/2023 | 7 days   | [Week-11](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki/Week11.md) |


## Findings Report

<!-- Below you should report all of your findings in each section. You can fill this out as the project progresses. -->
Some of the findings are listed below.

**1. Wine Dataset**
  - _**Data exploration and visualization**_ - No missing values in dataset, strong correlation between features(free sulfur and total sulfur) so one of them can be dropped. It was observed that wine quality is dependent on alcohol content and acidity.  
  - _**Clustering**_ - Including all the features could not conclude to optimal number of clusters(No clear elbow point with elbow method, low silhouette score). Clustering with selected features resulted into better silhouette score and clear elbow point suggesting k=3 number of clusters.  
  - _**Decision Trees**_ - For Wine type prediction, model was suspected to overfitting so we did k-fold cross validation, regularisation(pruning), after this model is fitting well giving good accuracy. For wine qualiity, there was significat difference between test and training accuracy, show overfitting. Even after regularisation, we could not fit the model for better accuracy for wine quality.
    
**2. Fruits image dataset**
  - **_Artificial Neural Networks(ANN)_** - We implemented single layer network(around 84 percent accuracy), with 3 hidden layers(92 % accuracy). Adding layers helped to increase the accuracy. 
  - **_Convolutional neural networks(CNN)_** - This gave around 96 percent accuracy with two hidden layers, depecting that CNN works better than ANN in our case. 

### Datasets
1.  **Wine Quality Dataset:**
    
    -   **Source:** UCI Machine Learning Repository - [Wine Quality](https://doi.org/10.24432/C56S3T)
    -   **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
    -   **Ethical Considerations:** No privacy or logistic issues, and the dataset is openly shared for any purpose with appropriate credit given. 
    
2.  **Fruit 360 Dataset:**
    
    -   **Source:** Kaggle - [Fruit 360](https://www.kaggle.com/datasets/moltean/fruits)
    -   **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
    -   **Ethical Considerations:** No privacy or logistic issues with this dataset as the dataset contains only images of fruits without any human or brand-related aspects.

### Dataset description

##### Wine Quality Dataset
The Wine Quality Dataset is a dataset that contains information about the chemical properties of red and white wine, example below shows the information about the attributes and the data of this dataset.
  
##### Fruit 360 Dataset
Fruits 360 dataset is a dataset that contains images of different fruits with 100 * 100 pixel dimension . These images cover different types of fruits. 

We have only included some fruits(with all images of each fruit) as the whole dataset was taking a lot of memory(to load) and time(to train). We have included only one variety of apple(Apple Braeburn) and only fruits with starting letter from A to D.
Link for sub dataset is [Fruit-360](https://heriotwatt-my.sharepoint.com/personal/pv2008_hw_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW%2Felbow%2Epng&parent=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW)


### Dataset examples
<!-- Add a couple of example instances and the dataset format -->

##### Wine Quality Dataset
The data is typically organized in a tabular format, with rows representing individual wine samples and columns for each attribute.
| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | quality | wine_name |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
| 7.4	 | 0.70 | 0.00 | 1.9 | 0.076 | 11.0 | 34.0 | 0.9978 | 3.51 | 0.56 | 9.4	 | 5 | red |
| 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | 25.0 | 67.0 | 0.9968 | 3.20 | 0.68 | 9.8 | 5 | white |


##### Fruit 360 Dataset
The dataset primarily consists of fruit images with dimention of 100x100 pixels, with each image corresponding to a specific class and different varieties of the same fruit are stored as separate classes.
| Example | 1                                       | 2                                                                     |
|---------|-----------------------------------------|-----------------------------------------------------------------------|
| Image   | [Cavendish](.../Banana/100_100.jpg)     | [Granny Smith](.../Apple%20Granny%20Smith/323_100.jpg)              |
| Class   | Banana                                  | Apple                                                                 |
| Size    | 100x100 pixels                          | 100x100 pixels                                                        |
| Variety | [e.g., Cavendish]                       | [e.g., Granny Smith]                                                  |

### Dataset Size and Structure
<!-- What is the size of the dataset? -->
##### Wine Quality Dataset
- The dataset consists of 6497 entries with 13 features, including numeric type and a categorical feature for wine name.
- The dataset is divided into 1599 entries for Red wine and 4898 entries for white wine.
  
##### Fruit 360 Dataset
- The original Fruits 360 Dataset includes a total of 90,483 images.
- The original dataset is divided into the following subsets:
  - Training set: 67,692 images, each featuring a single fruit.
  - Test set: 22,688 images, each showcasing a single fruit.
- There are 131 distinct classes in original dataset,  representing different fruits and different types of the same fruit are treated as separate classes, ensuring a variety of fruits.
- **For our model, we are using 28 distinc classes of fruits so our training and testing dataset contains less images than origial dataset.**
  
### Dataset Summary Statistics
<!-- Summary statistics of your dataset -->

|               | Fixed Acidity | Volatile Acidity | Citric Acid | Residual Sugar | Chlorides | Free Sulfur Dioxide | Total Sulfur Dioxide | Density | pH   | Sulphates | Alcohol | Quality |
|-----------------------|---------------|------------------|-------------|----------------|-----------|---------------------|----------------------|---------|------|-----------|---------|---------|
| Count                 | 6497          | 6497             | 6497        | 6497           | 6497      | 6497                | 6497                 | 6497    | 6497 | 6497      | 6497    | 6497    |
| Mean                  | 7.215         | 0.340            | 0.319       | 5.443          | 0.056     | 30.525              | 115.745              | 0.995   | 3.219| 0.531     | 10.492  | 5.818   |
| Std Dev               | 1.296         | 0.165            | 0.145       | 4.758          | 0.035     | 17.749              | 56.522               | 0.003   | 0.161| 0.149     | 1.193   | 0.873   |
| Min                   | 3.800         | 0.080            | 0.000       | 0.600          | 0.009     | 1.000               | 6.000                | 0.987   | 2.720| 0.220     | 8.000   | 3.000   |
| 25th Percentile      | 6.400         | 0.230            | 0.250       | 1.800          | 0.038     | 17.000              | 77.000               | 0.992   | 3.110| 0.430     | 9.500   | 5.000   |
| Median                | 7.000         | 0.290            | 0.310       | 3.000          | 0.047     | 29.000              | 118.000              | 0.995   | 3.210| 0.510     | 10.300  | 6.000   |
| 75th Percentile      | 7.700         | 0.400            | 0.390       | 8.100          | 0.065     | 41.000              | 156.000              | 0.997   | 3.320| 0.600     | 11.300  | 6.000   |
| Max                   | 15.900        | 1.580            | 1.660       | 65.800         | 0.611     | 289.000             | 440.000              | 1.039   | 4.010| 2.000     | 14.900  | 9.000   |

### Dataset Visualisation
<!-- Visualisations of your dataset -->

**Click to view :**  [Visualisation Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/WineDataViz.ipynb)

<img width="637" alt="correlation matrix" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/c9bc92a0-8cfa-485c-a5e3-d953a77b1489">


**Correlation matrix for wine features**



<img width="586" alt="scatter plot, alcohol and density" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/1c2362e8-73c1-4d28-b504-76a5b4f9c4d0">

### Dataset Analysis
<!-- Analysis of your dataset -->

- There is no missing value in this dataset to handle 
- Identified outliers in certain features in the dataset.
- Fixed acidity: The mean and median are close, suggesting that the data is symmetric and that there is no strong skewness. The max value is 15.9.
- Residual sugar: The mean is 5.44, the median is 3, and the max value is 65.8, indicating the presence of outliers and right skewness. The standard deviation is high, indicating a variety of sugar levels in different wines.
- Total sulfur dioxide and free sulfur dioxide: The standard deviation is significant, indicating high variability. The max values are high, so there are potential outliers.
- Alcohol: The mean of 10.49 and the median of 10.3 are quite close. The max is 14 and the min is 8.
- Quality of wine lies between (3-9), but 93% of wines are between 5-7 quality, and 99.5% wines have a quality between 4 to 8.
- The scatterplot suggests that density and alcohol content in the wine (both red and white) are somewhat negatively correlated. The correlation matrix also shows that alcohol and density are negatively correlated with a value of -0.6867.
- There are positive correlations between density and residual sugar (0.55), density and fixed acidity (0.46), and alcohol and quality (0.44).
- Total sulfur dioxide and free sulfur dioxide have a very strong correlation (0.72), so only one of these features should be selected during feature reduction.
- Wines with higher fixed acidity, alcohol content, and lower volatile acidity tend to be of higher quality.

### Clustering 
The mixed dataset of wine, a combination of the "Red Wine Dataset" and "White Wine Dataset," was utilized for implementing clustering algorithms. The primary objective was to observe clear clusters corresponding to the 9 wine quality levels. Initially, all features of the dataset were considered as input to determine the best clusters for predicting wine quality. Subsequently, after implementing various approaches, a selection of features from the dataset, specifically pH and Alcohol, was selected to achieve more efficient clustering results.

#### Experimental Design
<!-- Describe your experimental design and choices for the week. -->
We consider two types of Clustering Algorithms as an experiment analysis of clustering on the Wine dataset.

**Click to view :**
 - [K Means Algorithm Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/K_Means_Clustering_withScore.ipynb) 
- [EM Algorithm Notebook]( https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/EM.ipynb)

1.  **K-means Clustering:**  
	-  **9-cluster K-means Clustering with all features:** Initially, clustering was attempted with 9 clusters representing the wine quality levels. We used all the feature of the dataset as an input and also calculated homogeneity score and completeness score for this cluster. We implimented Elbow Method and silhouette score to identify optimum number of cluster for this dataset.
	-  **4-cluster K-means Clustering with all Features:**  As we got highest value of silhouette score for 4 number of cluster in initial implimentation stage , we applied 4 cluster K-means clustering including all the feature of the dataset and we also calculated homogeneity score and completeness score for this cluster.We again implimented Elbow method and silhouette score for this cluster as well.
	- **3-cluster K-means clustering with selected Features:** By considering result from above two cluster selection we moved to Feature selection for getting batter quality of cluster so, Feature selection of pH and alcohol was performed based on a correlation matrix and pair plot analysis. Clustering was then performed using K-means with 3 clusters on the selected features (pH and alcohol). We perform elbow method and silhouette score with homogeneity score and completeness score for this cluster. 
2.  **Expectation-Maximization (EM) Algorithm:**  - We observed joinplots for 'fixed acidity' vs. 'pH,' 'fixed acidity' vs. 'alcohol,' 'pH' vs. 'alcohol,' and 'sulphates' vs. 'alcohol' to assess the efficiency of clustering where we could try to extract different cluster that would help us predict the quality of a wine. The EM algorithm was applied to the dataset, considering 'alcohol', 'pH,' and 'fixed acidity' features. Those features offers a good cluster separation according to Quality. 

#### Results
<!-- Tables showing the results of your experiments -->
The following **Score Table** shows the key results of the clustering analysis:
| Clustering Technique                        | Homogeneity Score | Completeness Score | Optimal k (Elbow Method) | Average Silhouette Score |
|--------------------------------------------|-------------------|--------------------|--------------------------|---------------------------|
| 9-cluster K-means Clustering (All Features) | 0.41              | 0.24               | Not Clear                | 0.24537          |
| 4-cluster K-means Clustering (All Features) | 0.1857            | 0.0974             | Not Clear                | 0.2453                    |
| 3-cluster K-means Clustering (Selected Features) | 0.1448       | 0.0398             | 3                        | 0.85                      |

- Hyperparameter variations were explored for the number of clusters (k) in the K-means algorithm.
- Variations helped in selecting an appropriate number of clusters based on the silhouette score.

- **Elbow Method** for selecting optimal number of cluster.
<img width="786" alt="elbow" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/58567233-7edc-4408-a0c8-e0a778117028">

- **Silhouette coefficient Method** for selecting appropriate number of cluster
<img width="834" alt="silhoutte score" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/2c17c1c6-7e97-4534-afc1-199fa0c56c1e">

**EM Algorithm Result:** Mean quality of clusters ranged from 5.2 to 6.5, indicating inefficiency. The proportion of qualities is the same as in the global data set. This algorithms is not suited for Quality predicition.
- **Joinplots** to access efficiency of cluster
<img width="573" alt="joinplot" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/1e12f92b-45ef-44f3-8b4e-ef3775e7d713">


#### Discussion

<!-- A brief discussion on the results of your experiment -->
The K-Means and EM algorithms were used in an iterative manner for clustering. While feature selection based on joint plots improved interpretability, the EM algorithm did not demonstrate efficiency in clustering data and predicting wine quality. In K-Means, The initial attempt of 9 clusters aligned with the diverse quality levels did not produce meaningful results. 


After that, we observed that pH and alcohol feature selection significantly improved the accuracy of the clustering model. 
Validation with Elbow and Silhouette score method indicated appropriate number of clusters, Silhouette score came 0.85(maximum) for k = 3 and Elbow method also suggested K=3(number of clusters).In the end, 0.14 was Homogeneity score and 0.0398 was Completeness score. 

### Decision Trees

In Wine dataset, each row in the dataset represents a specific wine sample, and the features serve as inputs for the decision trees. 
We have implemented two decions trees, the output labels represent the wine type and quality of the wine respectively for these trees.
Also random forest has been implemented for wine quality as output label. 

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->

We consider two types of Decision tree as an experiment analysis on the dataset

**Click to view :**
  - [Decision Tree Classification Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreePara.ipynb) 
  - [Random Forest Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreeQuality.ipynb)

1.  **Decision Tree Classification:**
    In our exploration of decision tree models for wine type and wine quality prediction, we started with all available features. The initial decision tree showed impressive accuracy,     
    reaching 99.97% on the training set and 99.33% on the test set. However, we were worried that the model might be overfitting to the training data.
    We tried to address this by using K-fold cross-validation and adjusting the parameters of the tree. To evaluate the model's accuracy, we used the accuracy, F1-score, Precision, Recall etc.

#### Results
<!-- Tables showing the results of your experiments -->
The following **Performance Metrics Table** shows the results of the **Decision tree** analysis:
| Model                          | Accuracy Score | Precision | Recall | F1-score |
| ------------------------------ | -------------- | --------- | ------ | -------- |
| Wine Type(classification)      | 0.95           | 0.96      | 0.95   | 0.95     |
| Wine Quality(classification)   | 0.52           | 0.63      | 0.52   | 0.45     |

The following *Performance Metrics Table* shows the results of the **Random Forest** analysis:
| Model                               | Accuracy Score | Precision | Recall | F1-score |
|-------------------------------------|-----------------|-----------|--------|----------|
| Wine Quality(classification)        |      0.626      |    0.6159    |   0.626  |   0.607    |

**For Wine Type classification, visualization of pruned Decision tree and the confusion matrix is given below.**

<img width="743" alt="decision tree" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/496aa514-0800-45ca-a9e5-9450d0e76891">

![Confusion](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/ba181ee8-5b9d-4031-b168-1ee326d99806)

#### Discussion
<!-- A brief discussion on the results of your experiment -->

- For wine type classification, we implemented decision tree which was complex(high number of nodes and very high depth). Also score for training and test data was very high indicated possibility of overfitting. We implemented k-fold cross-validation and did some regularization(pruning). These techniques helped us to address potential overfitting issue.
- For wine quality classification, we implemented complex and regularized tree but scores were not very high as can be seen in the table above. 
- We tried Random forest for Wine Qualty and it was showing a little better results than decision tree. 
### Neural Networks

The Fruit-360 dataset, which consists of (100x100) pixel-sized fruit images, is used to train neural networks. The dataset includes images of 28 different types of fruit. Every image is an input to the neural network, and the label associated with it describes the category of fruit that each image represents. In other words, The output labels are the categorical representations of the fruit categories.

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->
We consider three types of Neural Networks as an experiment analysis on the dataset

**Click to view :** [Neural Network Implimentation Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DeepLearning2.ipynb)

1. ANN (Artificial Neural Network) : We applied a single Dense layer with softmax activation and utilized Adam as the optimizer for the hyperparameters in this model. To evaluate the model's accuracy, we used the accuracy score metric, and the categorical crossentropy was employed as the loss function.

2. ANN with Three Hidden Layers: We implemented a neural network with three Dense layers using ReLU activation functions, followed by an output Dense layer with softmax activation. Adam was employed as the optimizer for the hyperparameters. The model's accuracy was assessed using the accuracy score metric, and categorical crossentropy used as the chosen loss function.

3. CNN (Convolutional Neural Network): We utilized Convolutional layers with MaxPooling, followed by Dense layers and an output layer with softmax activation. Adam was used as the optimizer for the hyperparameters. To assess the model's accuracy, agin we used the accuracy score metric, and categorical crossentropy selected as the loss function.

#### Results

<!-- Tables showing the results of your experiments -->
The following **Performance Metrics Table** shows the results of the Neural Network analysis:
     
 | Model                        | Accuracy Score | Loss   | Precision | Recall | F1-score |
| ---------------------------- | --------------- | ------ | --------- | ------ | -------- |
| ANN with single Layer        | 0.84            | 3.2849 | 0.89      | 0.84   | 0.84     |
| ANN with three hidden Layers | 0.9199          | 0.4830 | 0.92      | 0.92   | 0.91     |
| CNN                          | 0.96            | 0.2222 | 0.97      | 0.96   | 0.96     |

**Confusion Matrix of CNN** for showing the accuracy of the model..
<img width="733" alt="Confusion Matrix CNN" src="https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/assets/100375781/47f9e9ab-b338-4e0d-b7dd-544885053080">

#### Discussion
- During the implementation of the model, we initially selected "Sigmoid" as an activation function. However, it was found that "Softmax" is suitable as we had more than 2 categories. 
- The results of our experiments show that a Convolutional Neural Network (CNN) performs better than other types of models when applied to the Fruit-360 dataset. This indicates that the design of the neural network structure has a big impact on how well the model works. It emphasizes that trying out different designs is crucial to finding the best setup for a particular dataset.

### Conclusion
<!-- Final conclusions regarding your initial objectives -->
Overall, this project explored the wine data and fruit image through advanced data analysis techniques. We used methods like clustering and decision trees to understand type of wine, what makes a good-quality wine and achieved accuracy in predicting wine type and wine quality. Clustering faced complexities in capturing wine quality clusters accurately, while Decision tree models showcased high accuracy for predicting type of wine. For fruit image classification, neural networks, especially CNNs, proved to be highly effective. The datasets we used are reliable and widely accepted in the field. However, if we were to use these models in the real world, we would like to use huge volume of data as we felt that wine dataset was too small(potentiall that is why we faced overfitting in decision trees),  we'd need to be careful about some challenges, such as the sensitivity of clustering to minor dataset variations and the need to adapt to changes in wine and image datasets over time.
