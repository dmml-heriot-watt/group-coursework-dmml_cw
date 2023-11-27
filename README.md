[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/58HShPQN)
# Data Mining and Machine Learning Group Coursework

> [!NOTE]
> You should update and customize this README as the project progresses.

> [!IMPORTANT]
> See the COURSEWORK specification on CANVAS for details of the group coursework criteria/deliverables

## Group Members

> [!IMPORTANT]
> Include your names and `@` your GitHub usernames for each.

1. Parthavi Vaghani  [@Parthvi579](https://github.com/Parthvi579)
2. Vikas Chaudhary [@VikasChaudhary123](https://github.com/VikasChaudhary123)
3. Kaushik Lathiya [@Kaushik0704](https://github.com/Kaushik0704)
4. Augustin Lobjoie [@Augustinlobjoie](https://github.com/Augustinlobjoie)


## Initial Project Proposal

> [!NOTE]
This project aims to explore the fields of computer vision and viticulture by conducting a comprehensive analysis of fruit image classification and wine quality prediction. For the wine dataset, we plan to apply clustering techniques based on quality and chemical properties such as alcohol content, sugar levels, and acid concentrations. Additionally, decision tree algorithms will be employed to identify the key features contributing to the best wine quality. In the fruit dataset, the focus will shift towards the application of neural networks and convolutional neural networks (CNNs) for image classification. The primary goal is to perform image classification and assess the performance using metrics such as accuracy, precision, recall, and F-score.

## Project Planning
We'll make use of GitHub's collaborative features, such as issue tracking, version control, and a dedicated wiki, to organise project tasks and documentation in a way that will facilitate effective development and teamwork.
- **Click to view -** [Wiki](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki)
  
### Research objectives

> [!NOTE]
> What are the questions you are trying to answer? What are the goals of your project?
The objectives of this project are:
> Apply clustering techniques on wine data based on quality and chemical properties.
> Identify key features contributing to the best wine quality using decision tree algorithms.
> Develop effective image classification models for fruit recognition using neural networks and CNNs.
> Assess model performance using metrics like accuracy, precision, recall, and F-score.

### Milestones

> [!NOTE]
> Create a bullet list of the key milestones for your project. Discuss these with your group. You may need to update these as the project progresses.
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
**Objective:** The objective of this dataset is to build a machine learning model that can predict the quality of a wine based on its chemical composition. 

The Wine Quality Dataset is a dataset that contains information about the chemical properties of various wines, example below shows the information about the attributes and the data of this dataset.
  
##### Fruit 360 Dataset
**Objective:** This dataset is used for neural network tasks, including classification and regression.

Fruits 360 dataset is a dataset that contains images of different fruits with 100 * 100 pixel dimension . These images cover different types of fruits. Each fruit is shown from various angles and under different lighting conditions, making the dataset representative of real-world scenarios. 

### Dataset examples
<!-- Add a couple of example instances and the dataset format -->

##### Wine Quality Dataset
The data is typically organized in a tabular format, with rows representing individual wine samples and columns for each attribute.
| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | quality | wine_name |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
| 7.4	 | 0.70 | 0.00 | 1.9 | 0.076 | 11.0 | 34.0 | 0.9978 | 3.51 | 0.56 | 9.4	 | 5 | red |
| 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | 25.0 | 67.0 | 0.9968 | 3.20 | 0.68 | 9.8 | 5 | red |


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
- The Fruits 360 Dataset includes a total of 90,483 images.
- The dataset is divided into the following subsets:
  - Training set: 67,692 images, each featuring a single fruit.
  - Test set: 22,688 images, each showcasing a single fruit.
- There are 131 distinct classes representing different fruits and different types of the same fruit are treated as separate classes, ensuring a variety of fruits.
  
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
-   There are no missing values in the dataset.
-   - Click to view [Correlation Matrix](https://heriotwatt-my.sharepoint.com/personal/pv2008_hw_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW%2Fcorrelation%20matrix%2Epng&parent=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW)
-   Some observations from the summary statistics:
    -   Fixed acidity: The mean and median are close, suggesting that the data is symmetric and that there is no strong skewness. The max value is 15.9.
    -   Residual sugar: The mean is 5.44, the median is 3, and the max value is 65.8, indicating the presence of outliers and right skewness. The standard deviation is high, indicating a variety of sugar levels in different wines.
    -   Total sulfur dioxide and free sulfur dioxide: The standard deviation is significant, indicating high variability. The max values are high, so there are potential outliers.
    -   Alcohol: The mean of 10.49 and the median of 10.3 are quite close. The max is 14 and the min is 8.
-   Quality of wine lies between (3-9), but 93% of wines are between 5-7 quality, and 99.5% wines have a quality between 4 to 8.
-   The scatterplot suggests that density and alcohol content in the wine (both red and white) are somewhat negatively correlated. The correlation matrix also shows that alcohol and density are negatively correlated with a value of -0.6867.
-   There are positive correlations between density and residual sugar (0.55), density and fixed acidity (0.46), and alcohol and quality (0.44).
-   Total sulfur dioxide and free sulfur dioxide have a very strong correlation (0.72), so only one of these features should be selected during feature reduction.
-   Wines with higher fixed acidity, alcohol content, and lower volatile acidity tend to be of higher quality.

### Dataset Analysis
<!-- Analysis of your dataset -->

-  There is no missing value in this dataset to handle 
-  Identified outliers in certain features in the dataset.

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
2.  **Expectation-Maximization (EM) Algorithm:**  - The EM algorithm was applied to the dataset, considering all features. We observed joinplots for 'fixed acidity' vs. 'pH,' 'fixed acidity' vs. 'alcohol,' 'pH' vs. 'alcohol,' and 'sulphates' vs. 'alcohol' to assess the efficiency of clustering where we could try to extract different cluster that would help us predict the quality of a wine. 

#### Results
<!-- Tables showing the results of your experiments -->
The following **Score Table** shows the key results of the clustering analysis:
| Clustering Technique                        | Homogeneity Score | Completeness Score | Optimal k (Elbow Method) | Average Silhouette Score |
|--------------------------------------------|-------------------|--------------------|--------------------------|---------------------------|
| 9-cluster K-means Clustering (All Features) | 0.41              | 0.24               | Not Clear                | 0.24537          |
| 4-cluster K-means Clustering (All Features) | 0.1857            | 0.0974             | Not Clear                | 0.2453                    |
| 3-cluster K-means Clustering (Selected Features) | 0.1448       | 0.0398             | 3                        | 0.85                      |

**EM Algorithm:** Mean quality of clusters ranged from 5.2 to 6.5, indicating inefficiency.

- Hyperparameter variations were explored for the number of clusters (k) in the K-means algorithm.
- Variations helped in selecting an appropriate number of clusters based on the silhouette score.

- **Elbow Method** for selecting optimal number of cluster.
[Elbow of 3-cluster K-means Clustering (Selected Features)](https://heriotwatt-my.sharepoint.com/personal/pv2008_hw_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW%2Felbow%2Epng&parent=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW)
- **Silhouette coefficient Method** for selecting appropriate number of cluster
[Silhouette Score of 3-cluster K-means Clustering (Selected Features)](https://heriotwatt-my.sharepoint.com/personal/pv2008_hw_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW%2Fsilhoutte%20score%2Epng&parent=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW)
- **Joinplots** to access efficiency of cluster
[Join plots for EM algorithm using selected feature )](https://heriotwatt-my.sharepoint.com/personal/pv2008_hw_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW%2Fjoinplot%2Epng&parent=%2Fpersonal%2Fpv2008%5Fhw%5Fac%5Fuk%2FDocuments%2FCW)

#### Discussion
<!-- A brief discussion on the results of your experiment -->
The K-Means and EM algorithms were used in an iterative manner for clustering. While feature selection based on joint plots improved interpretability, the EM algorithm did not demonstrate efficiency in clustering data and predicting wine quality. In K-Means, The initial attempt of 9 clusters aligned with the diverse quality levels did not produce meaningful results. After that, we observed that pH and alcohol feature selection significantly improved the accuracy of the clustering model. Hyperparameter variations played a crucial role in selecting the right number of clusters for the chosen features. Variations in number of clusters helped in selecting an appropriate number of clusters based on the silhouette score.In the end, 0.14 Homogeneity score and 0.0398 Completeness score indicates that the clustering algorithm may not be effectively capturing the wine quality in the dataset.


### Decision Trees

The Wine Quality is a dataset that contains information about the chemical properties of various wines. Each row in the dataset represents a specific wine sample, and the features serve as inputs for the decision trees.The output labels represent the quality of the wine. Specifically the labels are assigned based on the quality rating of each wine.

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->

We consider two types of Decision tree as an experiment analysis on the dataset

**Click to view :**
  - [Decision Tree Classification Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreePara.ipynb) 
  - [Random Forest Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DecisionTreeQuality.ipynb)

1.  **CART Algorithm:**
    In our exploration of decision tree models for wine quality prediction, we initiated with a comprehensive analysis using all available features. The initial decision tree showed impressive accuracy,     
    reaching 99.96% on the training set and 99.20% on the test set. However, we worried that the model might be overfitting to the training data, leading to poor performance on new data. We tried to address     
    this by using K-fold cross-validation and adjusting the complexity of the tree. To address concerns about the model's complexity, we tuned the model's parameters. To evaluate the model's accuracy, we used       the accuracy score metric.


#### Results
<!-- Tables showing the results of your experiments -->
The following **Performance Metrics Table** shows the results of the Decision tree analysis:
| Model                        | Accuracy Score | Precision | Recall | F1-score |
|------------------------------|-----------------|-----------|--------|----------|
| Classification        |      0.58      |    0.57    |   0.58  |   0.57    |

#### Discussion
<!-- A brief discussion on the results of your experiment -->

- During the model implementation, our initial approach involved implementing a baseline classification model. We implemented k-fold cross-validation and constructed a complexity tree. These techniques helped us identify and address potential overfitting issues.
- After making some adjustments to the model, we were able to achieve an accuracy of 96%. This shows that the adjustments we made helped the model to be more accurate and reliable.

### Neural Networks

The Fruit-360 dataset, which consists of (100,100) pixel-sized fruit images, is used to train neural networks. The dataset includes images of 28 different types of fruit. Every image is an input to the neural network, and the label associated with it describes the category of fruit that each image represents. In other words, The output labels are the categorical representations of the fruit categories.

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

[Confusion Matrix of CNN](https://heriotwatt-my.sharepoint.com/:i:/r/personal/pv2008_hw_ac_uk/Documents/CW/Confusion%20Matrix%20CNN.png?csf=1&web=1&e=XusctI)

#### Discussion
- During the implementation of the model, we initially selected "Sigmoid" as an activation function. However, it was found that "Softmax" is more accurate for this dataset.
- The results of our experiments show that a Convolutional Neural Network (CNN) performs better than other types of models when applied to the Fruit-360 dataset. This indicates that the design of the neural network structure has a big impact on how well the model works. It emphasizes that trying out different designs is crucial to finding the best setup for a particular dataset.

### Conclusion
<!-- Final conclusions regarding your initial objectives -->
Overall, this project explored the wine data and fruit image through advanced data analysis techniques. We used methods like clustering and decision trees to understand what makes a good-quality wine and achieved impressive accuracy in predicting wine quality. Clustering faced complexities in capturing wine quality clusters accurately, while Decision tree models showcased high accuracy,For fruit image classification, neural networks, especially CNNs, proved to be highly effective. The datasets we used are reliable and widely accepted in the field. However, if we were to use these models in the real world, we'd need to be careful about some challenges, such as the sensitivity of clustering to minor dataset variations and the need to adapt to changes in wine and image datasets over time.
