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
>Wine Dataset
1. Decision tree to find which wine has best quality .
2. Clustering based on quality or chemical properties like alcohol, sugar,  acid content etc .

>Fruit Dataset
1. For Neural Networks and CNN
2. Classification or Regression    
3. Performance Assessment: to measure accuracy, precision, recall, F score etc.
4. Generalization: to make accurate prediction for new data
5. Optimization: to achieve the best possible performance by adjusting parameters

## Project Planning
Wiki for maintain weekly task of the project
- [Wiki](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/documentation/wiki)
### Research objectives

> [!NOTE]
> What are the questions you are trying to answer? What are the goals of your project?

### Milestones

> [!NOTE]
> Create a bullet list of the key milestones for your project. Discuss these with your group. You may need to update these as the project progresses.


## Findings Report

<!-- Below you should report all of your findings in each section. You can fill this out as the project progresses. -->

### Research objectives
<!-- What questions you are trying to answer? -->

### Datasets
1. Wine Quality Dataset - https://doi.org/10.24432/C56S3T
2. Fruit 360 Dataset - https://www.kaggle.com/datasets/moltean/fruits

#### Dataset description

##### Wine Quality Dataset
The Wine Quality Dataset is a dataset that contains information about the chemical properties of various wines, including attributes like acidity, pH, alcohol content, and more. Each wine is associated with a quality rating. The objective of this dataset is to build a machine learning model that can predict the quality of a wine based on its chemical composition. This task involves data preprocessing, and model building using Python and scikit-learn.

Additional Variable Information
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
12 - quality (score between 0 and 10)

Dataset: Wine Quality Dataset
Source: Cortez, Paulo, Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Accessed on: 20/09/2023

##### Fruit 360 Dataset

Fruits 360 dataset is a dataset that contains images of different fruits. These images cover different types of fruits. Each fruit is shown from various angles and under different lighting conditions, making the dataset representative of real-world scenarios. The Fruit Dataset is a valuable resource tailored for neural networks. This dataset serves multiple purposes, including classification and regression tasks. It is used for performance assessment, measuring metrics like accuracy, precision, recall, and F-score.

Dataset: Fruit 360 Dataset
Source: Horea Muresan and Mihai Oltean. Fruits 360. Kaggle. https://www.kaggle.com/datasets/moltean/fruits.
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Accessed on: 20/09/2023

#### Dataset examples
<!-- Add a couple of example instances and the dataset format -->
| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | quality | wine_name |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
| 7.4	 | 0.70 | 0.00 | 1.9 | 0.076 | 11.0 | 34.0 | 0.9978 | 3.51 | 0.56 | 9.4	 | 5 | red |
| 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | 25.0 | 67.0 | 0.9968 | 3.20 | 0.68 | 9.8 | 5 | red |


#### Dataset exploration
<!-- What is the size of the dataset? -->
**Size and Structure**
- The dataset consists of 6497 entries and 13 features including a Numeric type and a wine_name feature of categorical type.
- The dataset is divided into 1599 entries for Red wine and 4898 entries for white wine.
  
<!-- Train,validation,splits? -->
**Train, Validation and Split**
<!-- Summary statistics of your dataset -->
<!-- Visualisations of your dataset -->
<!-- Analysis of your dataset -->

##### Data Visualization 
**Insights from wine dataset:**
-   There are no missing values in the dataset.
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


### Clustering
Implemented K-Means Clustering, used Elbow method and silhouette score to verify number of clusters. 
#### Experimental design
<!-- Describe your experimental design and choices for the week. -->

#### Results
<!-- Tables showing the results of your experiments -->

#### Discussion
<!-- A brief discussion on the results of your experiment -->

### Decision Trees

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->

#### Results

<!-- Tables showing the results of your experiments -->

#### Discussion
<!-- A brief discussion on the results of your experiment -->

### Neural Networks

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->

#### Results

<!-- Tables showing the results of your experiments -->

#### Discussion
<!-- A brief discussion on the results of your experiment -->


### Conclusion
<!-- Final conclusions regarding your initial objectives -->
