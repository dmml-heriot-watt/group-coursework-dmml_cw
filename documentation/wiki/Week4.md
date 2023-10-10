## Week- 4 Task
- [x] Meet Team Member (4/10/23)
- [x] Update GitHub Repository 
- [x] Explore Dataset
- [x] [Data Visualization](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/WineDataViz.ipynb)


## Data Visualization 
**Insights from wine dataset:**

-   The dataset contains 6497 entries and 13 features, including a Numeric type and a wine_name feature of categorical type.
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


