## Week- 9 Task
- [x] Meet Team Member 
- [x] Update GitHub Repository 
- [x] Implement Neural Network

### Neural Networks

For implementation of neural network we have selected Fruit-360 dataset, which consists of (100,100) pixel-sized fruit images, is used to train neural networks. The dataset includes images of 28 different types of fruit. Every image is an input to the neural network, and the label associated with it describes the category of fruit that each image represents. In other words, The output labels are the categorical representations of the fruit categories.

### Experiment

We consider Artificial Neural Networks as an experiment analysis on the dataset

**Click to view :**  [Neural Network Implimentation Notebook](https://github.com/dmml-heriot-watt/group-coursework-dmml_cw/blob/main/notebooks/DeepLearning2.ipynb)

1.  ANN (Artificial Neural Network) : We applied a single Dense layer with softmax activation and utilized Adam as the optimizer for the hyperparameters in this model. In the initial development  to calculate the accuracy of model, we used the accuracy score metric, and the categorical crossentropy was employed as the loss function.
    
2.  ANN with Three Hidden Layers: We implemented a neural network with three Dense layers using ReLU activation functions, followed by an output Dense layer with softmax activation. Adam was employed as the optimizer for the hyperparameters. The model's accuracy was assessed using the accuracy score metric, and categorical crossentropy used as the chosen loss function.
        
-   During the implementation of the model, we initially selected "Sigmoid" as an activation function. However, it was found that "Softmax" is more accurate for this dataset.
