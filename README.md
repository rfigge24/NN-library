# Neural Network library
I started this free-time project to get a feeling for the deeplearning fundamentals and maths. 
This project is based on the tutorials of ["The Independent Code"](https://www.youtube.com/@independentcode/videos). 
I added a lot of my own, and updated the whole project to support batch input, mini-batch gradient descent and many activation functions.
I found that this project helpt me a lot with understanding and getting feeling for the linear algebra math needed for deeplearning. 

## includes:
* A Keras sequential like api, that is suitable for batch learning. 
* 2 demo projects that use the api:
  - An XOR function
  - An interactive color preference learning model:
    - This project was adviced by [Jabrils](https://www.youtube.com/@Jabrils). Props to him for making me interested in AI and deeplearning!

### supported activation functions:
- Binary Step
- ELU
- Leaky ReLU
- ReLU
- Sigmoid
- SoftMax
- SoftPlus
- TanH

### supported layers:
- Dense layer
- Activation function layer

### supported loss functions:
- Mean Squared Error
- Categorical Cross Entropy

## Functionality:
```
from neural_network import Neural_Network
from dense import Dense
import activations as act
import losses
```

Step 1. Define your network as a list of layers:
```
network = [
    Dense(2,4),
    act.ELU(1),
    Dense(4,1),
    act.ELU(1)
    ]
```
Step 2. Define the model: 
with the networks sequential list, loss function and optional the solver (standard is sgd, for batch gradient descent set to None)
```
model = Neural_Network(layers=network, loss_function=losses.MSE, solver='SGD')
```
step 3. fit the model:
```
model.fit(X, Y, epochs=10000, learning_rate=0.1, shuffle=True, verbose=true)
```
step 3. use the model for inference:
```
output = model.predict(input_data=input)
```


