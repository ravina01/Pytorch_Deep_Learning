# Pytorch_Deep_Learning
Basic To Intermediate : Covers Pytorch Fundamentals, Workflow, Neural Network Classification, Computer Vision, Custom Datasets

Heads up : I use ML and DL pretty interchangably.

## Fundamentals : CODE + MATH
Well internet is the best place to find the definations but this is what I think -

### What's deep learning ?
Turning things (Data - visual, audio, almost anything) into numbers and finding patterns in those numbers.

I think you can ML/DL for literally anything as long as you can convert it into numbers and program it to find patterns. Literally it could be anything any form of input or output from the god damn universe.

### Why use deep learning ?
Good reason : Because Why Not ? 

Better reasons : Complex system to figure out the pattern.

1. Problems with long lists of rules - when the traditional approach fails, ML/DL may help.
2. DL can adapt/ learn from new scenarios - continually changing environments
3. Discovering insights within large collection of data - can you imagine trying hand-craft rules for what 101 different kinds of food look like ? You will use DL approch.

### What deep learning is not Good for ? Interseting !
1. when you need explainability - the patterns learned by a DL model are typically uninterpretable by a human.(weights and biases)
2. if you can accomaplish what you need with simple rule-based system, then you dont need deep learning. (Google's No.1 rule of ML handbook)
3. when errors are unacceptable, outputs of DL modle are not always predictable.
4. when you dont have much data - DL models usually require fairly large amount of data to produce great results. (Data augmentation is to the rescue though)

### ML  vs DL
1. You use ML on structured data, with help of XGBoost - mainly used in production settings
2. DL is used for unstructured data (image, text, audio, voice assistant etc) - we can turn this data to have a structure through the beauty of a tensor.

### ML vs DL (Algorithms edition)
#### ML Algos --> 
Random Forest, Gradient boosted models, Naive Bayes, Nearest neighbors, Support vector machine, and man more
#### since the advent of deep learning these are reffered to as "Shallow algorithms"
#### DL Algos --> 
Many different layers of algos --> Neural networks, Fully connected neural network, Convolutional neural network, Recurrent neural network, Transformer.. many more


 ### Heads up Time : 
This will only cover --> Neural networks, Fully connected neural network, Convolutional neural network

Excellenet thing is that - If we learn these foundational building blocks, then we can get into these other styles of things here. (with Pytorch)

### What are neural networks ?
search - 3 Blue 1 Brown

Numerical encoding of input data - Images/ texts/ audio - then we pass it thorugh neural network to learn patterns/ features/ weights - Learnt representations are considered as outputs - convert these outputs into human underastadable terms. 

Anatomy of Neural Networks -
Input layer (data goes in here) - Hidden Layer(s) (learns pattern in data) - Output Layer (prediction probabilities)

#### Note : 
pattern is an arbitary term, you will often hear embedding, weights, feature representation, feature vectors - all refering to similar things.

Each layer in neural netwrok is using linear/ non linear functions to find/draw patterns in our data.

#### Paradigms of how neural network learns -
1. Supervised Learning - Loads of labelled data : data + label
2. Unsupervised & self-supervised Learning - just the data w/o any label : data + NO Label - Forms clusters / different the datas but doesn't know the type(label) of data.
3. Transfer Learning - more like head start - its very powerful technique - It takes the pattern that one model has learnt of a certain dataset and transferring it to another model.
(sorry not talking about Reinforcement learning here)

#### Note: We will focus on - Supervised and Transfer Learning for now. 

#### Applications - 
1. Recommendations systems
2. Sequence to sequence (seq2seq): Translations, speech recognition
3. Classification/ Regression: Computer vision, Natural language processing

### Let's Talk about Pytorch - it's research favourite
Don't forget internet is your friend. Your ground truth is - https://pytorch.org/
Make sure you install and read basic documentation.

#### Most popular research deep learning framework - by Meta
1. Write fast DL code in python (able to run on GPU/ many GPUs)
2. Able to access many pre-built DL models
3. Whole stack: pre-process data, model data, deploy model in your application/ cloud
   
#### Tracks latest and greatest DL papers with Code - https://paperswithcode.com/trends

Pytorch levereges CUDA to use your ML/DL code on NVIDIA GPUs(Graphics Processing Unit)

### What is a Tensor ?
Any representation of numbers - scalars (0-dimensional), vectors (1-dimensional), and matrices (2-dimensional) to potentially higher dimensions.
Do watch : What's a Tensor? by Dan Fleisch (https://www.youtube.com/watch?v=f5liqUk0ZTw)

#### Best Way to learn ML,DL
![image](https://github.com/ravina01/Pytorch_Deep_Learning/assets/91215561/6e2e4a65-7ad9-4afb-8752-54dc97df0978)

#### What are we going to cover Broadly ?
Pytorch basics & fundamentals (dealing with tensors and tesnor operations)

#### What's Next? 
1. Pre-processing data (getting it into tensors)
2. Building and using pre-trained Deep Learning models
3. Fitting model to the data (learning patterns)
4. Making predictions with a model (using patterns)
5. Evaluating model predictions
6. Saving and loading model
7. Using a trained model to make predictions on custom data

#### We will be cooking up lots of code! ML, DL is little bit of science and little bit of arts.

### Pytorch Workflow
![image](https://github.com/ravina01/Pytorch_Deep_Learning/assets/91215561/9a50f09e-32fd-4e32-8ea3-b389d315c8f6)

### TIPS:
1. Motto#1 - Write and Run the code
2. Motto#2 - Explore and experiment - Best way to learn anything - Experiment, experiment, experiment!
3. Motto#3 - Visualize what you don't understand - Visualize, visualize, visualize!
4. Motto#4 - If in doubt, run the code and find out!

### Let's get started -
# PyTorch Fundamentals

---
### 00. PyTorch Fundamentals

### Q. whats tensor ? - torch.tenso() : print(torch.__version__) : 2.0.0+cu117
- main building block of data in deep learning
- multi-dimensional numeric data
- scalars - ndim = 0,shape - torch.Size([]), vectors - ndim = 1,shape = torch.Size([Number of elements inside vector])
-(scalar - use item() to get numeric value instead of tensors)
- matrix: ndim =2, torch.Size([2, 2])
- tensor: ndim =3 or more.. torch.shape([1,3,3]) - Total number of 2d matrix [row x col] - 1, 3, 3 or 2, 2, 2etc
- Number of Square bracets give you - ndim count.
- scalar , vector - lower case (usually)
- matrix and tensors - Upper case (usually)

### Random Tensor -
- They are important as they start with tensors full of random numbers and then adjust those random numbers to better represent data.
- generates a tensor filled with random numbers drawn from a uniform distribution on the interval [0,1]
- start with randome numbers-> look at data -> update random numbers
- torch.rand(shape/size- 1,2,3 or 2,2) - 2d/3d dim
- randome_image_size_tensor = torch.rand(size=(244, 244, 3)) # height, width, no of channels
- we can create all zeroes / all ones tensors of any shape
 - dtype - default datatype - its torch.float32: single precision (1 sign bit, 8 exponent bits, 23 significand bits).
 - torch.range(0,10) - deprecated - use torch.arange(start, end, step) instead
 - when creating tenors you can pass - value, dtype, device, requires_grad (weather or not track gradients with this tensor)
 - data types talk about precision in computing, 
   
### Typical errors we might face when dealing with tensors -
1. Tensors not right datatype
2. Tenosrs not right shape
3. Tensors not on right device, by default its cpu

### Tensor Opeerations 
#### 1. Addition , Sub, Divison, multiplication(element wise, matrxi multiplication aka dot product)

     One of the most common errors in DL - tensor shape : Investigation
     1. (3,2) @ (2, 3) - @ = matmul
     2. Match the inner dimensions
     3. You can take transpose

#### 2. Tensor Aggregation - min, max, mean and sum of tensors
- We might get error as not correct data type when calculating mean of tensors.
- Solution - change the data type of tensor as mean can't be done on long datatypes(tensor_A.dtype = int64)
- Finding positional min and max tensors
- argmin and argmax - Find the position in tensor that has the min/max value with argmin()/ argmax()
- Returns index at of min/max element
- argmax(), argmin() is used in Softmax

#### 3. Reshaping, Viewing and Stacking, Squeeze and Unsqueeze, Permuting tensors
1. Reshape - reshapes to a defined shape - we can reshape in multiple of shape of tensor
- Tensor of size 1, 10 - can we shaped to 5,2 or 10, 1 colum wise

2. View - return a  view of input tensor of a certain shape but, keep the same memory as orginal tensor
- if z is view of x and we modify the z then it modifies the x also.

3. Stacking - combine multiple tensors, on top of each other - vstack or side by side (hstack)
- x_stck = torch.stack([x, x, x])
- Concatenates a sequence of tensors along new dim
- dim takes 0, 1,-1(colum wise) values
  
4. Squeeze - remove one dim from tensors
- Removes all single dimensions from target tensors

5. Unsqueeze - adds one dim to the target tensor at specific dim=0,1,2, depends
- making single list/vector of 10 elements into 1,10 or 10,1
  
6. Permute - Return a view(shares same memo) of the input with swapped order of dimenstions (height , width, channels) can be (channels, H, W)
- rearanges the tensors.
- torch.permute(x, (2, 0, 1)).size()
  
#### 4. Selecting data from Tensors - Indexing
- slicing - to get column, indexing
  
#### 5. Tensors and Numpy
- converting pytorch data in numpy arrays and vice-versa.
- torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.- float64 by default
- torch.Tensor.numpy() - PyTorch tensor -> NumPy array. - float 32 by default
- Default data type of numpy is float64 and Pytorch tensor is flaot 32 - Dont forget.
  
#### 5. Pytorch Reproducibility - Trying to take random out of random
- In short how neural net learns ->
- start with random numbers -> tensor operations -> update the random tensors to try and make them better represntations
- of the data -> again, again, again...
- to reproduce the randomness in the neural network - comes the concept of **random seed**
- torch.rand() doesnt have seed method so we have to explicitly set seed to 42 or any number before producing random tensor
- manual seed flavors the randomness - makes it reproducable
  
```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
t3= torch.rand(3,3)
torch.manual_seed(RANDOM_SEED)
t4 = torch.rand(3,3)
print(t3==t4)
```
#### 5. Running GPU on Pytorch - faster Computations

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
#### 6. Setting up Device-agnostic code and putting tensors on and off the GPU
- Putting tensors/ models on GPU 0 faster computations
- Numpy works  on CPU so move tensors back to CPU before converting into numpy array
- if tensor is on gpu we cant transform it to numpy
  
```python
t1_on_gpu = t1.to("cuda")
t1_on_gpu # tensor([1, 2, 3], device='cuda:0')

tensor_back_on_cpu = t1_on_gpu.cpu()
tensor_back_on_cpu.device # device(type='cpu')
tensor_back_on_cpu = t1_on_gpu.cpu().numpy() # its on cpu
```

# PyTorch Workflow Fundamentals

---

### 01. PyTorch Workflow Fundamentals

#### 1. Data (preparing and loading)
- Split data into training and test sets

#### 2. Build model
- PyTorch model building essentials
- Checking the contents of a PyTorch model
- Making predictions using `torch.inference_mode()`

#### 3. Train model
- Creating a loss function and optimizer in PyTorch
- Creating an optimization loop in PyTorch
- PyTorch training loop
- PyTorch testing loop

#### 4. Making predictions with a trained PyTorch model (inference)

#### 5. Saving and loading a PyTorch model
- Saving a PyTorch model's `state_dict()`
- Loading a saved PyTorch model's `state_dict()`

#### 6. Putting it all together
- 6.1 Data
- 6.2 Building a PyTorch linear model
- 6.3 Training
- 6.4 Making predictions
- 6.5 Saving and loading a model

#### 1. Data (preparing and loading)
1. Setup - Will use Torch.nn (Read more - https://pytorch.org/docs/stable/nn.html)
2. Creating a Simple Dataset Using the Linear Regression Formula
- Data can be almost anything -> Excel spreadsheet, Images, Videos, Audio, DNA and pattterns, and Text etc.
3. Machine learning is a game of two parts:
Turn your data, whatever it is, into numbers (a representation).
Pick or build a model to learn the representation as best as possible.
4. Let's Create a Simple Dataset Using the Linear Regression Formula
5. Splitting data into traing and test, validation(not always) sets
- Scikit learn's split adds randomness to the data
  
![image](https://github.com/user-attachments/assets/f10e7cb6-6de6-402b-9c6c-3c4fd2a98ccd)

#### 2. Build model
##### 2.1 Flow
- Start with random values (weights and bias)
- look at training data and adjust the random values to better represent (or get closer to) the ideal values
- How does it do so ?
- Through 2 main algorithms - Pytorch is taking care of below algorirthms so that we don't have to work on them.
- 1. Gradient Descent
  2. Backpropogation

![image](https://github.com/user-attachments/assets/eb39ff65-7e8e-44c7-b4f6-76dd4a9298f7)


```python
class LinearReg(nn.Module): # inhherits

    def __init__(self) -> None:
        super().__init__() # calls constructor of parent class

        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # torch.randn(1): Generates a tensor with a single random value from a normal distribution (mean 0, variance 1).
        # nn.Parameter(): Wraps the tensor as a parameter, indicating that it should be optimized during training.
        # requires_grad=True: Specifies that gradients should be computed for this parameter during backpropagation.


    def forward(self, x:torch.Tensor) -> torch.Tensor: # x is input data
        #  forward pass is the computation performed to generate the output of the model from the input data.
        # Any subclass of nn.module needs to override the forward method.
        # Defines forward computation of the model
        return self.weights * x + self.bias
```

##### 2.2 Pytorch Model Building Essentials -
* torch.nn - contains all the buildings for computational graphs (neural net is computational graph itself)
* torch.nn.Parameter - Wraps the tensor as a parameter, indicating that it should be optimized during training.
* torch.nn.Module - The base class for all neural netwoek modules. If you subclass it, you should override the forward() method.
* torch.optim - Pytorch optimizers this will help with Gradient Descent.
* def forward() - All nn.Module subclass require you to override the forward(), this method defines what happens in forward computation.
  
![image](https://github.com/user-attachments/assets/761b21d8-2528-41cf-bf57-63fb0ea39a38)

**Refer - Pytorch cheat sheet - https://pytorch.org/tutorials/beginner/ptcheat.html**

![image](https://github.com/user-attachments/assets/4c50351c-9bfb-4ff1-a9fc-873bcd6f540c)

```python
MANUAL_SEED = 42

torch.manual_seed(MANUAL_SEED)
model_linearReg = LinearReg()
list(model_linearReg.parameters())

'''
[Parameter containing:
 tensor([0.3367], requires_grad=True),
 Parameter containing:
 tensor([0.1288], requires_grad=True)]
'''

model_linearReg.state_dict()
OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])
```

##### 2.3 Let's make Predictions ->
- start with random numbers and with help of neural nets progress towards the ideal numbers (weights, bias) which better fit the data.
- We make predictions using torch.inference_mode() method.
- To check our models' predictive powe, let see how well it predicts 'y_test' based on 'X_test'
- Input data is passed through forward() method.
  
```python
with torch.inference_mode():
    y_preds = model_linearReg(X_test)
'''
y_preds = tensor([[0.3982],
        [0.4049],
        [0.4116],
        [0.4184],
        [0.4251],
        [0.4318],
        [0.4386],
        [0.4453],
        [0.4520],
        [0.4588]])

y_test =  tensor([[0.8600],
         [0.8740],
         [0.8880],
         [0.9020],
         [0.9160],
         [0.9300],
         [0.9440],
         [0.9580],
         [0.9720],
         [0.9860]]))
'''
```
y_preds and y_tests are not even close, let's visualize the data

![image](https://github.com/user-attachments/assets/66021384-7de8-41d4-badd-8c361e07405c)

##### 2.4 To Improvise the Model and better fit the linear regression model, we need to change few things ->
- Inference model turns off the gradient tracking.
- Always write predictions with the with torch.inference_mode():
- If you pass your input data through the instance of the class, then the predictions would be containg the gradient tracking.
- Benefit - it saves memory, no need to keep track of the gradients in inference mode.
- Pytorch official Guidelines, replace torch.no_grad() -> torch.inference_mode()
- torch.inference_mode() is Preferred.

##### 2.5 Let's Train the data
- We saw the predictions are not even close to data plots, start with random values and train the model to estimate the idel values of training parameters which better represent the data.
- The whole idea of training is to move from **unknown(random)** parameters to some **known** parameters.
- To measure how poor the predictions are will use Loss functions.
- **Note**: Loss functions may be also called cost functions / criterion in different areas.
  
- **Things we need to Train ->**
  
- 1. **Loss Functions** - talk about how wrong your model's predictions are to the ideal outputs. Lower the better.
- 2. **Optimizer** - takes into account the loss of the model and adjusts the model's parameters. (weights and bias) to improve the loss function.
  3. We need 2 loops - 1. Training Loop and 2. Testing Loop
  4. Will setup MAE(Mean absolute error/ L1) loss function and SGD(Stoacstic(random) Gradient Descent) Optimizer.
  5. Stochastic Gradient Descent (SGD) in deep learning updates model parameters by computing the gradient of the loss function using a single randomly selected data point or a mini-batch, adjusting the parameters in the direction that minimizes the loss. This process is repeated iteratively to gradually converge to the optimal parameters.
  6. Loss functions tells the error between expected and estimated output while optimizers try to get ideal values of model parameters.
  7. **Learning rate** - The learning rate is a small value that controls how much the model's parameters are adjusted during each step of the training process. It determines the size of the steps the model takes towards minimizing the error, balancing between too slow progress and overshooting the optimal solution.
     
```python
print(model_linearReg.state_dict())
# Parameters - OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])

# Loss Function
loss_fn = nn.L1Loss()

#Learning rate - Hyperparameter. Steps taken to optimize the parameters
lr = 1e-2

# Optimizer - pass models Parameters + LR
optimizer = torch.optim.SGD(model_linearReg.parameters(), lr)
```
**Q. Which loss function and optimizer should I use ?**
- This is problem specific. But with experince, you will get an idea, of what works and what doesn't work with your problem set.
- 1. Regression Problem -
     - Loss Function - L1 Loss
     - Optimizer - SGD
- 2. Classification Problem -
     - Loss - Binary Cross Entropy Loss (Binary Classification Problem)
     - Optimizer - SGD, Adam, RMS Prop
     
##### 2.6 Training Loop Steps and Intution
Steps ->
- Loop through data
- Forward pass - calculates forward propagation - to make predictions on data.
- Calculate the loss (compare the forward pass predictions to GT labels)
- Optimizer zero grad - By default, how the optimizer changes will accumulate through the loop so, we have to 0 it in next iteration.
- Loss backward - move backwards through network to calculate the gradients of each of the parameters of our model with respect to the loss. (**backpropogation**)
- Optimizer Step - use optimizer to adjust/updtae the model's parameters - to try and minimze(improve) the loss. (**gradient decent**) - minimize the gradients -> 0
- This is how our model goes from random parameters to beter parameters using math.
- Math is taken care by Pytorch.
  
```python
torch.manual_seed(42)
# Epoch is one loop through data - Hyperparameter
epochs = 1

# Loop through the data.

for epoch in range(epochs):

    # set moodel to traing mode
    # Train mode - sets all paras that requites gradients to True
    model_linearReg.train()

    #1. Forward pass
    y_pred = model_linearReg.forward(X_train)

    # 2. Calculate Loss Function
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Loss Backward - Perform backpropoogation on the loss wrt to the parameters 
    loss.backward()

    # 5.Optimizer Step
    optimizer.step()
    # By default, how the optimizer changes will accumulate through the loop so, 
    # we have to zero them above in step 3
```

![image](https://github.com/user-attachments/assets/9f44fb5d-d74d-4af5-b8b8-485d20eb4c0e)

**Note** - Learning rate Scheduling - We can start with largers steps and then we can take the smaller steps to reach the minimum loss value.

![image](https://github.com/user-attachments/assets/0182a112-4cf2-44eb-a6ac-73346b4d9140)

Results after Training - 
```python
model_linearReg.state_dict()
# OrderedDict([('weights', tensor([0.6990])), ('bias', tensor([0.3093]))])

with torch.inference_mode():
    y_preds_new_test = model_linearReg(X_test)
plot_predictions(predictions=y_preds_new_test)

```

![image](https://github.com/user-attachments/assets/0ab0a6d7-eb82-45de-856d-0e6dc70e08db)

##### 2.7 Testing Loop Steps and Intution
- Tetsing mode is set by model.eval()
- It turns off different settings in the model no needed for evalution/testing (dropout/batchNorm layers)
- Steps ->
- 1. set to eval mode
  2. with torch.inference_mode() - turns off gradient calculation
  3. forward() pass on Test data - calcualte the test predictions
  4. Calculate test loss (test_data and its labels)

 ```python
torch.manual_seed(42)
# Epoch is one loop through data - Hyperparameter
epochs = 100


# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

# Loop through the data.
for epoch in range(epochs):

    # TRAINING CODE

    # set moodel to traing mode
    # Train mode - sets all paras that requites gradients to True
    model_linearReg.train()

    #1. Forward pass
    y_pred = model_linearReg.forward(X_train)

    # 2. Calculate Loss Function
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Loss Backward - Perform backpropoogation on the loss wrt to the parameters 
    loss.backward()

    # 5.Optimizer Step
    optimizer.step()
    # By default, how the optimizer changes will accumulate through the loop so, 
    # we have to zero them above in step 3


    # TESTING CODE
    model_linearReg.eval() # turns off gradient Tracking

    with torch.inference_mode():
        y_test_pred = model_linearReg.forward(X_test)

        test_loss = loss_fn(y_test_pred, y_test)

        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        print(model_linearReg.state_dict())
```
![image](https://github.com/user-attachments/assets/d678625f-276a-4f7d-a9f6-45e37d0ee3aa)

##### 3. Save and Load Model - Inference Pipeline
Three main methods to save and load (Serialize and Deserialize)
- 1. torch.save() - allows you to save Pytorch object in Python's pickle format
  2. torch.load() - allows you to load a saved Pytorch Object 
  3. torch.nn.Module.load_state_dict() - allows to load a model's saved state dictionary - all paras and their final/updated values which resulated into minimum Loss.
  4. We can save and load entire model or just the state_dict()
     
```python
from pathlib import Path

# 1. Create directory
MODEL_PATH = Path("models")

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_Linear_Reg.pth" # or .partition

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)
# models/01_pytorch_workflow_Linear_Reg.pth

# 3. Save the model state_dict()
torch.save(obj=model_linearReg.state_dict(), f=MODEL_SAVE_PATH)
```


#### 02. PyTorch Neural Network Classification
#### 03. PyTorch Computer Vision
#### 04. PyTorch Custom Datasets
