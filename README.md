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
##### Load The model ->
- 1. Since we saved the state_dict() rather than the entire model, will now create new instance of the model class and then load the saved_dict() into that.

```python
# 1. Create nee instance of model class
loaded_model = LinearReg()

# 2. Load the saved state_dict()
loaded_state_dict = torch.load(f=MODEL_SAVE_PATH)

loaded_model.load_state_dict(loaded_state_dict)

loaded_model.state_dict()
# OrderedDict([('weights', tensor([0.6990])), ('bias', tensor([0.3093]))])
```

You can Predict on these loaded values  
```python
# 1. Put the loaded model into evaluation mode
loaded_model.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test) # perform a forward pass on the test data with the loaded model

# Compare previous model predictions with loaded model predictions (these should be the same)
y_preds == loaded_model_preds
# return vector of True (10,1)
```

# PyTorch Neural Network Classification

---

### 02. PyTorch Neural Network Classification

- A classification problem involves predicting whether something is one thing or another.
  
##### Types of classification -
1. Binary Classification
2. Multiclass Classification
3. Multilabel Classification
   
![image](https://github.com/user-attachments/assets/e96bb96c-98d4-47ad-a3a4-9ba1a5561b66)

**What are we going to cover -->**
* Architecture of a neural Network Classification Model
* Input shapes and Output shapes of a classification model (features and labels)
* Creative custom data to view, fit on and predict on
* Steps in modelling -
   * Creating a model, setting a loss function and optimizer, creating a training loop, evaluating a model.
* Saving and loading models
* Harnessing the power of non-linearity
* Different classification evaluation methods
  
##### 1. Classification Input and Output

When we use batch size 32 that means our machine look at 32 images at a time. Sadly, our computers don't have infinite computing power.
Input and Output shapes vary dependind on the problem you are working on. The principle of encoding your data as numerical representation remians the same for the inputs, outputs would be some sort of prediction probabilities. 

![image](https://github.com/user-attachments/assets/8bf64c4d-642b-41b9-8666-d02a31afe617)

![image](https://github.com/user-attachments/assets/32fe5a2e-085e-41db-9773-2bfcd23f6a0c)

##### 2. High level Architecture of Classification Model (Overview)

- Inputs to Classification Model - some form of numerical representation
- Ouputs to Classification Model - some form of prediction probability
- 
![image](https://github.com/user-attachments/assets/0f0b4ec6-60c2-4072-967c-777648af7ff2)

![image](https://github.com/user-attachments/assets/0626331c-1de7-4275-bd21-8af55c7ce0a8)


Enough Theory, Now Let's Code ->
Mini-Project - Toy Classification Dataset

![image](https://github.com/user-attachments/assets/f843d316-338c-41b4-86cc-121afa422fa1)


### 03. PyTorch Computer Vision
---
1. Gettning a vision datatset to work with using torchvision.datasets
2. Architecture of CNN with PyTorch
3. End-to-end Multi-class Image Classification Problem
4. Steps in modelling with CNN in Pytorch
   - 1. Creating a CNN model with PyTorch
   - 2. Picking loss and optimizer
     3. Training a model
     4. Evaluate a model


#### CNN Architecture

![image](https://github.com/user-attachments/assets/ebb8a953-7624-46d5-81f5-b71f48c8f744)

#### Base CV Libraries - 
 - The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
 - torchvision.models - get pre-trained cv models
 - torchvision.datasets - get datasets and data loading functions for CV
 - torchvision.transforms - manipulating images to be suitable for use with an ML model, Turns image data into numerical data
 - torch.utils.data.Dataset - base dataset class for PyTorch.
 - torch.utils.data.Dataloader - creates python itererable over a dataset.
 - - torch.nn has cnn layers, loss function, optimizers, etc
   
```python
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor # numpy array to tensors

# Visualization
import matplotlib.pyplot as plt
```
#### Model 0 - 1. Getting dataset - fashion MNIST - from torchvision.datasets
```python
## Setup training data

train_data = datasets.FashionMNIST(
    root = "data", # where to download
    train = True, # do we want to download training datasets
    download= True,
    transform=ToTensor(), # transform the data , tensors
    target_transform=None # transforming the labels
)

test_data = datasets.FashionMNIST(
    root="data",
    train = False, # do we want to download training datasets
    download= True,
    transform=ToTensor(), # transform the data , tensors
    target_transform=None # transforming the labels
)

image, label = train_data[0]
print(label, image.shape)
# image, label = train_data[0]
print(label, image.shape)
```

- Now will find patterns in Train data and Test on test_data
- what totensor does to image data ?
 - converts PIL image / numpy.ndarray (Hx W x C) in the ramge [0, 255] to a torch.FloatTensor of shape(Cx H X W) in range[0.0 , 1.0]
 - Normalized image data.
 - size (28, 28, 1) to  torch.Size([1, 28, 28]

```python
class_to_idx = train_data.class_to_idx
class_to_idx
# This gives us dicttionary of data label to index value
```

![image](https://github.com/user-attachments/assets/851da304-feb0-4a7e-994b-d7295a98c6ce)

#### 2. Getting dataloader - turns our dataset into python iterable
- Turn data into batches/ mini-batches
- It's computationally efficient to group images and then use it for training purpose.
- Your computing hardware wont be able to look at 60000 images in one hit. batch so we break it down to 32 size.
- It gives our NN more chances to updates its gradient per epoch.
- 
![image](https://github.com/user-attachments/assets/d563b7ac-990e-41a7-be09-3f90cf8ca7dc)

Turning datasets into dataloader -
```python

BATCH_SIZE = 32

# turn datasets into iterables. (batches)

train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False) # dont shuffle.
len(train_dataloader), len(test_dataloader)
# total batches = (1875, 313) divided by 32
```
we can chnage batch_size 32/64/128 etc

Batch shape = 32, 1, 28, 28

#### 3. Build baseLine model
- start simply and add complexity when necessary.
  
![image](https://github.com/user-attachments/assets/d4f959d5-b111-4776-a149-17588bf38bab)

- Loss function - Crossnetropy for multi class classification
- Optimizer - SGD
- Evaluation metric
- ML is experimental you will often want to track are:
- 1. Models performnace (loss + accuracy)
  2. How fast it runs.

Creating training Loop + TrainingModelon batches of data
1. Loop through epochs
2. Loop through training batches, perform training steps,
3. calculate train loss / batch
4. loop thriugh test batches, perform testing steps
5. calculate test loss/ batch

```python

# Import tqdm for progress bar

from tqdm.auto import tqdm

torch.manual_seed(42)

train_time_start_on_cpu = timer()

epochs = 3

# Create traina nd test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---")


    # Training

    train_loss = 0

    for batch, (image, label) in enumerate(train_dataloader):
        
        model_0.train()

        # 1. forward pass

        y_pred = model_0.forward(image)

        # calculate loss

        loss = loss_fn(y_pred, label)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(image)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 

    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) 


            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

    # Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))
```

#### 4. Make predictions + get results
```python
torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
model_0_results

```

- Without non-linearity we got - 84 as acciracy after 3 epochs on test data.
- We need to introduce non-linearity now.

#### Model 1-  Building Better model with non-linearity
After adding no-linearity by RELU. lets see how the model performance.

It didn't perform well with added non-linearity + GPU.
FashionMNISTV0 - CPU
FashionMNISTV1 - GPU

- {'model_name': 'FashionMNISTV0',
 'model_loss': 0.4778009355068207,
 'model_acc': 83.33666134185303}
- {'model_name': 'FashionMNISTV1',
 'model_loss': 0.6850008964538574,
 'model_acc': 75.01996805111821}

**NOTE**: Sometimes, The training time on CUDA vs CPU will depend largely on the quality of the CPU/GPU you're using. Read on for a more explained answer.
- Question: "I used a a GPU but my model didn't train faster, why might that be?"
- Answer: Well, one reason could be because your dataset and model are both so small (like the dataset and model we're working with) the benefits of using a GPU are outweighed by the time it actually takes to transfer the data there.
- There's a small bottleneck between copying data from the CPU memory (default) to the GPU memory.
- So for smaller models and datasets, the CPU might actually be the optimal place to compute on.
- But for larger datasets and models, the speed of computing the GPU can offer usually far outweighs the cost of getting the data there.
- Theres overhead in copying data from cpu to gpu.
However, this is largely dependant on the hardware you're using. With practice, you will get used to where the best place to train your models is.

Read more from https://horace.io/brrr_intro.html

### Model 2: Building a Convolutional Neural Network (CNN)

![image](https://github.com/user-attachments/assets/26107503-5c27-4fd3-94f1-9a77c3af7ae2)

![image](https://github.com/user-attachments/assets/5b5b2251-b343-459d-9154-2888365e917d)

Resource - https://poloclub.github.io/cnn-explainer/

1. It's time to create a Convolutional Neural Network (CNN or ConvNet).
2. CNN's are known for their capabilities to find patterns in visual data.
3. The CNN model we're going to be using is known as TinyVGG from the CNN Explainer website.
4. It follows the typical structure of a convolutional neural network:

Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer

![image](https://github.com/user-attachments/assets/d2eed454-3432-4e8b-9036-8b26aced4faf)

![image](https://github.com/user-attachments/assets/1d2b861b-d492-40d4-83e4-1cea8dd80b87)

CNN Resukt 
- {'model_name': 'FashionMNISTModelV21',
 'model_loss': 2.3023061752319336,
 'model_acc': 9.994009584664537}

![image](https://github.com/user-attachments/assets/728acd57-a793-44d9-85c7-7ec75bdd1d22)


V0 - Only flatten Linear layers - CPU
V1 - Relu + GPu
V2 - CNN + GPU

![image](https://github.com/user-attachments/assets/aa8a2677-0617-4f64-804d-d407ae5217cd)

Make Predictions with CNN -



### 04. PyTorch Custom Datasets
---


#### what are we going to do ?

- getting custom dataset with pytorch
- Becoming one with the data (Prepare and visualize)
- Transforming data for use with a model
- Loading custom data with pre-built functions and custom functions
- Building food vision mini to classify images
- Comparing models with and without data augmentation
- Making predictions on custom data

Dataset - 101 food (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

- Instead of 101 food classes though, we're going to start with 3: pizza, steak and sushi.
- Its subset of 101 dataset.
- 101 different classes of food. 1000 / class 750 train and 250 test total 100000 images
- 3 classes of food
- Try things on smaller scale.
- Speed up how fast we can experiment

 ![image](https://github.com/user-attachments/assets/a240f3be-6dce-4cb5-809d-3a3d008c88d2)

 
#### 1. Visualize the image data
Let's write some code to:

- Get all of the image paths using pathlib.Path.glob() to find all of the files ending in .jpg.
- Pick a random image path using Python's random.choice().
- Get the image class name using pathlib.Path.parent.stem.
- And since we're working with images, we'll open the random image path using PIL.Image.open() (PIL stands for Python Image Library).
- We'll then show the image and print some metadata.

```python
import random
from PIL import Image

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img
```

Now Visualize with Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

img_as_array = np.asarray(img)
print(img_as_array.shape)
plt.imshow(img_as_array)

plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
```

#### 2. Transforming Data - Turn target data into tensors

Before we can use our image data with PyTorch we need to:

- Turn it into tensors (numerical representations of our images).
- Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader, we'll call these Dataset and DataLoader for short.

#### 3. Loading image dataset 

#### 3.1 Option 1 - using Image Folder
- we can load image classification data using 'torchvision.datasets.Imagefolder' (https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- This class inherits from DatasetFolder so the same methods can be overridden to customize the dataset.
```python
# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
```


#### 3.2 Turn loaded images into DataLoader's


Turning our Dataset's into DataLoader's makes them iterable so a model can go through learn the relationships between samples and targets (features and labels).

To keep things simple, we'll use a batch_size=1 and num_workers=1.

What's num_workers?

Good question.

It defines how many subprocesses will be created to load your data.

Think of it like this, the higher value num_workers is set to, the more compute power PyTorch will use to load your data.

Personally, I usually set it to the total number of CPUs on my machine via Python's os.cpu_count(). its 16 oin my system
```python
from torch.utils.data import DataLoader
BATCH_SIZE = 4
NUM_WORKER = 1

# turn datasets into iterables. (batches)

train_dataloader = DataLoader(train_data, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)
```

#### 3.3 Option 2 : Creating Custom ImageFolder - Create a custom Dataset to replicate ImageFolder
1. Want to be able to load images from file
2. want to be able to get class names from dataset
3. want to be able to get classes as Dictionary from datasets
we used pre-existing ImageFolder function.

Pros -

- can create a dataset out of almost anything
- Not limited to Pytorch pre-built dataset fnctions

Cons -
- doesnt mean it will work
- we have to write more code and this might be prone to errors


Creating a helper function to get class names
Let's write a helper function capable of creating a list of class names and a dictionary of class names and their indexes given a directory path.

To do so, we'll:

- Get the class names using os.scandir() to traverse a target directory (ideally the directory is in standard image classification format).
- Raise an error if the class names aren't found (if this happens, there might be something wrong with the directory structure).
- Turn the class names into a dictionary of numerical labels, one for each class.

------
**We'll build one to replicate the functionality of torchvision.datasets.ImageFolder().**

This will be good practice, plus, it'll reveal a few of the required steps to make your own custom Dataset.

It'll be a fair bit of a code... but nothing we can't handle!

Let's break it down:

1. Subclass torch.utils.data.Dataset.
2. Initialize our subclass with a targ_dir parameter (the target data directory) and transform parameter (so we have the option to transform our data if needed).
3. Create several attributes for paths (the paths of our target images), transform (the transforms we might like to use, this can be None), classes and class_to_idx (from our find_classes() function).
4. Create a function to load images from file and return them, this could be using PIL or torchvision.io (for input/output of vision data).
5. Overwrite the __len__ method of torch.utils.data.Dataset to return the number of samples in the Dataset, this is recommended but not required. This is so you can call len(Dataset).
6. Overwrite the __getitem__ method of torch.utils.data.Dataset to return a single sample from the Dataset, this is required.
7. Will create len, getitem, we get class list and dictionary of keys and values of class names.

```python
"""  
target_dir = to get the data from
paths = paths of our images
transform - transform the way you like
classes 0 list of traget classes
class_to_idx = a dict of the target classes mapped to int labels

method - 
1. load_images - read and open image at ith index
2. __len__ method - length of dataset
3. __getitem()__ - return given sample when passed index
"""

class ImageFolderCustom(Dataset):

    def __init__(self, target_dir:str,
                 transform = None) -> None:
        super().__init__()

        # Get all image paths
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))

        # set up transform
        self.tranform = data_transform

        # create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(target_dir)
        # one is list and nother one is map

    
    def load_images(self, index:int)-> Image.Image:

        img_path = self.paths[index]
        img = Image.open(img_path)
        return img
        

    def __len__(self)->int:
        return len(self.paths)

    "Returns one sample of data, data and label (X, y)."
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # class name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
```
**However, now we've written it once, we could move it into a .py file such as data_loader.py along with some other helpful data functions and reuse it later on.**

#### Transforming and Augmenting images -
- Data augmentation is the process of altering your data in such a way that you artificially increase the diversity of your training set.
- Machine learning is all about harnessing the power of randomness and research shows that random transforms (like transforms.RandAugment() and transforms.TrivialAugmentWide()) generally perform better than hand-picked transforms.
- apply image transformations to training data.
- Training a model on this artificially altered dataset hopefully results in a model that is capable of better generalization (the patterns it learns are more robust to future unseen examples).
  
![image](https://github.com/user-attachments/assets/dadc4c76-1f80-40e6-9ea9-93f1a9cc37bd)


```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense range[0,31]
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

```

#### Impact on the Training Dataset
- While the dataset size on disk remains the same, the number of unique images seen by the model during training effectively increases due to the augmentations. This helps the model generalize better by exposing it to a wider variety of data.
- Advanced Augmentation with Libraries
For more advanced data augmentation techniques, you can use libraries like **albumentations**, which offer a wide range of transformations.
- Generalizes on unseen data.
![image](https://github.com/user-attachments/assets/1bd5da2a-626a-4ce1-b8bc-33c499b8adf4)

#### Key Features of TrivialAugmentWide
- Tuning-Free: Unlike other augmentation methods that require extensive hyperparameter tuning to find the optimal augmentation policy, TrivialAugmentWide does not need any tuning. It simplifies the augmentation process by removing the need for manual adjustments.
- Random Augmentation: It applies a single random augmentation operation to each image, chosen from a set of predefined augmentations. The intensity of the augmentation is also chosen randomly.
-Wide Range of Operations: The predefined set of augmentations includes a wide variety of operations such as rotations, translations, flips, color adjustments, and more. This ensures that the model is exposed to diverse transformations during training, improving its robustness and generalization.



#### Model 0 : Tiny VGG without data Augmentation
1. Load train/test data transform the imgs to tenosrs and perfrom.
2. Resize = 64x64, Tiny VGG in channels and then transform to tensors
3. Train/test data from ImageFolder(default)/ Custom Made
4. Lets build Tiny VGG Architecture
5. torchinfo to print summary of our model - Use torchinfo to get an idea of the shapes going through our model¶
6. Create Training + Testing loop Functions
7. Train_step() - takes in a model and dataloader and trains the model on the dataloader.
8. test_step() - takes in a model and dataloader and evaluate the model on the dataloader.
9. Create Train Function to Train and Evaluate our Models = Combines train_step() +  test_step()
10. Train the model putting it all together.
11. Plot loss curve - track models progress over time - use train_loss, test_loss, test_acc, train_acc

```python
# 1. LOAD TRAIN/TEST DATA
from torchvision import datasets
train_data = datasets.ImageFolder(root = train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root = test_dir,
                                  transform=data_transform,
                                  target_transform=None)
train_data, test_data

```


```python
# 2. TRANFORMS - WITHOUT DATA AUGMENTATION
simple_transform = transforms.Compose(
    [
        transforms.Resize(size=(64,64)),
          # input to Tiny VGG
        transforms.ToTensor()
    ]
)

```

```python
# 3. DataLoader - Iterators to datsets in terms of batches of images
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_data = DataLoader(dataset=train_data,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        shuffle=True)

test_data = DataLoader(dataset=test_data,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS
                        shuffle=False)
```

![image](https://github.com/user-attachments/assets/ba11de23-f0f5-4f25-8575-8621b453cfba)

```python
# TRAIN LOOP
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):
    
    model.to(device)
    model.tran()
    train_loss, train_acc = 0,0

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        # 1. forward pass
        pred_y = model.forward(X)

        # 2. Loss fun
        loss = loss_fn(pred_y, y)
        train_loss += loss.item()

        # 3. Optimizer
        optimizer.zero_grad()

        # 4. backpropogation
        loss.backward()

        # 5. Optimizer
        optimizer.step()

        # Calculate accuracy
        # y_pred_class = pred_y.argmax(dim=1)
        y_pred_class = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len()

    
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc

```


**Note on Logits and Softmax** : In the context of neural networks, particularly those used for classification tasks, the forward pass often outputs logits rather than probabilities. Logits are the raw, unnormalized scores produced by the last layer of the network before any activation function like softmax is applied.


```python
# TEST LOOP
def test_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):
    

    model.to(device)
    model.eval()

    test_loss, test_acc = 0,0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)

            y_pred = model.forward(X)

            # 2. Loss fun
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)# Softmax converts logits to probabilities.
            test_acc += (y_pred_class == y).sum().item()/len()

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    return test_loss, test_acc
```

```python
# Train Loop
import torch.utils
import torch.utils.data

# Create Train Function to Train and Evaluate our Models = Combines train_step() +  test_step()

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), # defalult multi-class classifications
          epochs:int = 5,
          device=device):
    
    # 2. Make dict of results - empty

    results = {
        "train_loss": [],
        "test_loss" : [],
        "train_acc": [],
        "test_acc":  []
    }
    # 3. Loop through training + testing steps for a number of epochs

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)

        test_loss, test_acc = test_step(model, train_dataloader, loss_fn, device)

        # 4. update res dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)

        # 5. whats happening ? 
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

    return results

```

```python
# Putting it all together
torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 10

# recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

# setup Loss function + optimizer
optimizer = torch.optim.Adam(model_0.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

#start the timer

from timeit import default_timer as timer
start_time = timer()

# Train model_0 without data augmentation

model_0_results = train(model_0, train_dataloader_simple, test_dataloader_simple, optimizer, loss_fn, NUM_EPOCHS, device)

# end timre

end_time = timer()

total_training_time = end_time - start_time
print(f"Total Training time = {total_training_time:.3f} seconds")
```
![image](https://github.com/user-attachments/assets/39ed67af-4fe9-4236-b86b-9eb6be35a703)

![image](https://github.com/user-attachments/assets/0d6fc6ab-44d9-4c4f-9ce5-f875f46549ab)


```python
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
```
![image](https://github.com/user-attachments/assets/970975c9-1b95-4d13-905a-c85216a418ec)



#### Balance between Overfitting + Underfitting, How to deal with each ?

![image](https://github.com/user-attachments/assets/e2662f25-b7e6-461d-8c9e-a55273922202)

Left: If your training and test loss curves aren't as low as you'd like, this is considered underfitting. *Middle: When your test/validation loss is higher than your training loss this is considered overfitting. Right: The ideal scenario is when your training and test loss curves line up over time. This means your model is generalizing well. 

![image](https://github.com/user-attachments/assets/b9e3e8da-dea0-4ebc-ae7c-c01724aedac5)

1. An overfitting model is one that performs better (often by a considerable margin) on the training set than the validation/test set.
2. If your training loss is far lower than your test loss, your model is overfitting.
3. The other side is when your training and test loss are not as low as you'd like, this is considered underfitting.

![image](https://github.com/user-attachments/assets/a77abc38-9970-43c4-8c86-8ad23f5c587e)
![image](https://github.com/user-attachments/assets/ba5c1b63-9748-400b-b17a-56351752cb46)

- One way we are going to try now is **data augmentation** for underfitting -

#### tinyVGG with Data Augmentation - 
- Lets try another experiment
- same model use different transform
- load data and dataloders with data augmenttaion
- Use same Tiny VGG
- Train with same function with train_step + test_step
- Results - still underfitting the data see below image

![image](https://github.com/user-attachments/assets/4af5855b-3fc9-4b6a-8daf-56cb0eff682f)

#### How to compare results of 2 models
1. hard coding
2. Pytorch + Tensorboard
3. Weights + Bias
4. MLFlow

![image](https://github.com/user-attachments/assets/81ff2633-06ae-472d-955c-c6b409c6fcaf)

Build something like - https://nutrify.app/

#### Making Prediction on custom image on trained Pytorch model - not in train/test dataset
- Convert image to tensors and pass it thorugh the inference model
- make sure its in sam format as your previous trained data
- torch.float32 tensor - convert from uint8 to floa32 and then toTensor()
- shape 64 x 64 x 3 - change the shape, resize before sending it to trained model
- on the right device - convert from cpu to gpu
- we can read image using torchvision
  
- PyTorch's torchvision has several input and output ("IO" or "io" for short) methods for reading and writing images and video in torchvision.io.

- Since we want to load in an image, we'll use torchvision.io.read_image().

- This method will read a JPEG or PNG image and turn it into a 3 dimensional RGB or grayscale torch.Tensor with values of datatype uint8 in range [0, 255].
  
```python
# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 

# Print out image data
# print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")

"""
Custom image shape: torch.Size([3, 4032, 3024])
Custom image dtype: torch.float32
"""

```

- Our model was trained on images with shape [3, 64, 64], whereas our custom image is currently [3, 4032, 3024].
```python
# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out original shape and new shape
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")

```
```python
model_1.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # Make a prediction on image with an extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
"""
Custom image transformed shape: torch.Size([3, 64, 64])
Unsqueezed custom image shape: torch.Size([1, 3, 64, 64])
"""

print(custom_image_pred)
# tensor([[-0.3249, -0.3194, -0.3367]], device='cuda:0')
```

**Convert these logits into -> prediction probabilities -> labels**
```python
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(custom_image_pred_probs)
# tensor([[0.3340, 0.3359, 0.3301]], device='cuda:0')
# the probabilities are spread out, we need to assign label to it

custom_image_pred_lable = torch.argmax(custom_image_pred_probs, dim=1).cpu()
print(custom_image_pred_lable)
# tensor([1], device='cuda:0')

custom_image_pred_class = train_data.classes[custom_image_pred_lable]
print(custom_image_pred_class)

```
**Takeaways**
![image](https://github.com/user-attachments/assets/ccaa9029-2feb-4cd3-a93b-ea757a0ab2d7)


### 05. PyTorch Going Modular
---
- I have written nice code in a notebook, can I reuse it elsewhere?

**What is going modular?**

Going modular involves turning notebook code (from a Jupyter Notebook or Google Colab notebook) into a series of different Python scripts that offer similar functionality.

For example, we could turn our notebook code from a series of cells into the following Python files:

- data_setup.py - a file to prepare and download data if needed.
- engine.py - a file containing various training functions.
- model_builder.py or model.py - a file to create a PyTorch model.
- train.py - a file to leverage all other files and train a target PyTorch model.
- utils.py - a file dedicated to helpful utility functions.

![image](https://github.com/user-attachments/assets/8b18eab5-2420-410c-b2a6-fe96d815068d)

![image](https://github.com/user-attachments/assets/a65447fb-cb32-4791-ba5e-a026ce819388)

![image](https://github.com/user-attachments/assets/af29d53b-d40f-4388-9bc6-1cec95dbf0e5)

![image](https://github.com/user-attachments/assets/94cec09d-02b3-4242-9540-194f1b9c4dfb)

**Goal:** we are going to write a train.py script that would be able to train our model.

- notebooks are great way to visualize and experiment.
- we can shift that code in python scripts and reuse the code blocks whereever necessary.
- will convert the code from cell mode to python scripts

#### Lets see what are we going to cover now ?

The main concept of this section is: turn useful notebook code cells into reusable Python files.
- Transorming data for use with a model
- Loading custom data with pre-built functions
- Building food vision mini to classify images
- Turning useful notebook code into Python scripts
- Training a pytorch model from command line
  
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/

  
















