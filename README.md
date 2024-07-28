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

### Conatins -
#### 00. PyTorch Fundamentals

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
- to reproduce the randomness in the neural network - comes the concept of *random seed*
- 
#### 4. Selecting data from Tensors - Indexing


#### 01. PyTorch Workflow Fundamentals
#### 02. PyTorch Neural Network Classification
#### 03. PyTorch Computer Vision
#### 04. PyTorch Custom Datasets






