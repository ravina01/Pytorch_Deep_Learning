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







