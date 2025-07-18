def make_me_a_classifying_RNN_please(filepath: str, label_col: str | int, split: float = 0.2, optimizer_type: str = 'Adam', learning_rate: float = 0.001, epochs: int = 5000, export: bool = True, visualize: bool = True) -> None:
  """
  Inputs a data file to create a classifying recurrent neural network (RNN) on, and returns the training data + exported model in the form of a .pth file

  For now, returns a RNN with the model specifications tailored to any "feature_extracted" datasets (988 explanatory variables), and contains 1 hidden layer with 988 nodes
  ** More customizability to follow, hopefully **

  - filepath: A string containing the file path to the training dataset (ideally a .csv file)
  - label_col: A string or integer that contains either the name or the index of the column containing the labels of the data
  - split: A float specifying the train/test split ratio, referencing the proportion of testing data (default = 0.2)
  - learning_rate: A float representing the learning coefficient for the training of the RNN (default = 0.001)
  - optimizer_type: A string, either "Adam" or "SGD", denoting which optimization function to use, **be literal** (default = "Adam")
      - Adam: Adaptive Movement Estimation, utilizes a measure called "momentum" to optimize based on previous entries
      - SGD: Stochastic Gradient Descent, approximates the local minima of a function effectively and efficiently
  - epochs: An integer to determine how many epochs, or generations, the RNN will train for; must be at least 100 (default = 5000)
  - export: A boolean value deciding whether or not a .pth copy of the model is downloaded, prompts model naming from user (default = True)
  - visualize: A boolean value to determine if a visualization of the loss/accuracy over the epochs is desired (default = True)
  """
  # Checking for valid function inputs ---------------------------------------------------------------------------------
  if type(filepath) != str:
    return print('filepath is not a string')
  
  if type(label_col) != str and type(label_col) != int:
    return print("label_col not a valid input, make sure it is either a string or an index")

  if not(type(split) == float and split > 0 and split < 1):
    return print("split is not a valid input, make sure it is a float between 0 and 1")

  if not(type(optimizer_type) == str and optimizer_type in ['Adam', 'SGD']):
    return print('optimizer_type is not a string, and/or neither "Adam" nor "SGD"')
  
  if not(type(learning_rate) == float and learning_rate > 0 and learning_rate < 1):
    return print("learning_rate is not a valid input, make sure it is a float between 0 and 1")
  
  if not(type(epochs) == int and epochs >= 100):
    return print("epochs is not a valid input, make sure it is an integer of at least 100")

  if type(export) != bool:
    return print("export is not a valid input, make sure it is a boolean value")

  if type(visualize) != bool:
    return print("visualize is not a valid input, make sure it is a boolean value")

  # Importing necessary packages ---------------------------------------------------------------------------------------
  import torch
  from torch import nn
  from torch.autograd import Variable
  from torch.utils.data import DataLoader, TensorDataset

  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split

  # Preparing data for training ----------------------------------------------------------------------------------------
  ## Import data as pandas dataframe, np.array does not translate well
  data = pd.read_csv(filepath)

  ## Convert and split data into explanatory and response (X and y) variables
  columns = list(data.columns)

  if type(label_col) == str:
    columns.remove(label_col)
    
  elif type(label_col) == int:
    columns.pop(label_col)

  X = np.array(data[columns])
  y = np.array(data[label_col])

  ## Convert data into PyTorch-friendly tensors
  X = torch.from_numpy(X).type(torch.float)
  y = torch.from_numpy(y).type(torch.LongTensor)

  ## Train-test split at the `split` proportion
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 9)

  ## Convert training and testing splits into Tensor Datasets (pairs)
  train = TensorDataset(X_train, y_train)
  test = TensorDataset(X_test, y_test)

  ## Data Loader, allows for easy feeding of data into model for train/test
  batch_size = len(X_train)
  train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
  test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

  # Constructing the RNN ----------------------------------------------------------------------------------
  class TestRNN(nn.Module):

    ## Create architecture of RNN (input size, hidden size, hidden layer count, output size)
    def __init__(self, input_size, hidden_size, layer_size, output_size):
      super(TestRNN, self).__init__()

      self.hidden_size = hidden_size
      self.layer_size = layer_size

      ## RNN model via Torch, hidden layers
      self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first = True, nonlinearity = 'relu')
      ## Hidden layer values -> output layer
      self.fc = nn.Linear(hidden_size, output_size)

    ## Define how the RNN moves forward (The math behind the process)
    def forward(self, x):
      h0 = Variable(torch.zeros(self.layer_size, x.size(0), self.hidden_size))
      out, hn = self.rnn(x, h0)
      out = self.fc(out[:, -1, :])
      return out

  # Setting parameters and methods to begin training ------------------------------------------------------
  ## Creating model parameters (SOON TO BE CUSTOMIZEABLE, HOPEFULLY)
  input_size = 988
  hidden_size = 988
  layer_size = 1
  output_size = 3

  ## Define model with above parameters
  model = TestRNN(input_size, hidden_size, layer_size, output_size)

  ## Cross Entropy Loss - (Currently) Best method to represent loss for multiclass categorization
  error = nn.CrossEntropyLoss()

  ## Select optimizer
  if optimizer_type == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
  elif optimizer_type == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

  # Training the RNN model ----------------------------------------------------------------------------------
  ## Empty lists to contain visualizing data
  loss_list = []
  iteration_list = []
  accuracy_list = []
  count = 0

  ## Begin training loop for `epochs` epochs
  for epoch in range(epochs):
      for i, (explanatory, labels) in enumerate(train_loader):

          train  = Variable(explanatory.view(batch_size, 1, input_size))
          labels = Variable(labels.view(-1))
              
          ## Clear gradients
          optimizer.zero_grad()
          
          ## Forward propagation
          outputs = model(train)
          
          ## Calculate softmax and cross entropy loss
          loss = error(outputs, labels)
          
          ## Calculating gradients
          loss.backward()
          
          ## Update parameters
          optimizer.step()
          
          count += 1
          
          ## Enter accuracy of model to visualize for every 100 epochs
          if count % 100 == 0:      
              correct = 0
              total = 0

              ## Iterate through test dataset
              for explanatory, labels in test_loader:
                  explanatory = Variable(explanatory.view(-1, 1, input_size))

                  ## Forward propagation
                  outputs = model(explanatory)
                  
                  ## Get predictions from the maximum value
                  predicted = torch.max(outputs.data, 1)[1]
                  
                  ## Total number of labels
                  total += labels.size(0)
                  correct += (predicted == labels).sum()
            
              accuracy = 100 * correct / float(total)
              
              ## Store loss and iteration for later visualization
              loss_list.append(loss.data)
              iteration_list.append(count)
              accuracy_list.append(accuracy)

              ## Print loss and accuracy for every 500 iterations
              if count % 500 == 0:
                  print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))

  # Finalizing and delivering outputs if desired ---------------------------------------------------------------------
  ## If an outputted model is requested, request model name
  if export:

    model_name = input("Model complete! Enter a name for your .pth model file: ")
    if '.pth' not in model_name:
      torch.save(model, model_name + '.pth')
    else:
      torch.save(model, model_name)

  ## If visualizations are requested
  if visualize:

    ## Loss
    plt.plot(iteration_list,loss_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("RNN: Loss vs Number of iteration")
    plt.show()

    ## Accuracy
    plt.plot(iteration_list,accuracy_list,color = "red")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.title("RNN: Accuracy vs Number of iteration")
    plt.show()

