# CrossSeriesForecast

## Introduction

This is a cross-series training framework for sparse time series prediction. When making predictions for multiple time series and the time series contain a lot of missing values/ zeros, sometimes we can group the time series with similar properties to overcome the data limitation.

Our experiments show that cross-series training is helpful for the product demand and sales forecasting.

## Usage

Here I will show the general way to train and use our cross-series forecast model.

### Initialize Machine Learning Models

We implemented widely used forecasting models based on statsmodels, sklearn and tensorflow which are
  - EST: Exponential Smoothing
  - LinearRegression: Linear Regression
  - SVR: Support Vector Regression
  - RandomForest: Random Forecast
  - MLP: Multi-layer perceptrons
  - RNN: Recurrent Neural Network

For example, the model you are going to use is RNN. You should import it from Forecast package and build it with default parameters.

```
from Forecast import RNN

rnn = RNN.Model()
```
You can use the function named **show_param_list()** to check the available parameters.

```
RNN.show_param_list()
```
You will get

```
-------------------- Model Params --------------------
name [RNN]
n_steps [5]
n_preds [1]
n_neurons [100]
n_inputs [1]
input_features [x]
n_outputs [1]
output_feature [x]
ckpt_path [../data/RNN_ckpt]
-------------------- Train Params --------------------
init_lr [1e-3]
n_epochs [100]
batch_size [32]
lr_decay [0.1]
lr_decay_steps [2]
lr_schedule [learning rate schedule function]
ckpt [False]
training_sample_num [1000]
device [/gpu:0]
```

Next, you can write your own model and trainig parameters by using the type dict, for example:

```
model_params = {
  'input_features': ['sale', 'price'],
  'output_feature': 'sale',
  'n_steps': 10,
  'n_neurons': 500
}

training_params = {
  'init_lr': 1e-4,
  'n_epochs': 50
}
```

And set the parameters to the model

```
rnn.set_model_params(model_params)
rnn.set_training_params(training_params)
```

### Prepare Datasets

Suppose you have **n** days of sales and prices for **m** products. You will can save them in two **[m x n]** matrices called sales and prices respectively. You can split the data for training and testing and save them in dicts with the keys same as the input features you set in the model. We have three examples to show how do we preprocess the data in Forecast/M5.py, Forecast/NN5.py and Forecast/CIF2016.py

```
train_datalist = {'sale': train_sales, 'price': train_prices}
test_datalist = {'sale': test_sales, 'price': test_prices}
```

Then you can get your validation datalist by using the model function **get_val_data**.

```
val_datalist = {
  'sale': rnn.get_val_data(train_datalist['sale'], test_datalist['sale']),
  'price': rnn.get_val_data(train_datalist['price'], test_datalist['price'])
}
```

We use rolling forecasting origin to build our datasets

```
train_set = rnn.create_set(train_datalist)
val_set = rnn.create_set(val_datalist)
```

### Train and use the Model

Then, you can train the model by using the train_set and val_set

```
rnn.train(train_set, val_set)
```

After training, you can save the model parameters by using **save_model_params** for future use.

```
rnn.save_model_params('./rnn.json')
```

When you want to reuse a model

```
rnn = RNN.Model()
rnn.load_model_params('./rnn.json')
```

You can use **get_preds** to get the predictions with validated input

```
pred = rnn.get_preds(X)
```

### Evaluation

**Forecast/Util.py** provided four types of metrics to evaluate your model performance

- eva_R(pred, y): R squared value
- eva_NMSE(pred, y): Normalized Mean Squared Error
- eva_NMAE(pred, y): Normalized Mean Absolute Error
- eva_SMAPE(pred, y): Symmetric Mean Absolute Percentage Error

### Grid Search

Util also prove grid search function **grid_search_model_params**.

We provide serveral scripts (e.g. Run_M5_FOODS1_CA1_RNN_Global.py) for guidance.

After grid searching, you can use **collect_best_model_preds** to collect the results from the models wth the best performance.
