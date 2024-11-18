import random
random.seed(2021)
from numpy.random import seed
seed(2021)
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(2021)

import random as python_random
python_random.seed(2021)

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import IPython
#from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperModel
from numpy import array
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Input, Conv1D, Layer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import time
from contextlib import redirect_stdout

os.environ['TF_DETERMINISTIC_OPS'] = '0'

# Parameters
# Parameters
SCALE_ON = 1
OutputsFolderName = "OUTPUT_FILES"
window_size = [5, 8, 11]
forecast_horizon = 5
NUM_FEATURES = 1
MaxTrials = 30 #30
EpochsTuning = 5 #5
EpochsTraining = 500
ValidationSplit = 0.2
PATIENCE1 = 10
PATIENCE2 = 50

REMOVE_THIS_MANY_ROWS_FROM_END = 1

# Create necessary directories for outputs
parentDirectory = os.path.dirname(os.path.dirname(__file__))
outFolder = os.path.join(parentDirectory, "out", OutputsFolderName)
historydir = os.path.join(outFolder, 'history')
modelsdir = os.path.join(outFolder, 'models')
summarydir = os.path.join(outFolder, 'summary')
predictionsdir = os.path.join(outFolder, 'predictions')

if not os.path.exists(outFolder):
    os.makedirs(outFolder)
    os.makedirs(historydir)
    os.makedirs(modelsdir)
    os.makedirs(summarydir)
    os.makedirs(predictionsdir)

# Load data
DataFile = "DATASET.csv"
data_file_path = os.path.join(parentDirectory, "d", DataFile)
dataset_train_full = pd.read_csv(data_file_path)

columns = dataset_train_full.columns
areaNames = dataset_train_full.iloc[:, 2]

# Define actual forecast period for calculating errors
ActualDataStart = columns.get_loc("2020")
ActualDataEnd = columns.get_loc("2024")
ActualTruth = dataset_train_full.iloc[:, ActualDataStart:(ActualDataEnd + 1)]

# Data for training (until the jump-off year)
from_Column_ERPs = columns.get_loc("1971")
to_Column_ERPs = columns.get_loc("2019")
dataset_for_forecasts = dataset_train_full.iloc[:, from_Column_ERPs:(to_Column_ERPs + 1)]

# Transpose the array so each column is a small area
y_full = dataset_for_forecasts.T
y_multiSeries = pd.DataFrame(y_full[:(len(y_full))])

# Remove remainder row from training data
y_multiSeries = y_multiSeries.iloc[:, :(y_multiSeries.shape[1] - REMOVE_THIS_MANY_ROWS_FROM_END)]
y_multiSeries = np.array(y_multiSeries)

# Keep the original dataset for scaling purposes
OriginalDataForScaling = y_multiSeries.copy()

# Debugging: check the shape of the data
print("Shape of y_multiSeries:", y_multiSeries.shape)

# Multi-Head Attention Layer
class MultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)

    def call(self, query, value):
        # Apply multi-head attention
        attention_output = self.attention(query=query, value=value)
        return attention_output

class CNN_LSTM_Attention_HyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)

        # CNN stream
        x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                                       kernel_size=3, activation='relu')(inputs)
        x_cnn = tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                                       kernel_size=3, activation='relu')(x_cnn)
        x_cnn = tf.keras.layers.Flatten()(x_cnn)

        # LSTM stream (two separate LSTM layers)
        lstm_out_1 = tf.keras.layers.LSTM(units=hp.Int('units_1', min_value=128, max_value=512, step=32),
                                          return_sequences=True, activation='relu')(inputs)
        lstm_out_2 = tf.keras.layers.LSTM(units=hp.Int('units_2', min_value=128, max_value=512, step=32),
                                          return_sequences=True, activation='relu')(lstm_out_1)

        # Apply Multi-Head Attention
        multi_head_attention = MultiHeadAttention(
            num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=2),
            key_dim=hp.Int('key_dim', min_value=16, max_value=64, step=16)
        )
        attention_output = multi_head_attention(query=lstm_out_2, value=lstm_out_2)

        # Combine CNN and attention outputs
        x = tf.keras.layers.concatenate([x_cnn, tf.keras.layers.Flatten()(attention_output)])

        # Dense layers
        x = tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=32),
                                  activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

        return model
def lstm_data_transform(x_data, y_data, num_steps, forecast_horizon):
    X, y = [], []
    for i in range(x_data.shape[0]):
        end_ix = i + num_steps
        if end_ix - 1 >= x_data.shape[0]:
            break
        seq_X = x_data[i:end_ix]
        X.append(seq_X)
    x_array = np.array(X)
    return x_array

def lstm_full_data_transform(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y = [], []
    print("x_data shape before transformation:", x_data.shape)

    if scale_on == 1:
        x_data = (x_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())
        y_data = (y_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())

    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            end_ix = i + num_steps
            if end_ix >= x_data.shape[0]:
                break
            seq_X = x_data[i:end_ix, j]
            seq_y = y_data[end_ix, j]
            X.append(seq_X)
            y.append(seq_y)
    x_array = np.array(X)
    y_array = np.array(y)
    x_array = x_array.reshape(x_array.shape[0], x_array.shape[1], 1)
    return x_array, y_array

# Data transformation for validation
def lstm_full_data_transform2(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y, Xval, Yval, Xtrain, Ytrain = [], [], [], [], [], []
    nVal = 3

    if scale_on == 1:
        x_data = (x_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())
        y_data = (y_data - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())

    for j in range(x_data.shape[1]):
        for i in range(x_data.shape[0]):
            end_ix = i + num_steps
            if end_ix >= x_data.shape[0]:
                break
            seq_X = x_data[i:end_ix, j]
            seq_y = y_data[end_ix, j]
            X.append(seq_X)
            y.append(seq_y)

        # Keep the last nVal sequences for validation
        Xval.append(X[-nVal:])
        Yval.append(y[-nVal:])
        Xtrain.append(X[:-nVal])
        Ytrain.append(y[:-nVal])
        X, y = [], []

    x_array = np.array(Xtrain).reshape(-1, num_steps, 1)
    y_array = np.array(Ytrain).reshape(-1)
    Xval = np.array(Xval).reshape(-1, num_steps, 1)
    Yval = np.array(Yval).reshape(-1)
    return x_array, y_array, Xval, Yval

# Callback for clearing outputs after training
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

# Function to calculate Median Absolute Percentage Errors
def calcMAPEs(TheForecast, ActualTruth):
    MAPES = np.abs(np.array(TheForecast) - np.array(ActualTruth)) / np.array(ActualTruth) * 100
    medianMapes = np.median(MAPES, axis=0)
    meanMAPES = np.mean(MAPES, axis=0)
    MAPES_greater10 = np.count_nonzero(MAPES > 10, axis=0) / (MAPES.shape[0]) * 100
    MAPES_greater20 = np.count_nonzero(MAPES > 20, axis=0) / (MAPES.shape[0]) * 100
    ErrorSummary = pd.DataFrame([medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20])
    ErrorSummary = ErrorSummary.rename(index={0: "medianMapes", 1: "meanMAPES", 2: " >10%", 3: " >20%"})
    OneLine = pd.DataFrame(np.hstack((medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20)).ravel())
    MAPES = pd.DataFrame(MAPES)
    return MAPES, ErrorSummary, OneLine
def ArraysWithErrors(OneLine,ModelTypeName,num_steps_w,forecast_horizon,TheForecast,MAPES_pd,ErrorSummary,areaNames):
    
    ConcatArr=pd.concat([TheForecast.reset_index(drop=True),MAPES_pd.reset_index(drop=True),ErrorSummary.reset_index(drop=True)],axis=1)
    
    #let's add a line with details into our summary forecasts
    df=pd.DataFrame(columns=ConcatArr.columns)
    df.append(pd.Series(name='info'))
    df.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    df2=pd.concat([df,ConcatArr])

    newIndex=pd.concat([pd.Series(["info"]),areaNames])
    ind=pd.DataFrame(newIndex)
    ind=ind.rename(columns={ind.columns[0]:"SA2"})
    df2['SA2_names']=ind['SA2']
    df2.set_index('SA2_names',inplace=True)

    #Let's add details to the error summary so that when we aggregate them we know
    #which one is which
    df_errorArray=pd.DataFrame(columns=ErrorSummary.columns)
    df_errorArray.append(pd.Series(name='info'))
    df_errorArray.loc[0,:]=ModelTypeName+" "+str(num_steps_w)+" steps"
    ErrorSummary=pd.concat([df_errorArray,ErrorSummary])
    ErrorSummary=ErrorSummary.rename(index={0:"info"})
    
    return ErrorSummary,df2


# Function to perform forecasts
def runForecasts(dataset_train_full, num_features, num_steps, forecast_horizon, filename1, model1, SCALE_ON, OriginalDataForScaling, outFolder, areaNames):
    for c_f in range(len(dataset_train_full)):
        if (c_f % 100 == 0):
            print(c_f)
        if c_f == 0:
            # We will measure execution time for each loop
            ETime = pd.DataFrame(index=range(len(dataset_train_full)), columns=['eTime'])

        start1 = time.time()

        columns = dataset_train_full.columns
        from_Column_ERPs = columns.get_loc("1971")
        to_Column_ERPs = columns.get_loc("2019")
        
        y_full = dataset_train_full.iloc[c_f, from_Column_ERPs:(to_Column_ERPs + 1)]
        y = y_full.reset_index(drop=True)
        Area_name = areaNames.iloc[c_f]
        ActualData = pd.DataFrame(y_full)
        ActualData = ActualData.rename(columns={"0": Area_name})
        ActualData_df = ActualData.T

        if SCALE_ON == 1:
            y = (y - OriginalDataForScaling.min()) / (OriginalDataForScaling.max() - OriginalDataForScaling.min())

        y = np.array(y)

        x_new = lstm_data_transform(y, y, num_steps=num_steps, forecast_horizon=forecast_horizon)
        
        x_train = x_new

        test_input = x_new[-1:]
        test_input_prescaled = test_input
        temp1 = test_input

        test_input = test_input.reshape(1, num_steps, 1)

        PredictionsList = []
        LSTMPredictionsList = []

        # Perform rolling predictions
        for i in range(forecast_horizon):             
            test_input = test_input.reshape(1, num_steps, 1)
            test_input = np.asarray(test_input).astype('float32')
            test_output = model1.predict(test_input, verbose=0)
            
            test_input[0, :(num_steps-1)] = test_input[0, 1:]
            LSTMPredictionsList.append(test_output.reshape(1))
    
            test_input[0, (num_steps-1)] = test_output
            PredictionsList.append(test_output.reshape(1))
                
        PredictionsList = np.array(PredictionsList)
        predictions = PredictionsList.reshape(forecast_horizon, num_features)
        
        if SCALE_ON == 1:
            df_predict = predictions * (OriginalDataForScaling.max() - OriginalDataForScaling.min()) + OriginalDataForScaling.min()
        else:
            df_predict = predictions
        
        # Label the predictions for each area in a larger dataframe
        df_predict = pd.DataFrame(df_predict)
        
        for count_through_columns in range(num_features):
            df_predict = df_predict.rename(columns={df_predict.columns[count_through_columns]: Area_name})

        temp_predict_df = df_predict.T        
        
        if c_f == 0:
            full_array_SA2s = temp_predict_df.iloc[[0]]
        else:
            frames_SA2 = [full_array_SA2s, temp_predict_df.iloc[[0]]]
            full_array_SA2s = pd.concat(frames_SA2)

        endLoop = time.time()
        ETime.iat[c_f, 0] = endLoop - start1

    timestr = time.strftime("%Y%m%d")
    FullArrayLocation = filename1 + timestr + ".csv"

    parentDirectory = os.path.dirname(os.path.dirname(__file__))
    file_path_df = os.path.join(outFolder, 'predictions', FullArrayLocation)

    full_array_SA2s.to_csv(file_path_df)

    # Return the predictions
    return full_array_SA2s

# Now we'll replace the LSTM models with the CNN+LSTM model with Multi-Head Attention
ExperimentalInfo = []
SummaryArrayFlag = 0
ModelTypeName = "MODEL NAME"

for current_window in range(len(window_size)):
    num_steps_w = window_size[current_window]
    INPUT_SHAPE = (num_steps_w, NUM_FEATURES)
    a = time.time()

    # Reset seeds
    seed(2021)
    tf.random.set_seed(2021)
    python_random.seed(2021)
    random.seed(2021)

    # Transform the data for validation and tuning
    x_new_w, y_new_w, x_new_w_VAL, y_new_VAL = lstm_full_data_transform2(
        y_multiSeries, y_multiSeries, num_steps_w, forecast_horizon, SCALE_ON, OriginalDataForScaling)

    # Initialize CNN + LSTM model with Multi-Head Attention
    CNN_LSTM_Attention_hypermodel = CNN_LSTM_Attention_HyperModel(input_shape=INPUT_SHAPE)

    projectName = "CNN_LSTM_" + str(num_steps_w)
    bayesian_opt_tuner = BayesianOptimization(
        CNN_LSTM_Attention_hypermodel,
        objective='mse',
        max_trials=MaxTrials,
        seed=2021,
        executions_per_trial=1,
        directory=os.path.normpath('C:/keras_tuning2'),
        project_name=projectName,
        overwrite=True
    )

    # Tune the model
    bayesian_opt_tuner.search(x_new_w, y_new_w, epochs=EpochsTuning, validation_data=(x_new_w_VAL, y_new_VAL),
                              verbose=2, callbacks=[ClearTrainingOutput(),
                              keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE1)])

    # Get the best model
    bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
    Model1 = bayes_opt_model_best_model[0]

    # Reduce learning rate if validation error plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=PATIENCE1, min_lr=0.000005, verbose=2)

    # Train the model with the best hyperparameters
    history = Model1.fit(x_new_w, y_new_w, epochs=EpochsTraining, validation_data=(x_new_w_VAL, y_new_VAL),
                         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE2, mode='auto', restore_best_weights=True), reduce_lr],
                         verbose=2)

    hist_getBest = Model1.history.history['val_loss']
    n_epochs_best = np.argmin(hist_getBest) + 1

    # Define folders for saving models, history, and predictions
    ModelSaveName2 = ModelTypeName + "_" + str(num_steps_w)
    save_model_here = os.path.join(modelsdir, (ModelSaveName2))
    SaveHistoryHere = os.path.join(historydir, (ModelSaveName2 + ".csv"))
    SaveSummaryHere = os.path.join(summarydir, (ModelSaveName2 + ".txt"))

    # Save the trained model
    Model1.save(save_model_here)

    # Save training history
    history_df = pd.DataFrame(history.history)
    with open(SaveHistoryHere, mode='w') as f:
        history_df.to_csv(f)

    # Save model summary
    with open(SaveSummaryHere, 'w') as f:
        with redirect_stdout(f):
            Model1.summary()

    # Run forecasts with the trained model
    TheForecast = runForecasts(dataset_for_forecasts, NUM_FEATURES, num_steps_w, forecast_horizon, ModelSaveName2, Model1, SCALE_ON, OriginalDataForScaling, outFolder, areaNames)

    # Calculate errors
    MAPES, ErrorSummary, OneLine = calcMAPEs(TheForecast, ActualTruth)
    OneLine = OneLine.rename(columns={OneLine.columns[0]: ModelTypeName + "_" + str(num_steps_w) + "_steps_" + str(forecast_horizon) + "_year"})
    ErrorSummary, df2 = ArraysWithErrors(OneLine, ModelTypeName, num_steps_w, forecast_horizon, TheForecast, MAPES, ErrorSummary, areaNames)

    # Save error summary
    ErrorSummaryFilePath = os.path.join(predictionsdir, (ModelSaveName2 + "ErrorSummary.csv"))
    ErrorSummary_df = pd.DataFrame(ErrorSummary.copy())
    with open(ErrorSummaryFilePath, mode='w') as f:
        ErrorSummary_df.to_csv(f)

    # Combine forecasts and errors for summary
    if SummaryArrayFlag == 0:
        SummaryArrayFlag = 1
        FullForecastArray = df2
        FullErrorArray = OneLine
    else:
        FullForecastArray = pd.concat([FullForecastArray, df2], axis=1)
        FullErrorArray = pd.concat([FullErrorArray, OneLine], axis=1)

    # Record time taken for this run
    b = time.time()
    c = b - a
    The_Learning_rate = tf.keras.backend.eval(Model1.optimizer.lr)

    print('Window length: ' + str(num_steps_w) + ', Time taken: ' + str(c) + ' seconds')

    # Save experimental details
    ModelConfig = Model1.optimizer.get_config()
    OurExperiments = [ModelTypeName, ModelSaveName2, forecast_horizon, num_steps_w, NUM_FEATURES, c, ModelSaveName2,
                      outFolder, MaxTrials, EpochsTuning, EpochsTraining, ValidationSplit, PATIENCE1, PATIENCE2, 
                      The_Learning_rate, SCALE_ON, REMOVE_THIS_MANY_ROWS_FROM_END, n_epochs_best, np.array(ModelConfig)]
    ExperimentalInfo.append(OurExperiments)
    
    # Clear session to avoid memory issues
    del Model1, CNN_LSTM_Attention_hypermodel
    keras.backend.clear_session()

import csv
# Save experiment summary
ExperimentalHistorySaveName = os.path.join(outFolder, 'Summary.csv')
with open(ExperimentalHistorySaveName, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ExperimentalInfo)

# Save all forecasts
FullPredictionsTogetherPath = os.path.join(outFolder, 'AllPredictions.csv')
FullForecastArray.to_csv(FullPredictionsTogetherPath)

# Save error summary
FullErrorArrayTogetherPath = os.path.join(outFolder, 'ErrorSummary.csv')
FullErrorArray.to_csv(FullErrorArrayTogetherPath)

print("Experiment completed successfully.")


# Workflow continues for tuning, training, and forecasting using this function
# (Rest of your logic for training and calling runForecasts remains unchanged)
