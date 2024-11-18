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
from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperModel
from numpy import array
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, GRU, Conv1D, MaxPooling1D, Bidirectional, Input
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import time
from contextlib import redirect_stdout

# Define paths and constants
SCALE_ON = 1
OutputsFolderName = "Kor_30x5_500x2_P10_P50"
window_size = [5, 8, 11]
forecast_horizon = 10
NUM_FEATURES = 1
MaxTrials = 5
EpochsTuning = 5
EpochsTraining = 100
ValidationSplit = 0.2
PATIENCE1 = 10
PATIENCE2 = 50
REMOVE_THIS_MANY_ROWS_FROM_END = 1

# Define data paths
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

DataFile = "Provinces-V3.csv"
data_file_path = os.path.join(parentDirectory, "data", DataFile)
dataset_train_full = pd.read_csv(data_file_path)
columns = dataset_train_full.columns
areaNames = dataset_train_full.iloc[:, 2]

# Preprocessing steps
ActualDataStart = columns.get_loc("2015")
ActualDataEnd = columns.get_loc("2024")
ActualTruth = dataset_train_full.iloc[:, ActualDataStart:(ActualDataEnd+1)]
from_Column_ERPs = columns.get_loc("1971")
to_Column_ERPs = columns.get_loc("2014")
dataset_for_forecasts = dataset_train_full.iloc[:, from_Column_ERPs:(to_Column_ERPs+1)]

y_full = dataset_for_forecasts.T
y_multiSeries = pd.DataFrame(y_full[:(len(y_full))])
y_multiSeries = y_multiSeries.iloc[:, :(y_multiSeries.shape[1] - REMOVE_THIS_MANY_ROWS_FROM_END)]
y_multiSeries = np.array(y_multiSeries)
OriginalDataForScaling = y_multiSeries.copy()

# Data transformation functions

def lstm_data_transform(x_data, y_data, num_steps, forecast_horizon):
    X, y = list(), list()
    for i in range(x_data.shape[0]):
        end_ix = i + num_steps
        if end_ix-1 >= x_data.shape[0]:
            break
        seq_X = x_data[i:end_ix]
        X.append(seq_X)
    x_array = np.array(X)
    return x_array

def lstm_full_data_transform(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y = list(), list()
    if SCALE_ON == 1:
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

def lstm_full_data_transform2(x_data, y_data, num_steps, forecast_horizon, scale_on, OriginalDataForScaling):
    X, y = list(), list()
    Xval, Yval = list(), list()
    Xtrain, Ytrain = list(), list()
    nVal = 3

    if SCALE_ON == 1:
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

        Xval.append(X[-nVal:])
        Yval.append(y[-nVal:])

        Xtrain.append(X[:-nVal])
        Ytrain.append(y[:-nVal])
        X, y = list(), list()

    x_array = np.array(Xtrain)
    y_array = np.array(Ytrain)
    x_array = x_array.reshape(x_array.shape[0] * x_array.shape[1], x_array.shape[2], 1)
    y_array = y_array.reshape(y_array.shape[0] * y_array.shape[1])

    Xval = np.array(Xval)
    Yval = np.array(Yval)
    Xval = Xval.reshape(Xval.shape[0] * Xval.shape[1], Xval.shape[2], 1)
    Yval = Yval.reshape(Yval.shape[0] * Yval.shape[1])

    return x_array, y_array, Xval, Yval

# Callbacks and utility functions

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

def calcMAPEs(TheForecast, ActualTruth):
    epsilon = 1e-10
    MAPES = np.abs(np.array(TheForecast) - np.array(ActualTruth)) / (np.array(ActualTruth) + epsilon) * 100
    medianMapes = np.median(MAPES, axis=0)
    meanMAPES = np.mean(MAPES, axis=0)
    MAPES_greater10 = np.count_nonzero(MAPES > 10, axis=0) / (MAPES.shape[0]) * 100
    MAPES_greater20 = np.count_nonzero(MAPES > 20, axis=0) / (MAPES.shape[0]) * 100
    ErrorSummary = pd.DataFrame([medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20])
    ErrorSummary = ErrorSummary.rename(index={
        0: "medianMapes",
        1: "meanMAPES",
        2: " >10%",
        3: " >20%",
    })
    OneLine = pd.DataFrame(np.hstack((medianMapes, meanMAPES, MAPES_greater10, MAPES_greater20)).ravel())
    MAPES = pd.DataFrame(MAPES)
    return MAPES, ErrorSummary, OneLine

def ArraysWithErrors(OneLine, ModelTypeName, num_steps_w, forecast_horizon, TheForecast, MAPES_pd, ErrorSummary, areaNames):
    ConcatArr = pd.concat([TheForecast.reset_index(drop=True), MAPES_pd.reset_index(drop=True), ErrorSummary.reset_index(drop=True)], axis=1)
    info_row = pd.DataFrame([[ModelTypeName + " " + str(num_steps_w) + " steps"] * ConcatArr.shape[1]], columns=ConcatArr.columns)
    df2 = pd.concat([info_row, ConcatArr])

    newIndex = pd.concat([pd.Series(["info"]), areaNames])
    ind = pd.DataFrame(newIndex)
    ind = ind.rename(columns={ind.columns[0]: "SA2"})
    df2['SA2_names'] = ind['SA2']
    df2.set_index('SA2_names', inplace=True)

    error_info_row = pd.DataFrame([[ModelTypeName + " " + str(num_steps_w) + " steps"] * ErrorSummary.shape[1]], columns=ErrorSummary.columns)
    ErrorSummary = pd.concat([error_info_row, ErrorSummary])
    ErrorSummary = ErrorSummary.rename(index={0: "info"})
    
    return ErrorSummary, df2

# Function to perform forecasts
def runForecasts(dataset_train_full, num_features, num_steps, forecast_horizon, filename1, model1, SCALE_ON, OriginalDataForScaling, outFolder, areaNames):
    for c_f in range(len(dataset_train_full)):
        if c_f % 100 == 0:
            print(c_f)
        if c_f == 0:
            ETime = pd.DataFrame(index=range(len(dataset_train_full)), columns=['eTime'])
        
        start1 = time.time()

        columns = dataset_train_full.columns
        from_Column_ERPs = columns.get_loc("1971")
        to_Column_ERPs = columns.get_loc("2014")
        
        y_full = dataset_train_full.iloc[c_f, from_Column_ERPs:(to_Column_ERPs+1)]
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

        PredictionsList = list()
        LSTMPredictionsList = list()

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

    return full_array_SA2s

# Define new HyperModel classes for CNN, GRU, CNN-LSTM, and CNN-GRU

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), kernel_size=hp.Int('kernel_size', 2, 5), activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

class GRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = GRU(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(inputs)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

class CNNLSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), kernel_size=hp.Int('kernel_size', 2, 5), activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

class CNNGRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), kernel_size=hp.Int('kernel_size', 2, 5), activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = GRU(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

# Define LSTM models as in the original code

class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu', input_shape=self.input_shape)(inputs)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        return model

class Bidirectional_LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = Bidirectional(LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu', input_shape=self.input_shape))(inputs)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        x = Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        return model

# We will test multiple window sizes to evaluate different models

ExperimentalInfo = list()
SummaryArrayFlag = 0

model_types = [
    ("Simple_LSTM", LSTMHyperModel),
    ("Bidirectional_LSTM", Bidirectional_LSTMHyperModel),
    ("CNN", CNNHyperModel),
    ("GRU", GRUHyperModel),
    ("CNN-LSTM", CNNLSTMHyperModel),
    ("CNN-GRU", CNNGRUHyperModel),
]

for modelTypeName, ModelClass in model_types:
    for current_window in range(len(window_size)):
        if current_window > 0:
            del x_new_w, y_new_w
        
        seed(2021)
        tf.random.set_seed(2021)
        python_random.seed(2021)
        random.seed(2021)

        num_steps_w = window_size[current_window]
        INPUT_SHAPE = (num_steps_w, NUM_FEATURES)

        a = time.time()

        x_new_w, y_new_w, x_new_w_VAL, y_new_VAL = lstm_full_data_transform2(
            y_multiSeries, y_multiSeries, num_steps_w, forecast_horizon, SCALE_ON, OriginalDataForScaling
        )
            
        LSTM_hypermodel = ModelClass(input_shape=INPUT_SHAPE)
        projectName = modelTypeName + "_" + str(num_steps_w)
        
        bayesian_opt_tuner = BayesianOptimization(
            LSTM_hypermodel,
            objective='mse',
            max_trials=MaxTrials,
            seed=2021,
            executions_per_trial=1,
            directory=os.path.normpath('C:/keras_tuning2'),
            project_name=projectName,
            overwrite=True
        )

        bayesian_opt_tuner.search(x_new_w, y_new_w, epochs=EpochsTuning, validation_data=(x_new_w_VAL, y_new_VAL), verbose=2, callbacks=[ClearTrainingOutput(), keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE1)])

        bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)

        Model1 = bayes_opt_model_best_model[0]
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE1, min_lr=0.000005, verbose=2)
        
        history = Model1.fit(x_new_w, y_new_w, epochs=EpochsTraining, validation_data=(x_new_w_VAL, y_new_VAL), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE2, mode='auto', restore_best_weights=True), reduce_lr], verbose=2)

        hist_getBest = Model1.history.history['val_loss']
        n_epochs_best = np.argmin(hist_getBest) + 1

        x_new_w2, x_new_w_VAL2, y_new_w2, y_new_VAL2 = train_test_split(x_new_w_VAL, y_new_VAL, test_size=ValidationSplit, random_state=2021)

        reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE1, min_lr=0.000001, verbose=2)

        history2 = Model1.fit(x_new_w2, y_new_w2, epochs=EpochsTraining, validation_data=(x_new_w_VAL2, y_new_VAL2), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE2, mode='auto', restore_best_weights=True), reduce_lr2], verbose=2)

        outFolder = os.path.join(parentDirectory, "out", OutputsFolderName)
        historydir = os.path.join(outFolder, 'history')
        modelsdir = os.path.join(outFolder, 'models')
        predictionsdir = os.path.join(outFolder, 'predictions')

        timestr1 = time.strftime("%Y%m%d")
        ModelSaveName2 = modelTypeName + "_" + str(num_steps_w)
        save_model_here = os.path.join(modelsdir, (ModelSaveName2))
        Model1.save(save_model_here)
        SaveHistoryHere = os.path.join(historydir, (ModelSaveName2 + ".csv"))
        SaveHistory2Here = os.path.join(historydir, (ModelSaveName2 + "_History2.csv"))

        history_df = pd.DataFrame(history.history)
        history2_def = pd.DataFrame(history2.history)
        
        with open(SaveHistoryHere, mode='w') as f:
            history_df.to_csv(f)

        with open(SaveHistory2Here, mode='w') as f:
            history2_def.to_csv(f)

        SaveSummaryHere = os.path.join(outFolder, "summary", (ModelSaveName2 + ".txt"))
        with open(SaveSummaryHere, 'w') as f:
            with redirect_stdout(f):
                Model1.summary()

        TheForecast = runForecasts(dataset_for_forecasts, NUM_FEATURES, num_steps_w, forecast_horizon, ModelSaveName2, Model1, SCALE_ON, OriginalDataForScaling, outFolder, areaNames)

        MAPES, ErrorSummary, OneLine = calcMAPEs(TheForecast, ActualTruth)
        OneLine = OneLine.rename(columns={OneLine.columns[0]: modelTypeName + "_" + str(num_steps_w) + "_steps_" + str(forecast_horizon) + "_year"})
        ErrorSummary, df2 = ArraysWithErrors(OneLine, modelTypeName, num_steps_w, forecast_horizon, TheForecast, MAPES, ErrorSummary, areaNames)
        
        ErrorSummary_df = pd.DataFrame(ErrorSummary.copy())
        ErrorSummaryFilePath = os.path.join(predictionsdir, (ModelSaveName2 + "ErrorSummary.csv"))
        with open(ErrorSummaryFilePath, mode='w') as f:
            ErrorSummary_df.to_csv(f)

        if SummaryArrayFlag == 0:
            SummaryArrayFlag = 1
            FullForecastArray = df2
            FullErrorArray = OneLine
        else:
            FullForecastArray = pd.concat([FullForecastArray, df2], axis=1)
            FullErrorArray = pd.concat([FullErrorArray, OneLine], axis=1)

        b = time.time()
        c = b - a
        The_Learning_rate = tf.keras.backend.eval(Model1.optimizer.lr)

        print('window length is: ' + str(num_steps_w) + ', the time to run is: ' + str(c))

        ModelConfig = Model1.optimizer.get_config()

        OurExperiments = [modelTypeName, ModelSaveName2, forecast_horizon, num_steps_w, NUM_FEATURES, c, ModelSaveName2, outFolder, MaxTrials, EpochsTuning, EpochsTraining, ValidationSplit, PATIENCE1, PATIENCE2, The_Learning_rate, SCALE_ON, REMOVE_THIS_MANY_ROWS_FROM_END, n_epochs_best, np.array(ModelConfig)]
        ExperimentalInfo.append(OurExperiments)
        del Model1, LSTM_hypermodel
        keras.backend.clear_session()

# Save experimental history and predictions
import csv

ExperimentalHistorySaveName = os.path.join(outFolder, 'Summary.csv')
with open(ExperimentalHistorySaveName, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ExperimentalInfo)

FullPredictionsTogetherPath = os.path.join(outFolder, 'AllPredictions.csv')
FullForecastArray.to_csv(FullPredictionsTogetherPath)

FullErrorArrayTogetherPath = os.path.join(outFolder, 'ErrorSummary.csv')
FullErrorArray.to_csv(FullErrorArrayTogetherPath)
