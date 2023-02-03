from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU, Conv1D, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.spatial.distance import correlation, cosine
import pickle
from matplotlib import pyplot
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from typing import List

# Set the random seed in TensorFlow
tf.random.set_seed(42)
def get_encoder_decoder(model, central_layer_name):
    layer = model.get_layer(central_layer_name)
    encoder_input = Input(model.input_shape[1:], name='input_input') # to avoid same name error
    #encoder_input = Input(shape=model.input_shape[1:])
    encoder_output = encoder_input
    decoder_input = Input(layer.output_shape[1:])
    #decoder_input = Input(shape=layer.output_shape[1:])
    decoder_output = decoder_input

    encoder = True
    for layer in model.layers:
        if encoder:
            encoder_output = layer(encoder_output)
        else:
            decoder_output = layer(decoder_output)
        if layer.name == central_layer_name:
            encoder = False

    encoder_model = Model(encoder_input, encoder_output)
    decoder_model = Model(decoder_input, decoder_output)

    return encoder_model, decoder_model

def get_results_model(model,Xtest):
    y=Xtest
    ypred=model.predict(Xtest)
    resultsmodel=[]
    for metric in [correlation, cosine]:
        metric_result=np.array([metric(y[i],ypred[i]) for i in range(len(y))]).mean()
        print(metric.__name__,metric_result)
        resultsmodel.append(metric_result)
    results_mae=mean_absolute_error(y, ypred)
    resultsmodel.append(results_mae)
    print('MAE:',results_mae)
    rmse_result=np.sqrt(mean_squared_error(y, ypred))
    resultsmodel.append(rmse_result)
    print('RMSE:',rmse_result)
    r2_result=r2_score(y, ypred, multioutput='variance_weighted')
    resultsmodel.append(r2_result)
    print('r2:',r2_result)
    yzeros=np.zeros(y.shape)
    results_rmse0=np.sqrt(mean_squared_error(y, yzeros))
    resultsmodel.append(results_rmse0)
    print('RMSE zero-vector:',results_rmse0)
    resultsmodel = [ float(r) for r in resultsmodel]
    return resultsmodel

def HyperParameterTestEncoding(
        dataset : pd.DataFrame = None,
        prefix_name : str = None,
        bottleneck_ratios : List[float] = None, 
        batch_sizes : List[int] = None ,   # [16,32,64] 
        epochs_list : List[int] = None ,  # [50,100,200]
        loss_functions : List[str] = None , # ['mse','log_cosh']
        learning_rates : List[float] = None, # [0.0005, 0.001,0.005]
        architectures : List[str] = None, # 2n_b default, depends what is implemented.
        savedir : str = './',
        ):
## Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder
## https://openreview.net/forum?id=rkglvsC9Ym
    # Loop through all combinations of hyperparameters
    for architecture in architectures:
      for bottleneck_ratio in bottleneck_ratios:
        for loss in loss_functions:
          for batch_size in batch_sizes:
            for epoch in epochs_list:
              for learning_rate in learning_rates:
                # Build and compile the autoencoder model with the current set of hyperparameters
                if savedir[-1] != '/':
                    savedir+='/'
                TestEncoding( prefix_name = prefix_name,
                              dataset = dataset,
                              compress_ratio = bottleneck_ratio,
                              architecture=architecture,
                              batch_size=batch_size,
                              epochs = epoch,
                              loss = loss,
                              learning_rate = learning_rate,
                              savedir=savedir+f"{prefix_name}_{architecture['arch']}",
                             )
            


def TestEncoding(prefix_name : str = 'Model', 
        dataset : pd.DataFrame = None,
        compress_ratio : list = None,
        architecture : dict = {'arch':'default'},
        savedir : str = '',
        logfile : str = 'EncoderResults.txt', 
        epochs : int = 100,
        learning_rate : float = 0.001,
        batch_size : int = 16, 
        random_state : int = 1,
        loss : str = 'mse',
        **kwargs
        ):     
    if 'n_factor' not in architecture.keys():
        architecture['n_factor'] = ''
    if prefix_name:
        logfile=f"{prefix_name}_{logfile}"
    name_encoder = f"{prefix_name}_{architecture['arch']}{architecture['n_factor']}_cr{compress_ratio}_bs{batch_size}_ep{epochs}_loss_{loss}_lr{learning_rate}"
    try:
        os.mkdir(savedir)
    except:
        print(f"{savedir} already created.")
    Xtoencode=dataset
    ## drop columns all 0 and register them
    columns_todrop = Xtoencode.columns[(Xtoencode == 0).all()]
    if savedir[-1] != '/':
        savedir+='/'
    with open(savedir+"dropped_columns.txt", 'w') as f:
        text=""
        for column in columns_todrop:
            text+=str(column)+"\n"
        f.write(text)
    Xtoencode = Xtoencode.loc[:, Xtoencode.any()]
    print(f"Shape of dataset to encode: {Xtoencode.shape}"  )
    ## save the columns that are encoded
    with open(savedir+"encoded_columns.txt", 'w') as f:
        text=""
        for column in Xtoencode.columns:
            text+=str(column)+"\n"
        f.write(text)
    # split into train test sets
    X_train, X_test, _, _ = train_test_split(Xtoencode, np.zeros(Xtoencode.shape[0]), 
                                             test_size=0.1, random_state=random_state)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    pickle.dump(t,open(savedir+f"Scaler_{prefix_name}.pkl","wb"))
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    # number of input columns
    n_inputs = Xtoencode.shape[1]
    results_filename=savedir+logfile
    if not os.path.exists(results_filename):
        with open(results_filename, 'w') as f:
            f.write(f"# Training {prefix_name} # Initial Number of Features: {n_inputs}\n")
            entries=['architecture','loss_fn','batch_size','epochs','learning_rate',
                    'n_bottleneck_ratio','n_bottleneck', 'train_loss', 'val_loss', 'correlation',
                    'cosine dist', 'MAE', 'RMSE', 'R2', 'RMSE zero-vector']
            text=''
            for i in range(len(entries)):
                text+=f"{entries[i]:>18}|"
            text+='\n'
            f.write(text)

    ## check if encoder already exists in folder, in this case goes to next 
    encoder_path=savedir+f'{name_encoder}_AutoEncoder.h5'
    if os.path.exists(encoder_path):
        print(f'File {name_encoder}_AutoEncoder.h5 exists in folder already, skiping this calculation.')
        return 0
    
    results=[]
    if architecture['arch'] == 'default': 
        n_bottleneck=int(n_inputs*compress_ratio)
        model = create_autoencoder(input_shape=n_inputs,
                                   layers_structure= [n_inputs*2, int(n_bottleneck) ],
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '2n_m2nb_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs*2, int((n_inputs*2+n_bottleneck)/2), 
                           int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '2n_n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs*2, n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == 'n_2n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs, 2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '3n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [3*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == 'n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == '2n_3n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [2*n_inputs, 3*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure,
                                   loss = loss )
    elif architecture['arch'] == '3n_2n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [3*n_inputs, 2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure, 
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == '2n_conv2_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, layers_structure=layers_structure, 
                                   loss = loss, lr=learning_rate, type_architecture = 'convoluted',
                                   conv_reduction = 2,
                                   )

    elif architecture['arch'] == 'custom_n_b':
        class TestFailed(Exception):
            pass
        if architecture['n_factor'] == '':
            raise TestFailed("you need to specify a n_factor in architecture to use custom_n_b.")
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [architecture['n_factor']*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(input_shape=n_inputs, 
                                   layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)
    FailedTraining=False
    try:
        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                            verbose=2, validation_data=(X_test,X_test))
        # plot loss
        print(f"COMPRESSED VECTOR SIZE: {n_bottleneck}")    
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        print(f"Loss in the autoencoder: {history.history['val_loss'][-1]}")
        mse_error_train=history.history['loss'][-1]
        mse_error_val=history.history['val_loss'][-1]
    except Exception as e:
        print(e)
        print("Training of this model failed.")
        mse_error_train='--'
        mse_error_val='--'
        FailedTraining=True
    architecture_details=str(architecture['n_factor'])+architecture['arch']
    results=[ architecture_details, loss, batch_size, epochs, learning_rate,
            np.round(compress_ratio,4), n_bottleneck, mse_error_train, mse_error_val]
    if not FailedTraining:
        results_model=get_results_model(model,X_test)
        pyplot.legend()
        pyplot.savefig(savedir+f"{name_encoder}.png")
        pyplot.clf()
        # save full autoencoder model (without the decoder)
        model.save(savedir+f'{name_encoder}_AutoEncoder.h5')
        # define and save an encoder model (without the decoder)
        encoder,decoder = get_encoder_decoder(model, "bottleneck")
        ## no need to save it loads corrupted after
        # encoder.save(f'{name_encoder}_encoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        # define and save a decoder model (without the decoder)
        ## no need to save it loads corrupted after
        # decoder.save(f'{name_encoder}_decoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        print("Full AutoEncoder")
        model.summary()
        print("Encoder")
        encoder.summary()
        print("Decoder")
        decoder.summary() 
    else:
        results_model=['--']*6        
    results+=results_model
    print(results,[type(i) for i in results])
    with open(results_filename, 'a') as f:
        text=""
        for i in range(len(results)):
            if isinstance(results[i], str):
                text+=f"{results[i]:>18} "
            else:
                text+=f"{round(results[i],8):18} "
                #print(results[i],isinstance(results[i], numbers.Number))
        f.write(text)
        f.write('\n')


def create_autoencoder(input_shape : int = None,
                       layers_structure : list = None,
                       loss : str = 'mse',
                       lr : float = 0.001,
                       type_architecture : str = 'default',
                       conv_reduction : int = 2, ## only if type_architecture is convoluted
                       ):
    
    # Define the number of layers and number of neurons in each layer
    neurons_per_layer = layers_structure # [64, 32, 16, 8]
    num_layers = len(neurons_per_layer)
    n_inputs=input_shape
    # The input layer is the same as the output layer
    input_layer = Input(shape=(n_inputs,))

    # Create the encoder layers
    # encoder_layers = []
    if type_architecture == 'default':
        for i in range(num_layers-1):
            if i == 0:
                e = Dense(neurons_per_layer[i])(input_layer)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(input_layer)))
            else:
                e = Dense(neurons_per_layer[i])(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(encoder_layers[i-1])))
                # encoder_layers.append(BatchNormalization()(Dense(neurons_per_layer[i], activation='relu')(encoder_layers[i-1])))

        # Create the decoder layers
        # decoder_layers = []
        for i in range(num_layers-1, -1, -1):   
            if i == num_layers-1:
                d = Dense(neurons_per_layer[i], name='bottleneck')(e)
                d = BatchNormalization()(d)
                d = ReLU()(d)
            else:
                d = Dense(neurons_per_layer[i])(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)

    elif type_architecture == 'convoluted' :
        for i in range(num_layers-1):
            if i == 0:
                e = Dense(neurons_per_layer[i])(input_layer)
                e = Reshape((-1, 1))(e)
                e = Conv1D(filters=1,strides=2,kernel_size=int(neurons_per_layer[i]/(conv_reduction)))(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                e = Flatten()(e)

            else:
                e = Dense(neurons_per_layer[i])(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)

        # Create the decoder layers
        # decoder_layers = []
        for i in range(num_layers-1, -1, -1):   
            if i == num_layers-1:
                d = Dense(neurons_per_layer[i], name='bottleneck')(e)
                d = Reshape((-1, 1))(d)
                d = Conv1DTranspose(filters=1, kernel_size=int(neurons_per_layer[0]/conv_reduction),
                                    )(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)
                d = Flatten()(d)
            else:
                d = Dense(neurons_per_layer[i])(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)
        

    # The output layer is the same as the input layer
    # output_layer = Dense(n_inputs, activation='linear')(decoder_layers[-1])
    output_layer = Dense(n_inputs, activation='linear')(d)

    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    autoencoder.compile( optimizer=Adam(learning_rate=lr), loss = loss)
    autoencoder.summary()
    return autoencoder
