from modnet.preprocessing import MODData
from typing import List,Union
import pandas as pd
import pickle 
from keras.models import load_model
from .autoencoder_setup import get_encoder_decoder, get_results_model

from keras.models import Model
def filter_features_dataset(dataset : Union[pd.DataFrame,MODData] = None,
        allowed_features_file : str = None,
        mode : str = 'default',
                  ):
    if mode == "default":
        Xtofilter=dataset
    elif mode == "MODData":
        Xtofilter=dataset.df_featurized

    file_encoded_columns = open(allowed_features_file, 'r')
    lines = file_encoded_columns.readlines()
    columns_encoded=[line.strip('\n') for line in lines]
    ## Xtoencode needs to have all encoded columns in the scaler and autoencoder
    ## if not it will throw error. Please get the missing features.
    ## But columns that are in X but not in columns_encoded are discarded.
   
    Xtofilter=Xtofilter[[c for c in Xtofilter.columns if c in columns_encoded]]
    Xset=set(Xtofilter.columns)
    colset=set(columns_encoded)
    colmissing=list(colset-Xset)
    print(f"All feats missing: {colmissing}")
    tocompute=set([i.split('|')[0] for i in colmissing])
    print(f"You probably need to compute the following features: {tocompute}")
    if len(tocompute) != 0 :
        colset_feats=set([i.split('|')[0] for i in colset])
        if tocompute.issubset(colset_feats):
            ## in this case the features were calculated but there are specific
            ## properties missing, we will include those and fill with 0s for the encoder.
            for missing in colmissing:
                Xtofilter[missing] = 0
        else:
            raise ValueError("Compute the aforementioned features before proceeding!")
    ## reorganizing columns in encoded columns
    Xtofilter = Xtofilter.reindex(columns_encoded, axis=1)

    if mode == "default":
        return Xtofilter
    elif mode == "MODData":
        dataset.df_featurized = Xtofilter
        return dataset

def encode_dataset(dataset : Union[pd.DataFrame,MODData] = None,
        scaler : str = None,
        columns_to_read : str = None,
        autoencoder : str = None,
        save_name : str = None,
        feat_prefix : str = "EncodedFeat",
        mode : str = "default",
        custom_objs : dict = None, 
        compile_model : bool = True,
        # for example: {'vae_loss':function}
        encoder_type : str = "regular",
                  ):
    if mode == "default":
        Xtoencode=dataset
        indexes=dataset.index
    elif mode == "MODData":
        Xtoencode=dataset.df_featurized
        indexes = dataset.df_featurized.index
    ## filtering features to encode dataset
    Xtoencode = filter_features_dataset(dataset=Xtoencode,allowed_features_file=columns_to_read)
    ## scaler data
    t=pickle.load(open(scaler,"rb"))
    Xtoencode = t.transform(Xtoencode)
    print(Xtoencode)
    if custom_objs is not None:
        if compile_model == True:
            autoencoder = load_model(autoencoder, 
                                    custom_objects=custom_objs)
        else:
            autoencoder = load_model(autoencoder, 
                                    custom_objects=custom_objs,
                                    compile=False)
    else:
        if compile_model == True:
            autoencoder = load_model(autoencoder)
        else:
            autoencoder = load_model(autoencoder, compile=False)
    if encoder_type == "regular":
        # if there is conflicting name this line may fix it.
        # The name "input_1" is used 2 times in the model. All layer names should be unique.
        # autoencoder.layers[0]._name='changed_input'
        encoder,decoder = get_encoder_decoder(autoencoder, "bottleneck")
        Xencoded=encoder.predict(Xtoencode)
        
        Xencoded=pd.DataFrame(Xencoded, columns=[f"{feat_prefix}|{idx}" for idx in range(Xencoded.shape[1])],
                            index=indexes)
        if mode == "default":
            pickle.dump(Xencoded, open(save_name,'wb'))
        elif mode == "MODData":
            dataset.df_featurized=Xencoded
            dataset.save(save_name)
        print(Xencoded)
        print('Final shape:', Xencoded.shape)
        print('Summary of results:', get_results_model(autoencoder,Xtoencode))
        return Xencoded
    elif encoder_type == "variational":
        encoder_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-2].output)
        Xencoded = encoder_layer_model.predict(Xtoencode, verbose=False)
        Xencoded=pd.DataFrame(Xencoded, columns=[f"{feat_prefix}|{idx}" for idx in range(Xencoded.shape[1])],
                    index=indexes)
        if mode == "default":
            pickle.dump(Xencoded, open(save_name,'wb'))
        elif mode == "MODData":
            dataset.df_featurized=Xencoded
            dataset.save(save_name)
        print(Xencoded)
        print('Final shape:', Xencoded.shape)
        print('Summary of results:', get_results_model(autoencoder,Xtoencode))
        return Xencoded
    