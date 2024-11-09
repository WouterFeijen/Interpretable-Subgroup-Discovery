import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.metrics import mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LeakyReLU,Reshape,Input



def reduce_dimensionality(data, reduction_method,
                          catColumns=[], numColumns=[], 
                          nFeatures=5, minSize=1, nEpochs=100,
                          deleteOld=True, verbose=False):
    """
    Function which takes data and a reduction method as input and reduces the dimensionality of the data using reduction_method in the reduce_with function. 

    Attributes:
    data: Dataset that dimension reduction ought to be performed on
    catColumns: List of categorical column names that are to be encoded
    numColumns: List of numerical column names that are to be reduced
    nFeatures: Number of principal components for dimensionality reduction
    minSize: Minimum size of a category not to be lumped into the "Other" category
    nEpochs: Number of epochs that the autoencoder performs (increase this to get better performance but also longer running time)
    deleteOld: Delete columns for which data reduction has been performed (True = Delete, False = Do not delete)
    verbose: Display additional output for diagnostic purposes
    """

    # Create copy of data to prevent unintentional overwrites
    data = data.copy()
    
    # Set less frequent categories to "other" to reduce running time (if <minSize> is high enough)
    for category in catColumns :
        counts = data[category].value_counts()
        data.loc[data[category].isin(counts[counts < minSize].index), category] = "Other"

    # One-hot encoding
    OHenc = OneHotEncoder() 
    dataCategorical = OHenc.fit_transform(data[catColumns]).toarray()
    dataNumerical = data[numColumns].to_numpy()
    reducable_data = np.concatenate((dataCategorical, dataNumerical), axis=1)

    # reduce dimensionality of the data using reduction_method
    reduced_data, mse = reduce_with(
                            reduction_method=reduction_method,
                            df=reducable_data,
                            nFeatures=nFeatures,
                            nEpochs=nEpochs,
                            verbose=verbose
                        )
    
    # remove original columns if deleteOld == True
    if deleteOld:
        data.drop(columns=catColumns + numColumns, inplace=True)
    
    # reset index to allow concatenation
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data, reduced_data], axis=1)


    # Return result
    return data, mse

# Function that performs dimensionality reduction based on a reduction method that is takes as input
def reduce_with(reduction_method : str, df : pd.core.frame.DataFrame, nFeatures : int, nEpochs : int, verbose : bool):
    """
    Use reduction_method to reduce the dimensionality of df from len(df.columns) to nFeatures.

    Attributes:

    reduction_method: method used to reduce the dimensionality of df (auto_encoder, PCA, or SPCA)
    df: data of which the dimensionality should be reduced
    nFeatures: number of features to which the data should be reduced
    """

    if reduction_method == 'auto_encoder':
        
        # Encoder
        encoder = Sequential()
        # encoder.add(Flatten(input_shape=[df.shape[1]]))
        encoder.add(Input(shape=(df.shape[1],))) # ! adjusted to new tensorflow version
        encoder.add(Dense(512,activation=LeakyReLU()))
        encoder.add(Dense(256,activation=LeakyReLU()))
        encoder.add(Dense(128,activation=LeakyReLU()))
        encoder.add(Dense(64,activation=LeakyReLU()))
        encoder.add(Dense(nFeatures,activation=LeakyReLU()))
        
        # Decoder
        decoder = Sequential()
        # decoder.add(Dense(64,input_shape=[nFeatures],activation=LeakyReLU())) # * old
        decoder.add(Input(shape=(nFeatures,)))  # ! New: use Input layer instead of Dense with input_shape
        decoder.add(Dense(64, activation=LeakyReLU())) # ! new
        decoder.add(Dense(128,activation=LeakyReLU()))
        decoder.add(Dense(256,activation=LeakyReLU()))
        decoder.add(Dense(512,activation=LeakyReLU()))
        decoder.add(Dense(df.shape[1], activation=LeakyReLU()))
        decoder.add(Reshape([df.shape[1]]))
        
        # Autoencoder
        autoencoder = Sequential([encoder,decoder])
        autoencoder.compile(loss="mse")

        # early stopping callback
        callback = EarlyStopping(monitor='loss', patience=75, min_delta=0.0001)
        
        # train the autoencoder
        fit = autoencoder.fit(df, df,
                            epochs=nEpochs, callbacks=[callback], verbose=verbose)
        # minimum mse
        mse = np.min(fit.history['loss'])

        # encode the data with the trained encoder
        encoded_nFeatures = encoder.predict(df)
        
        # Putting everything ino a new dataframe & then adding it back to the original data
        df_reduced = pd.DataFrame(encoded_nFeatures, columns = ["cat"+str(i) for i in range(1,nFeatures+1)])

    elif reduction_method == 'PCA':
        # ! standardize data before applying PCA -> put this in the general function? (reduce_dimensionality())
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(df)
        
        # apply PCA
        pca = PCA(n_components=nFeatures)
        principalComponents = pca.fit_transform(scaledData)
        
        # reconstruct the data from principal components
        reconstructedData = pca.inverse_transform(principalComponents)
        
        # calculate reconstruction error
        mse = mean_squared_error(scaledData, reconstructedData)
        
        # create DataFrame for the principal components
        df_reduced = pd.DataFrame(principalComponents, columns=[f'PC{i+1}' for i in range(nFeatures)])

    elif reduction_method == 'SPCA':
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(df)
        
        # apply PCA
        sparse_pca = SparsePCA(n_components=nFeatures)
        principalComponents = sparse_pca.fit_transform(scaledData)
        
        # reconstruct the data from principal components & calculate reconstruction error
        reconstructedData = sparse_pca.inverse_transform(principalComponents)
        mse = mean_squared_error(scaledData, reconstructedData)
        
        # create DataFrame for the principal components
        df_reduced = pd.DataFrame(principalComponents, columns=[f'PC{i+1}' for i in range(nFeatures)])

    else:
        raise KeyError(f"Method {reduction_method} is not a valid reduction method.")
    
    
    return df_reduced,mse