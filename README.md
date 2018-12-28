
# Building a Deep Auto-Encoder - Lab

## Introduction

Deep auto encoders are characterized by having more than one layers in their encoder and decoder components. In this lab, we will mainly repeat the last experiment, but using a deep architecture instead of a simple feed forward styles networks for encoding and decoding the last lab. 

## Objectives

- Build a deep autoencoder in Keras
- Create the encoder and decoder functions as multiple fully connected layers. 
- Train an autoencoder with selected loss function and optimizer.

## Deep Auto-Encoders

The extension of the simple Autoencoder is the __Deep Autoencoder__. The first layer of the Deep Autoencoder is used for first-order features in the raw input. The second layer is used for second-order features corresponding to patterns in the appearance of first-order features. This is also knows as FoF in deep learning i.e. Features of Features. Deeper layers of the Deep Autoencoder tend to learn even higher-order features.

A deep autoencoder is composed of two, symmetrical deep-belief networks

> __DBN (Deep belief Network)__ is a class of deep neural network which comprises of multiple layer of graphical model having both directed and undirected edges. It is composed of multiple layers of hidden units, where each layers are connected with each others but units are not. [Visist here](https://codeburst.io/deep-learning-deep-belief-network-fundamentals-d0dcfd80d7d4) for details. 



First four or five shallow layers representing the encoding half of the net.
The second set of four or five layers that make up the decoding half.

<img src="deep.png" width=600>

In our previous lab, we used single fully-connected layers for both the encoding and decoding models while building a simple AE. With deep AE, we can stack multiple fully-connected layers to make each of the encoder and decoder functions __deep__, turning our simple model into a deep architecture.

## Repeat the previous experiment with suggested modifications.

## Import the code for reading + preprocessing  fashion-MNIST dataset 


```python
# Install tensorflow and keras if you haven't done so already
# !pip install tensorflow
# !pip install keras

# Import necessary libraries
import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers

from IPython.display import Image
import matplotlib.pyplot as plt

# Load the training and test data sets (ignoring labels)
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize the train and test data to a range between 0 and 1.
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# Reshape the training data to create 1D vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

(x_train.shape, x_test.shape)
```

    /anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.





    ((60000, 784), (10000, 784))



## Build the Deep Autoencoder

So this time, we are building a deep autoencoder. The code for this wo't be much different to what we saw earlier. Here we are adding a few extra encoding and decoding layers as listed below:

- Use 3 fully-connected layers for the encoding model that inputs original 784 dimensions and decrease the dimensionality from 128 to 64 to 32. 

- Add 3 fully-connected decoder layers that reconstruct the image back to 784 dimensions.
- Except for the last layer, use ReLU activation functions in all other layers
- Show the model summary 




```python
# Build a deep AE
input_dim = x_train.shape[1] # input dimension = 784
encoding_dim = 32

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                2112      
    _________________________________________________________________
    dense_5 (Dense)              (None, 128)               8320      
    _________________________________________________________________
    dense_6 (Dense)              (None, 784)               101136    
    =================================================================
    Total params: 222,384
    Trainable params: 222,384
    Non-trainable params: 0
    _________________________________________________________________


## Extract the Encoder 

As seen previously, we will now extract the encoder model from the above. Remember the encoder model now consists of the first 3 layers in the autoencoder.


```python
# Extract the Encoder model 

input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 784)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                2080      
    =================================================================
    Total params: 110,816
    Trainable params: 110,816
    Non-trainable params: 0
    _________________________________________________________________


## Train the Model

We can now compile the model with Adam optimizer and binary cross entropy loss. Use 20 epochs and a batch size of 256. 


```python
# Compile and train and the model 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                validation_data=(x_test, x_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 4s 61us/step - loss: 0.3789 - val_loss: 0.3175
    Epoch 2/20
    60000/60000 [==============================] - 3s 49us/step - loss: 0.3089 - val_loss: 0.3061
    Epoch 3/20
    60000/60000 [==============================] - 3s 45us/step - loss: 0.3004 - val_loss: 0.3011
    Epoch 4/20
    60000/60000 [==============================] - 3s 46us/step - loss: 0.2958 - val_loss: 0.2969
    Epoch 5/20
    60000/60000 [==============================] - 3s 45us/step - loss: 0.2933 - val_loss: 0.2944
    Epoch 6/20
    60000/60000 [==============================] - 3s 43us/step - loss: 0.2913 - val_loss: 0.2940
    Epoch 7/20
    60000/60000 [==============================] - 3s 47us/step - loss: 0.2898 - val_loss: 0.2912
    Epoch 8/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2884 - val_loss: 0.2900
    Epoch 9/20
    60000/60000 [==============================] - 3s 47us/step - loss: 0.2872 - val_loss: 0.2890
    Epoch 10/20
    60000/60000 [==============================] - 3s 47us/step - loss: 0.2862 - val_loss: 0.2877
    Epoch 11/20
    60000/60000 [==============================] - 3s 47us/step - loss: 0.2851 - val_loss: 0.2870
    Epoch 12/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2842 - val_loss: 0.2861
    Epoch 13/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2835 - val_loss: 0.2855
    Epoch 14/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2828 - val_loss: 0.2848
    Epoch 15/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2821 - val_loss: 0.2839
    Epoch 16/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2816 - val_loss: 0.2835
    Epoch 17/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2811 - val_loss: 0.2831
    Epoch 18/20
    60000/60000 [==============================] - 3s 50us/step - loss: 0.2806 - val_loss: 0.2827
    Epoch 19/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2801 - val_loss: 0.2824
    Epoch 20/20
    60000/60000 [==============================] - 3s 48us/step - loss: 0.2797 - val_loss: 0.2819





    <keras.callbacks.History at 0x1834e90898>



## View the Code and Reconstruction

Bring in the code from previous experiment to view the reconstruction and encoding performed by our deep AE for 10 random images. 


```python
# Inspect Results 
num_images = 10
np.random.seed(45)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```


![png](index_files/index_13_0.png)


---

Do you notice any improvement over the previous model ? Let's admit it, there is not a huge change , due to the fact that we did not train the model to the point of convergence (for saving some time). Also, we did not use any cross validation techniques. We can improve these models by performing following tasks:


## Level Up - Optional 

- Train both (simple and deep) AEs to 100 epochs and compare the results
- Apply k-fold cross validation with deep AE (highly recommended for avoiding overfitting in deep networks) and check for any improvements.
- Repeat the simple and deep AE labs with MNIST dataset (available in Keras).
- Try this experiment with a high resolution (very high dimensionality) dataset. Caution: The training time may reach upto hours for a large dataset (or even days) - Thats where GPU/cloud computing comes into play. 

## Summary 

In this lab, we created a deep Autoencoder following the similar approach and dataset from our previous lab. We developed 3 layer encoder and decoder functions in keras and trained the network for 20 epochs. Next, we shall look into an AE architecture which is highly suitable for Image data - The Convolutional Auto-Encoder. 
