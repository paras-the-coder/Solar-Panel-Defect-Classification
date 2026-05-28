import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications import ResNet50

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=6, regularized=True):
    """
    Builds the baseline sequential convolutional neural network.
    Optionally adds BatchNormalization to combat severe training overfitting.
    """
    model = Sequential([
        Input(shape=input_shape)
    ])
    
    # Conv Block 1
    model.add(Conv2D(32, (3, 3), activation='relu'))
    if regularized:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Conv Block 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    if regularized:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Conv Block 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    if regularized:
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def build_resnet_model(num_classes=6, dense_units=512, dropout_rate=0.2, input_shape=(224, 224, 3)):
    """
    Builds the ResNet50 transfer learning model.
    Preprocessing is handled once in the data loading pipeline and once at app inference.
    """
    # Load pre-trained ResNet50 backbone (weights frozen initially)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # Build Sequential classification head
    model = Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model
