import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.applications.resnet50 import preprocess_input

def get_image_paths_and_labels(data_dir):
    """
    Scans the data directory and extracts image file paths and their corresponding labels.
    """
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    file_paths = []
    labels = []
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    file_paths.append(os.path.join(root, file))
                    labels.append(class_to_idx[class_name])
                    
    return np.array(file_paths), np.array(labels), class_names

def load_and_preprocess_image(path, label, preprocess_fn=preprocess_input):
    """
    Reads an image from path, decodes it, resizes it to 224x224,
    and applies preprocessing.
    """
    image_raw = tf.io.read_file(path)
    # decode_image handles JPEG, PNG, GIF, BMP etc.
    image = tf.image.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image.set_shape([224, 224, 3])
    
    if preprocess_fn is not None:
        image = preprocess_fn(image)
        
    return image, label

def augment_image(image, label):
    """
    Applies random horizontal flips, brightness, and contrast adjustments for data augmentation.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def prepare_datasets(data_dir, batch_size=32, val_size=0.15, test_size=0.15, seed=42):
    """
    Performs stratified splitting into Train/Val/Test and builds tf.data pipelines.
    Returns: (train_ds, val_ds, test_ds), class_names, class_weight_dict
    """
    # 1. Retrieve all paths and labels
    paths, labels, class_names = get_image_paths_and_labels(data_dir)
    num_classes = len(class_names)
    
    if len(paths) == 0:
        raise ValueError(f"No valid images found in directory: {data_dir}")
        
    # 2. Stratified splitting to handle class imbalance and prevent data leakage
    # First split off the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    
    # Next, split train and val
    relative_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=relative_val_size, 
        stratify=train_val_labels, 
        random_state=seed
    )
    
    print(f"Dataset split summary:")
    print(f"  Training files:   {len(train_paths)}")
    print(f"  Validation files: {len(val_paths)}")
    print(f"  Testing files:    {len(test_paths)}")
    
    # 3. Compute class weights to handle imbalance during training
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(weights))
    
    # 4. Create tf.data.Dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    
    # 5. Map, shuffle, cache and prefetch pipelines
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Training dataset: load, augment, shuffle, cache, batch, prefetch
    train_ds = (train_ds
                .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
                .map(augment_image, num_parallel_calls=AUTOTUNE)
                .shuffle(buffer_size=1000)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))
                
    # Validation dataset: load, batch, prefetch
    val_ds = (val_ds
              .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
              .batch(batch_size)
              .prefetch(buffer_size=AUTOTUNE))
              
    # Test dataset: load, batch, prefetch
    test_ds = (test_ds
               .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
               .batch(batch_size)
               .prefetch(buffer_size=AUTOTUNE))
               
    return (train_ds, val_ds, test_ds), class_names, class_weight_dict
