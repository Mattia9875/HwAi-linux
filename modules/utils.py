import os 
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageOps
from pathlib import Path
import imghdr,os
from wand.image import Image as Im2
import tensorflow as tf
from tabulate import tabulate

def import_images():
    SIZE = 512
    DIR = "/home/mattiamorabito/Downloads/rotten"
    DIR_RESIZED = "/home/mattiamorabito/Downloads/resized"
    for dir_name in tqdm(os.listdir(DIR)):
        new_dir = os.path.join(DIR,dir_name)
        new_dir_resized = os.path.join(DIR_RESIZED,dir_name)
        index = 0;
        for img_name in os.listdir(new_dir):
            try:
                img_path = os.path.join(new_dir,img_name)
                img = Image.open(img_path)
                #img = img.resize((SIZE,SIZE), Image.Resampling.LANCZOS)
                img = ImageOps.contain(img, (SIZE,SIZE))
                name = "r"+str(index)+".jpg"
                
                img.save(os.path.join(new_dir_resized,name))
                index += 1
            except:
                print(img_name)
                os.remove(img_path)

def check_images():
    DIR = "/home/mattiamorabito/Documents/HwAi-linux/newds/"
    image_extensions = [".png", ".jpg",".jpeg"] # add there all your images file extensions
    img_type_accepted_by_tf = ["png", "jpg","jpeg"]
    for filepath in Path(DIR).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            try:
                img_type = imghdr.what(filepath)
                im = Image.open(filepath)
            except Exception as e:
                print(e, filepath)
           
            if img_type is None:
                print(f"{filepath} is not an image")
                #os.remove(filepath)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                if img_type == "webp":
                    #print(type(filepath),filepath)
                    #im = Image.open(str(filepath)).convert("RGB")
                    #im.save(str(filepath), "jpeg")
                    pass

def checkDuplicates():
    DIR = "/home/mattiamorabito/Documents/HwAi-linux/newds/"
    for dir_name in tqdm(os.listdir(DIR)):
        signatureList = []
        new_dir = os.path.join(DIR,dir_name)
        for img_name in os.listdir(new_dir):
            img_path = os.path.join(new_dir,img_name)
            img = Im2(filename=img_path)
            signatureList.append(img.signature)
        seen = set()
        dupes = [x for x in signatureList if x in seen or seen.add(x)]
        print(dupes)

def keras_model_memory_usage(model, batch_size,log=True):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes..
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = []
    head = []
    table_vert = []
    table_vert.append(["N#","Layer","Datatype","Feature Map Memory (KB)","Parameters","Output Shape"])
    internal_model_mem_count = 0
    count = 1
    if isinstance(model, list):
        trainable_count = 0
        non_trainable_count = 0
        for layer in model:
            l_list = []
            single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            wh = layer.weights
            num_param = 0
            for elem in wh:
                num_param += tf.keras.backend.count_params(elem)
            l_list.append(count)
            l_list.append(str(layer.name))
            l_list.append(str(layer.dtype))
            l_list.append(single_layer_mem/1000) #in Kb
            l_list.append(num_param)
            l_list.append(out_shape)
            count += 1
            table_vert.append(l_list)
            head.append(str(layer.name+ " (KB)"))
            shapes_mem_count.append(single_layer_mem/1000) #in Kb
            
            trainable_count += sum(
                            [tf.keras.backend.count_params(p) for p in layer.trainable_weights]
                            )
            non_trainable_count += sum(
                            [tf.keras.backend.count_params(p) for p in layer.non_trainable_weights]
                            )
    else:
        for layer in model.layers:
            l_list = []
            single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            wh = layer.weights
            num_param = 0
            for elem in wh:
                num_param += tf.keras.backend.count_params(elem)
            l_list.append(count)
            l_list.append(str(layer.name))
            l_list.append(str(layer.dtype))
            l_list.append(single_layer_mem/1000) #in Kb
            l_list.append(num_param)
            l_list.append(out_shape)
            count += 1
            table_vert.append(l_list)
            head.append(str(layer.name + " (KB)"))
            shapes_mem_count.append(single_layer_mem/1000) #in Kb

    
        trainable_count = sum(
            [tf.keras.backend.count_params(p) for p in model.trainable_weights]
        )
        non_trainable_count = sum(
            [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
        )
    
    max_act=np.sum(np.sort(np.array(shapes_mem_count))[-2:])
    table_vert.append(["All",model.name,"","{:.3f}".format(np.sum(np.array(shapes_mem_count))),str((trainable_count+non_trainable_count)/1000)+" K",""])
    table = [ ["Internal Memory (KB)","Trainable paramaters (kN)","Non-trainable parameters (kN)","Max Consecutive Activation (KB)"],
                [internal_model_mem_count/1000,trainable_count/1000,non_trainable_count/1000,max_act]]
    if log:
        
        print(tabulate(table_vert,headers='firstrow', tablefmt='fancy_grid'))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    return_values = [trainable_count+non_trainable_count,max_act,count-1]
    return return_values, table_vert
    #return head,shapes_mem_count, internal_model_mem_count,trainable_count,non_trainable_count



def prepare(ds, shuffle=False, augment=False, batch_size=32,img_size=96):
  
    AUTOTUNE = tf.data.AUTOTUNE
    aug = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=(-0.05, 0.05),fill_mode="constant"),
        tf.keras.layers.RandomZoom( height_factor=(-0.05, 0.05), fill_mode="constant"),
        tf.keras.layers.RandomBrightness(0.05, value_range=(0, 1))])

    normalization = tf.keras.models.Sequential([ 
        tf.keras.layers.Resizing(img_size, img_size),
        tf.keras.layers.Rescaling(1./255)])
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (normalization(x), y), 
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    #ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

def predict_from_dataset(ds,model):
    predictions = np.array([])
    labels =  np.array([])
    for x, y in tqdm(ds):
        # selecet the class for which i have the max confidence
        predictions = np.concatenate([predictions, np.argmax(model.predict(x,verbose=0), axis = -1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    return labels,predictions
