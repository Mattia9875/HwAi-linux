import os 
from tqdm import tqdm
from PIL import Image,ImageOps

def import_images():
    SIZE = 512
    DIR = "/home/mattiamorabito/Documents/HwAi/testing_fullsize"
    DIR_RESIZED = "/home/mattiamorabito/Documents/HwAi/testing"
    for dir_name in tqdm(os.listdir(DIR)):
        new_dir = os.path.join(DIR,dir_name)
        new_dir_resized = os.path.join(DIR_RESIZED,dir_name)
        index = 0;
        for img_name in os.listdir(new_dir):
            img_path = os.path.join(new_dir,img_name)
            img = Image.open(img_path)
            #img = img.resize((SIZE,SIZE), Image.Resampling.LANCZOS)
            img = ImageOps.contain(img, (SIZE,SIZE))
            name = str(index)+".jpg"
            
            img.save(os.path.join(new_dir_resized,name))
            index += 1