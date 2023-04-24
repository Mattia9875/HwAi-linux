import os 
from tqdm import tqdm
from PIL import Image,ImageOps
from pathlib import Path
import imghdr,os
from wand.image import Image as Im2

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