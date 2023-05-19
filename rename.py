import sys,os


dir = "/home/mattiamorabito/Documents/HwAi-linux/dataset_vanilla/test"
count = 1
for dir_name in os.listdir(dir):
    newdir = os.path.join(dir,dir_name)
    print(newdir)
    file_base = dir_name
    for file_name in os.listdir(newdir):
        old_name = os.path.join(newdir,file_name)
        new_name = os.path.join(newdir,file_base+"_"+str(count)+".jpg")
        os.rename(old_name,new_name)
        count +=1
    count = 1