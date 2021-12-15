from formatting import label_all_files
import numpy as np
from PIL import Image
import os.path
from os import path
import shutil



#generate the name for jpg file with correct format
def generate_frame_name(num):
    i = str(num)
    if num < 10:
        name = "000" + i
    elif num < 100:
        name = "00" + i
    elif num < 1000:
        name = "0" + i
    else:
        name = i
    return "/frame" + name + ".jpg"

#produce a single clip composing of N_FRAME frames begin at START
def produce_single_clip(path, n_frame, start, width, height):
    data = []
    for i in range(n_frame):
        pic_name = generate_frame_name(i+start)
        image = Image.open(path + pic_name).resize((width,height))
        pic = np.asarray(image)
        data.append(pic)
    data = np.array(data)
    data = np.expand_dims(data, axis = 0)
    return data

#produce a batch of clips
def produce_batch_clip(file_path, n_frame, start, width, height):
    imgs = []
    num = start
    while True:
        pic_name = generate_frame_name(num)
        num += 1
        if path.exists(file_path + pic_name):
            imgs.append(np.asarray(Image.open(file_path + pic_name).resize((width,height))))
        else:
            break
    imgs = np.array(imgs)
    print("source file has " + str(imgs.shape[0]) + "images")
    data = np.zeros((imgs.shape[0]- n_frame, n_frame, width, height, 3))
    for i in range(imgs.shape[0] - n_frame):
        data[i, :, :, :, :] = np.array([imgs[start+i+j] for j in range(n_frame)])
    return data

def produce_batch(file_path, n_frame, start, width, height, batch_size):
    imgs = []
    num = start
    while num < start + batch_size + n_frame - 1:
        pic_name = generate_frame_name(num)
        num += 1
        if path.exists(file_path + pic_name):
            imgs.append(np.asarray(Image.open(file_path + pic_name).resize((width,height))))
        else:
            break
    imgs = np.array(imgs)
    data = np.zeros((imgs.shape[0]- n_frame + 1, n_frame, width, height, 3))
    for i in range(imgs.shape[0] - n_frame + 1):
        data[i, :, :, :, :] = np.array([imgs[i+j] for j in range(n_frame)])
    return data

def data_aug(source_path, up_list, down_list):
    if os.path.exists(up_list):
        print('Upsampling Data...')
        up_file = open(up_list, 'r')
        up_contents = up_file.readlines()
        for name in up_contents:
            upsample(source_path, name)

    if os.path.exists(down_list):
        print('Downsampling Data...')
        down_file = open(down_list, 'r')
        down_contents = down_file.readlines()
        for name in down_contents:
            downsample(source_path, name)

def upsample(source_path, name):
    folder_name = source_path + '/' + name
    up_folder_name = folder_name + '_upsample'
    label_name = source_path + '/' + name + '.txt'
    up_label_name = source_path + '/' + name + '_upsample.txt'

    if not (os.path.exists(folder_name) and os.path.exists(label_name)):
        print("Required files not exist; Upsampling of " + name + " failed.")
        return
    
    if os.path.exists(up_folder_name):
        shutil.rmtree(up_folder_name)
    os.mkdir(up_folder_name)

    file_names = os.listdir(folder_name)
    for i in range(len(file_names)):
        new_pic1 = up_folder_name + generate_frame_name(i*2)
        new_pic2 = up_folder_name + generate_frame_name(i*2 + 1)
        shutil.copy(os.path.join(folder_name, file_names[i]), new_pic1)
        shutil.copy(os.path.join(folder_name, file_names[i]), new_pic2)

    label_file = open(label_name, 'r')
    new_label_file = open(up_label_name, 'w+')
    labels = label_file.readlines()
    for label in labels:
        new_label_file.write(label)
        new_label_file.write(label)
    label_file.close()
    new_label_file.close()
    return

def downsample(source_path, name):
    folder_name = source_path + '/' + name
    down_folder_name = folder_name + '_downsample'
    label_name = source_path + '/' + name + '.txt'
    down_label_name = source_path + '/' + name + '_downsample.txt'

    if not (os.path.exists(folder_name) and os.path.exists(label_name)):
        print("Required files not exist; Downsampling of " + name + " failed.")
        return
    
    if os.path.exists(down_folder_name):
        shutil.rmtree(down_folder_name)
    os.mkdir(down_folder_name)

    file_names = os.listdir(folder_name)
    for i in range(len(file_names)):
        new_pic1 = down_folder_name + generate_frame_name(i*2)
        new_pic2 = down_folder_name + generate_frame_name(i*2 + 1)
        shutil.copy(os.path.join(folder_name, file_names[i]), new_pic1)
        shutil.copy(os.path.join(folder_name, file_names[i]), new_pic2)

    label_file = open(label_name, 'r')
    new_label_file = open(down_label_name, 'w+')
    labels = label_file.readlines()
    for label in labels:
        new_label_file.write(label)
        new_label_file.write(label)
    label_file.close()
    new_label_file.close()
    return