'''
Created on May 06, 2024
@author: <Alexey Chernikov>

    The file executes two functions:
        a) cuts the video based on sharpness of the frame and its similarity to other frames - proc_folders function
        b) converts PRODIGY labels to Yolo8 format and prepares the data - prodigy_to_yolo function

    For Yolo8 training - copy to Yolo8 and run the file Yolo8_foodUN.py
'''

import os
from glob import glob
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import json


THRESH=40
CLASSES={'WP':0,'HW':1,'GB':2,'DW':3}
OUTPUT_DIR='/home/raistlin/Downloads/TMP/'
YOUR_DIR=''


def filter_dups(images,output_path,prefix):

    def find_close_pairs(arr, threshold=20000):
        import imagehash
        sorted_arr = arr

        close_pairs = []

        for i in range(len(sorted_arr) - 1):
            for j in range(i + 1, len(sorted_arr)):
                im1=Image.fromarray(images[i][1])
                im2 = Image.fromarray(images[j][1])
                hash1 = imagehash.phash(im1)
                hash2 = imagehash.phash(im2)
                if hash1 - hash2 <= threshold:
                    close_pairs.append((images[i][0], images[j][0], hash1 - hash2))
        return close_pairs
    inxs=pd.DataFrame([(x[0],x[2]) for x in images],columns=['frame','shrp'])
    pairs = find_close_pairs(images, threshold=16)
    pairs = pd.DataFrame(pairs, columns=['u1', 'u2', 'score'])
    keep=set(inxs['frame'].values) - set(pairs['u2'])
    inxs_keep=inxs[inxs.frame.isin(keep)].index.values
    for i in inxs_keep:
        cv2.imwrite(output_path+f'/{prefix}_{images[i][0]}_{int(images[i][2])}.jpg',images[i][1])


def cutVideo(input_path,output_path):
    vdo = cv2.VideoCapture()
    assert os.path.isfile(input_path), "Path error"
    vdo.open(input_path)
    assert vdo.isOpened()
    print('Done. Load video file ', input_path)

    tot_frames=int(vdo.get(cv2.CAP_PROP_FRAME_COUNT))
    to_take=list(range(0,tot_frames,15))

    prefix=os.path.basename(input_path)
    i=0
    res=[]
    while vdo.grab():
        _, image = vdo.retrieve()
        if i in to_take:
            sharp_coef = variance_of_laplacian(image)
            if sharp_coef>=THRESH:
                res.append((i,image,sharp_coef))
        i+=1
    filter_dups(res,output_path,prefix)


def variance_of_laplacian(img2):
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def proc_folder(fpath,output_root):
    folders=glob(fpath+'/*')
    folders=[x for x in folders if os.path.isdir(x)]
    for folder in folders:
        files=glob(folder+'/*.mp4')
        output_path=OUTPUT_DIR+output_root
        output_path_fold = os.path.join(output_path, os.path.basename(folder))
        os.makedirs(output_path_fold, exist_ok=True)
        for f in files:
            cutVideo(f,output_path_fold)


def prepare_data_vinc(item_class,file,im_folder,photos_dir,save_dir,make_relative=True):
    photos=glob.glob(photos_dir + '/**/*.jpg', recursive=True)
    images=glob.glob(im_folder + '/**/*.jpg', recursive=True)
    comb_pics=photos+images
    im_df=pd.DataFrame([(os.path.basename(x),x) for x in comb_pics],columns=['name','path'])

    labels_present=glob.glob(save_dir + '/**/*.txt', recursive=True)
    lbl_df=pd.DataFrame([(os.path.basename(x),x) for x in labels_present],columns=['name','path']).set_index('name')

    data = []
    with open(file) as f:
        for i,line in enumerate(f):
            line=json.loads(line)
            # data.append(line)
            if 'spans' not in line or isinstance(line['spans'], float):
                continue
            imname=os.path.basename(line['image'])
            impath=im_df[im_df['name'] == imname]
            if len(impath)==0:
                print('Not found:',imname)
                continue
            impath=impath.iloc[0,1]
            im_size = Image.open(impath).size
            for s in line['spans']:
                if make_relative:
                    x, y, w, h = round((s['x']+s['width']/2) / im_size[0], 5), round((s['y']+s['height']/2) / im_size[1], 5), \
                                 round(s['width'] / im_size[0], 5), round(s['height'] / im_size[1], 5)
                else:
                    x,y,w,h=int(s['x']),int(s['y']),int(s['width']),int(s['height'])
                data.append((imname,s['label'],x,y,w,h))
    df=pd.DataFrame(data,columns=['image','entity','x','y','w','h']) #.apply(lambda x: json.loads(x))

    # Remove incorrect boxes with big errors
    ind_big = np.where(df.iloc[:, 2:].values < -0.01)[0]
    df=df[~df.index.isin(ind_big)].reset_index(drop=True)

    ind_small_neg=np.where((df.iloc[:, 2:].values < 0) & (df.iloc[:, 2:].values > -0.01))[0]
    sdf=df[((df.iloc[:, 2:].values < 0) & (df.iloc[:, 2:].values > -0.01))].iloc[:, 2:].clip(lower=0)
    df.iloc[ind_small_neg, 2:]=sdf

    grs=df.groupby(by=['image'])
    for g in grs:
        image=g[0][0]
        sdf=g[1]
        image=image.replace('.jpg','.txt')
        fpath=save_dir+'/labels/'+image
        # if lbl_df[lbl_df.index==image].empty:
        with open(fpath,'a') as f:
            sub_sdf=pd.concat([pd.Series([CLASSES[item_class]]*len(sdf)),sdf.reset_index(drop=True).iloc[:,2:]],axis=1)
            f.writelines(sub_sdf.to_string(index=None,header=None))


def proc_folders():
    proc_folder('/ML/Datasets/Projects/FoodUN/Data/Videos/Attempt 5/','AT5')


def prodigy_to_yolo():
    save_dir=f'/{YOUR_DIR}/FoodUN/Data/Yolo/v1/'
    prepare_data_vinc('DW', f'/{YOUR_DIR}/FoodUN/Data/Labels/Vinc/at5_DW_annotations.jsonl',
                         f'/{YOUR_DIR}/FoodUN/Data/Images/Attempt5',
                         photos_dir=f'/{YOUR_DIR}/FoodUN/Data/Videos/Attempt 5', save_dir=save_dir)
