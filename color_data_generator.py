# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:10:02 2021

@author: malrawi


# Dizygotic Net

"""

import argparse
from PIL import Image, ImageChops# , ImageDraw
import random
from IPython.display import display 
from clothcoparse_dataset import  ImageDataset # main directory should be on top of FashionColor and ColorCounter
from clothing_class_names import get_59_class_names
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from datetime import datetime
import json



parser = argparse.ArgumentParser()
parser.add_argument("--num_color_permutations", type=int, default=10, help="number of colors permutations for each image")
parser.add_argument("--source_dataset_name", type=str, default="ClothCoParse", help="name of the dataset: {ClothCoParse, or Modanet}")
parser.add_argument("--generated_dataset_name", type=str, default="colors_of_fashion.json", help="name of the output dataset, as a json file")
# parser.add_argument("--max_num_colors", type=int, default=10, help="max number of colors the user wants")
# parser.add_argument('--save_images', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses a pretrained model")
# parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
# parser.add_argument("--HPC_run", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default False; set to True if running on HPC")


cnf = parser.parse_args()
cnf.num_color_permutations=2
cnf.rgb_min = (0, 0, 0) # lower bound colors used to draw the text
cnf.rgb_max = (255, 255, 255) # lower bound colors used to draw the text


path = 'C://MyPrograms//Data/ColorCountData//'

def get_random_rgb(rgb_min, rgb_max):
    R = random.randint(rgb_min[0], rgb_max[0])
    G = random.randint(rgb_min[1], rgb_max[1])
    B = random.randint(rgb_min[2], rgb_max[2])
    return R, G, B


def duplicate_mask(mask):
    sx, sy = mask.shape    
    mask1 = resize(mask, (sx//2, sy) )
    
    return mask1
     
    
def save_as_json(fname, colors_in_images, image_names, labels_in_images, cnf):
    color_dict={}
    color_dict['cnf'] = vars(cnf)
    color_dict['colors_in_images'] = colors_in_images
    color_dict['image_names'] = image_names
    color_dict['labels_in_images'] = labels_in_images
    with open(fname, 'w') as fp:
        json.dump(color_dict, fp, indent=True)
       
def read_from_json(fname):
    with open(fname) as json_file:
        data_dict = json.load(json_file)
    return data_dict

dataset = ImageDataset(root= "../data/%s" % cnf.source_dataset_name, 
                               class_names_and_colors= get_59_class_names(),                                
                                mode="train",                          
                                HPC_run=False,                                 
                            )
def generate_color_dataset(dataset, cnf, display_image = True, num_images_to_use=1e10):
    image_names = []
    for ii, item in enumerate(dataset):
        if ii>num_images_to_use: break
        image_A, masked_img, labels, image_id, masks, fname = item
        
        labels_in_images = []
        colors_in_images = []            
        for jj in range(cnf.num_color_permutations):        
            image_base = Image.new("RGBA", image_A.size, (0, 0, 0, 0))
            for kk, mask in enumerate(masks):        
                R, G, B = get_random_rgb(cnf.rgb_min, cnf.rgb_max)        
                sx, sy = mask.shape
                image = Image.new("RGBA", (sy, sx), (R, G, B, 255)) # for some reason, the new image is transposed, we have to swap x-size and y-size; crazey stuff, right?
                mask4 = Image.fromarray(np.asarray(255*np.dstack([mask]*4), dtype='uint8'))     
                image = ImageChops.multiply(image, mask4)        
                image_base = Image.alpha_composite(image_base, image)
                colors_in_images.append([R,G, B])
                labels_in_images.append(labels)
                        
            image_name = fname+str(jj)+'.png'    
            image_base.save(path+image_name, 'png')
            image_names.append(image_name)
            display(image_base)
        
    return colors_in_images, image_names, labels_in_images

random.seed(datetime.now())
colors_in_images, image_names, labels_in_images = generate_color_dataset(dataset, cnf, num_images_to_use=1)
save_as_json(path+ cnf.generated_dataset_name, colors_in_images, image_names, labels_in_images, cnf)   
data_dict = read_from_json(path+cnf.generated_dataset_name)

























    

#  image_base = Image.new("RGBA", (512, 512))
#  for i in range(cnf.max_num_colors):
#     R, G, B = get_random_rgb(rgb_min, rgb_max)    
#     image = Image.new("RGB", (sizeX, sizeY), (R, G, B))    
#     # draw = ImageDraw.Draw(image)
#     # draw.rectangle(((0, 00), (100, 100)), fill ="#ffff33")    
#     image_base.paste(image, (50,50))
    
# 

# image_base = Image.new("RGBA", (512, 512))



# Imahttps://pillow.readthedocs.io/en/3.0.x/reference/ImageDraw.html
# PIL.ImageDraw.Draw.polygon(xy, fill=None, outline=None)
# Draws a polygon.

# The polygon outline consists of straight lines between the given coordinates, plus a straight line between the last and the first coordinate.

# Parameters:	
# xy – Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
# outline – Color to use for the outline.
# fill – Color to use for the fill.
# ImageDraw.ImageDraw.polygon(xy, fill=None, outline=None)
# Draws a rectangle.

# Parameters:	
# xy – Four points to define the bounding box. Sequence of either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
# outline – Color to use for the outline.
# fill – Color to use for the fil