# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:13:59 2021

@author: malrawi

"""

import json

def save_as_json(fname, colors, labels, cnf):
    color_dict={}
    color_dict['config'] = vars(cnf)
    color_dict['colors'] = colors    
    color_dict['labels'] = labels
    
    with open(fname, 'w') as fp:
        json.dump(color_dict, fp, indent=True)
       
def read_from_json(fname):
    with open(fname) as json_file:
        data_dict = json.load(json_file)
    return data_dict


def save_dict_as_json( data_dict, fname):
        
    with open(fname, 'w') as fp:
        json.dump(data_dict, fp, indent=True)