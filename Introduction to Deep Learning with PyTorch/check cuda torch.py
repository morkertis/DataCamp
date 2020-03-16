# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 08:12:01 2019

@author: mor
"""

import torch
print(torch.__version__)

#%%
if torch.cuda.device_count():
    print(torch.cuda.get_device_name(0))
else:
    print('Torch not compiled with CUDA enabled')

#%%



