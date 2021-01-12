#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:31:35 2021

@author: cheikhtoure
"""

import numpy as np
import torch.nn as nn

class Ae(nn.Module):
    '''
    Convolutional AutoEncoder:
    input:
    -----
    x: (B,Cin,l) where B is the batch size, Cin (number of channels), (l length)
    
    parameters:
    -----------
    kernel_size=3
    
    return:
    ------
    y:(B,Cout,l) reconstruction of x
    
    '''

    def __init__(self):
        super(Ae,self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32, kernel_size=3),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32,out_channels=16,kernel_size=3),
            nn.ReLU(True)
                                )
        
        
        self.decoder=nn.Sequential(
            nn.ConvTranspose1d(in_channels=16,out_channels=16,kernel_size=3),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(in_channels=16,out_channels=32,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32,out_channels=1,kernel_size=1)
                                )
        
        
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    
    

def normalize(valeur):
    
    return (np.array(valeur)-42.43835333580657)/28.073640279354013    