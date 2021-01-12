#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:18:37 2021

@author: cheikhtoure
"""
from flask import Flask, jsonify, request
import torch
import numpy as np

from model.common import Ae


    
    

path='model_state_dict.pt'

modelAE=Ae().double()
modelAE.load_state_dict(torch.load(path))
modelAE.eval()    
    



app = Flask(__name__)

@app.route("/predict", methods=['POST'])

def predict():
    
    if request.method=='POST':
        file=request.files['file']
        #value=file.read()
        value=np.loadtxt(file,delimiter=',')
        #value=normalize(value)
        with torch.no_grad():
            pred=modelAE(torch.tensor(value.reshape(1,1,288)))
        
        
        mae=np.mean(np.abs(value-pred.numpy().flatten()))
        if mae>0:
            return jsonify('anomalie')
        else:
            jsonify('nonanomlie')

    

if __name__=='__main__':
    
    app.run()    
    

       

