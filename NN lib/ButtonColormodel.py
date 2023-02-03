# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:23:30 2022

@author: rfigg
"""
from tkinter.ttk import *
from tkinter import *

import random
from neural_network import Neural_Network
from dense import Dense
import activations as act
from losses import MSE
import numpy as np
#


network = [
    Dense(3,15),
    act.Sigmoid(),
    Dense(15,15),
    act.Sigmoid(),
    Dense(15,1),
    act.Sigmoid()
    ]
#
model = Neural_Network(network, MSE)



#RGB code:
def rgbtohex(mytuple):
    r,g,b = mytuple
    return f'#{r:02x}{g:02x}{b:02x}'

def gennewcolor():
    return tuple(random.choices(range(256), k=3))




#Window part:
class MyApp:
    def __init__(self, root):
        
        root.title("Neural Network GUI")
        root.geometry('450x250')
        root.resizable(False,False)

        
        pane1 = Frame(root, height= 225, width=350)
        pane1.pack()
        pane1.grid_propagate(0)
        pane1.pack_propagate(0)
        
        pane2 = Frame(pane1, height= 125, width=350)
        pane2.pack()
        pane2.pack_propagate(0)
        
        pane3 = Frame(pane1, height= 100, width=350)
        pane3.pack()
        pane3.pack_propagate(0)
        
        pane4 = Frame(pane3, height= 125, width=175)
        pane4.pack(side='left')
        pane4.pack_propagate(0)
        
        pane5 = Frame(pane3, height= 125, width=175)
        pane5.pack(side='left')
        pane5.pack_propagate(0)
        
        label1 = Label(pane2, text="Text", bg=rgbtohex(currentcolor),font=("Arial", 25))
        label1.pack(expand=True, fill='both')
   
    
        def clickedlinks():
            global currentcolor
            model.fit(np.array([currentcolor]), np.array([[1]]),3,0.01)
            
            #initiating new collor and predicting it
            currentcolor = gennewcolor()
            print( model.predict(np.array([currentcolor])))
            b = act.BinaryStep(0.5)
            label1.configure(bg=rgbtohex(currentcolor), fg = "black" if b.activation(model.predict(np.array([currentcolor]))) == np.array([[1]]) else "white")
            button1.configure(bg=rgbtohex(currentcolor))
            button2.configure(bg=rgbtohex(currentcolor))
        
        def clickedrechts():
            global currentcolor
            model.fit(np.array([currentcolor]), np.array([[0]]),3,0.01)
            #initiating new collor and predicting it
            currentcolor = gennewcolor()
            print( model.predict(np.array([currentcolor])))
            b = act.BinaryStep(0.5)
            label1.configure(bg=rgbtohex(currentcolor), fg = "black" if b.activation(model.predict(np.array([currentcolor]))) == np.array([[1]]) else "white")
            button1.configure(bg=rgbtohex(currentcolor))
            button2.configure(bg=rgbtohex(currentcolor))    
    
    
    
    
        button1 = Button(pane4, text = "Text", command=clickedlinks, fg="black",bg=rgbtohex(currentcolor),font=("Arial", 25))
        button1.pack(side= LEFT, expand=True, fill='both', padx=(0,10),pady=(10,0))
        button2 = Button(pane5, text = "Text", command=clickedrechts, fg="white",bg=rgbtohex(currentcolor),font=("Arial", 25))
        button2.pack(side= LEFT, expand=True, fill='both', padx=(10,0),pady=(10,0))

currentcolor = gennewcolor()       
root = Tk()
MyApp(root)
root.mainloop()
#

