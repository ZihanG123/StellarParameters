import tkinter as tk
import predict2
from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showerror
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import numpy as np
import seaborn as sns
from fast_histogram import histogram2d
from matplotlib.ticker import PercentFormatter

class hyperPara:
    star_id = 100
    teff = 5777.
    teff_sig = 20
    numax = 3140.
    numax_sig = 0.5
    dnu = 150.
    dnu_sig = 0.5
    fe_h = 0.0
    fe_h_sig = 0.01
    luminosity = 1.
    luminosity_sig = 0.1
    output_fig = True
    num_samples = 50000

# root window
root = tk.Tk()
root.title('Stellar Parameter Predictor')
root.geometry('600x300')
# root.resizable(False, False)

stater = IntVar()
stater.set(1)

def prediction(hyperPara):
    """ function to predict stellar parameters
    """

    return predict2.infer(hyperPara)


frame = ttk.Frame(root)
options = {'padx': 5, 'pady': 5}

rdms = Radiobutton(frame,variable=stater, value=1)
rdms.grid(column=0, row=3, **options)
rdms_label = ttk.Label(frame, text='Main Sequence')
rdms_label.grid(column=1, row=3, **options)

rdrgb = Radiobutton(frame,variable=stater, value=2)
rdrgb.grid(column=3,row=3, **options)
rdrgb_label = ttk.Label(frame, text='Red Giant')
rdrgb_label.grid(column=4, row=3, **options)


# if num == 1:
#     starType = '/best_model_ms_new.torchmodel'
# else:
#     starType = '/best_model_rgb_new.torchmodel'

value_label = ttk.Label(frame, text='Value:')
value_label.grid(column=0, row=1, **options)
error_label = ttk.Label(frame, text='Error:')
error_label.grid(column=0, row=2, **options)

# temperature label
teff_label = ttk.Label(frame, text='Temperature')
teff_label.grid(column=1, row=0, **options)

# temperature entry
teff = tk.StringVar(value=5777)
teff_entry = ttk.Entry(frame, textvariable=teff, width=10, justify='center')
teff_entry.grid(column=1, row=1, **options)
teff_entry.focus()
teff_sigma = tk.StringVar(value=20)
teff_sigma_entry = ttk.Entry(frame, textvariable=teff_sigma, width=10, justify='center')
teff_sigma_entry.grid(column=1, row=2, **options)


# Fe_H label
teff_label = ttk.Label(frame, text='[Fe/H]')
teff_label.grid(column=2, row=0, **options)

# Fe_H entry
fe_h = tk.StringVar(value=0.0)
fe_h_entry = ttk.Entry(frame, textvariable=fe_h, width=10, justify='center')
fe_h_entry.grid(column=2, row=1, **options)
fe_h_sigma = tk.StringVar(value=0.01)
fe_h_sigma_entry = ttk.Entry(frame, textvariable=fe_h_sigma, width=10, justify='center')
fe_h_sigma_entry.grid(column=2, row=2, **options)

# numax label
numax_label = ttk.Label(frame, text='numax')
numax_label.grid(column=3, row=0, **options)

# numax entry
numax = tk.StringVar(value=3000)
numax_entry = ttk.Entry(frame, textvariable=numax, width=10, justify='center')
numax_entry.grid(column=3, row=1, **options)
numax_sigma = tk.StringVar(value=1)
numax_sigma_entry = ttk.Entry(frame, textvariable=numax_sigma, width=10, justify='center')
numax_sigma_entry.grid(column=3, row=2, **options)

# dnu label
dnu_label = ttk.Label(frame, text='Delta_nu')
dnu_label.grid(column=4, row=0, **options)

# dnu entry
dnu = tk.StringVar(value=130)
dnu_entry = ttk.Entry(frame, textvariable=dnu, width=10, justify='center')
dnu_entry.grid(column=4, row=1, **options)
dnu_sigma = tk.StringVar(value=0.5)
dnu_sigma_entry = ttk.Entry(frame, textvariable=dnu_sigma, width=10, justify='center')
dnu_sigma_entry.grid(column=4, row=2, **options)

# luminosity label
luminosity_label = ttk.Label(frame, text='Luminosity')
luminosity_label.grid(column=5, row=0, **options)

# luminosity entry
luminosity = tk.StringVar(value=1)
luminosity_entry = ttk.Entry(frame, textvariable=luminosity, width=10, justify='center')
luminosity_entry.grid(column=5, row=1, **options)
luminosity_sigma = tk.StringVar(value=0.1)
luminosity_sigma_entry = ttk.Entry(frame, textvariable=luminosity_sigma, width=10, justify='center')
luminosity_sigma_entry.grid(column=5, row=2, **options)

def submit_button_clicked():
    """  Handle submit button click event
    """
    try:
        hyperPara.teff = float(teff.get())
        hyperPara.teff_sig = float(teff_sigma.get())
        hyperPara.fe_h = float(fe_h.get())
        hyperPara.teff_sig = float(fe_h_sigma.get())
        hyperPara.numax = float(numax.get())
        hyperPara.numax_sig = float(numax_sigma.get())
        hyperPara.dnu = float(dnu.get())
        hyperPara.dnu_sig = float(dnu_sigma.get())
        hyperPara.luminosity = float(luminosity.get())
        hyperPara.luminosity_sig = float(luminosity_sigma.get())
        num = stater.get()
        n = {1: "ms", 2: 'rgb'}
        hyperPara.star_type = n[stater.get()]
        if num == 1:
            starType = '/best_model_ms_new.torchmodel'
        else:
            starType = '/best_model_rgb_new.torchmodel'
        hyperPara.star_type = starType
        median, up_err, low_err = prediction(hyperPara)
        result_label['text'] = "The stellar parameters are:\n"
        output_params = ['Mass', 'Age', 'X', 'Z', 'MLT', 'Radius']
        for i, params in enumerate(output_params):
            result_label['text']+=params + ': ' + '%.3f +%.3f -%.3f\n' % (median[i], up_err[i], low_err[i])
        print(starType)

    except ValueError as error:
        showerror(title='Error', message=error)


def clear_button_clicked():
    """  Handle clear button click event
    """
    result_label['text'] = 'The stellar parameters are:'


# submit button
submit_button = ttk.Button(frame, text='Submit')
submit_button.grid(column=0, row=4, sticky='W', **options)
submit_button.configure(command=submit_button_clicked)

# result label
result_label = ttk.Label(frame, text='The stellar parameters are:')
result_label.grid(row=5, columnspan=4, sticky='W', **options)

# submit button
clear_button = ttk.Button(frame, text='Clear')
clear_button.grid(column=1, row=4, sticky='W', **options)
clear_button.configure(command=clear_button_clicked)


# add padding to the frame and show it
frame.grid(padx=10, pady=10)

root.mainloop()