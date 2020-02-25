from tkinter import *
import time
import threading
import logging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
from datetime import datetime, timedelta
import call_to_twint as twintCall
import STAD_utils as util
import unlabeledPreProcessing as preproc
import pandas as pd
import subprocess


def get_gravity(vet):
    class0 = 0.0
    class1 = 0.0
    class2 = 0.0
    for w in vet:
        class0 = class0 + w[0]
        class1 = class1 + w[1] 
        class2 = class2 + w[2]
    return round((((class1 + (2 * class2)) / (class1 + class2 + class0)) * 100), 2)



def add_newElemet(new_elem, new_vet):
    i = 0
    
    while i < len(y_axes) - 1:
        y_axes[i] = y_axes[i+1]
        vett_count[i] = vett_count[i+1]
        i = i+1
        
    y_axes[35] = new_elem
    vett_count[35] = new_vet



def startFunction(btn_stop, string):
	btn_stop['state'] = 'normal'
	
	plotter.plot(x_axes,y_axes)
	canvas.draw()
    
	# current timestamp and 6 hours ago timestamp
	dateTimeObj = datetime.now()
	current_timestamp = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S")
	time_1h_earlier = dateTimeObj - timedelta(minutes=1)
	last_timestamp = time_1h_earlier.strftime("%Y-%m-%d %H:%M:%S")


	# message error
	allert = Label(text="POTENTIAL ALLERT")

	# Download tweet
	twintCall.scraping_by_time(current_timestamp, last_timestamp, x_cordinate.get(), y_cordinate.get(), km_radius.get()) 


	# preprocessing: lingua e url
	vet = preproc.preProcessing()

	# classicazione
	clf = util.load_classifier()
	prediction = util.get_prediction(vet, clf)    
    
    
	count = [0,0,0]
	for i in prediction:
		count[i] = count[i] + 1
		
	print(count)
		
	figure.clf()
	add_newElemet(count[1] + count[2], count)
	figure.add_subplot(111).plot(x_axes,y_axes)
	canvas.draw()

	# faccio il grafico con i valori ottenuti
    
	#while btn_stop['state'] == 'normal':
	while True:
		time.sleep(60) # 60 * 1 -> 10 min
		last_timestamp = current_timestamp
		dateTimeObj = datetime.now()
		current_timestamp = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S")


		command = 'sudo rm tweets.csv'
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

		# Download tweet
		twintCall.scraping_by_time(current_timestamp, last_timestamp, x_cordinate.get(), y_cordinate.get(), km_radius.get()) 


		# preprocessing: lingua e url
		vet = preproc.preProcessing()

		prediction = util.get_prediction(vet, clf)
		count = [0,0,0]
		for i in prediction:
			count[i] = count[i] + 1
        
        
		p = get_gravity(vett_count)
		if p > 5.0:
			allert.grid( row=3, column=3)
		else:
			allert.grid_remove()


		percentage.set(str(p) + " %")
		print(count)

		figure.clf()
		add_newElemet(count[1] + count[2], count)
		figure.add_subplot(111).plot(x_axes,y_axes)
		canvas.draw()
    


def stopFunction(btn_stop):
	print(x_cordinate.get())
	btn_stop.config(state='disable')
	

def main_screen():
	global screen 
	screen = Tk()
	screen.geometry("700x700")
	screen.title("STAD")

	global x_cordinate, y_cordinate, km_radius, percentage

	x_cordinate = StringVar()
	y_cordinate =StringVar()
	km_radius = StringVar()
	percentage = StringVar()

	global x_axes
	global y_axes
	global vett_count
	x_axes = np.arange(0,36)
	y_axes = np.zeros(36)
	vett_count = np.zeros((36,3))

	# input field and button
	Label(text="x", borderwidth=5).grid( row=1, column=1)
	x = Entry(textvariable = x_cordinate).grid( row=1, column=2)
	x_cordinate.set("43.7067293")

	Label(text="y", borderwidth=5).grid( row=2, column=1)
	y = Entry(textvariable = y_cordinate).grid( row=2, column=2) 
	y_cordinate.set("10.3253383")

	Label(text="km", borderwidth=5).grid( row=3, column=1)
	km = Entry(textvariable = km_radius).grid( row=3, column=2) 
	km_radius.set("100")

	Label(text="Gravity percentage").grid( row=1, column=3)
	perc = Entry(textvariable = percentage).grid( row=2, column=3)  



	btn_start = Button(text="START", height="2", width="30", command=lambda: startFunction(btn_stop, 'pippo'))
	btn_start.grid( row=6, column=2) 

	btn_stop = Button(text="STOP", height="2", width="30",state='disable')
	btn_stop['command']=stopFunction(btn_stop)
	btn_stop.grid( row=6, column=3)

	# component to build the plot 
	global figure
	global plotter
	global canvas
	figure = plt.Figure(figsize=(5,5), dpi=100)
	plotter = figure.add_subplot(111)
	canvas = FigureCanvasTkAgg(figure, master=screen)
	canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)

	screen.mainloop()
    

main_screen()


