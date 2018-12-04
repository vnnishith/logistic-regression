#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:22:56 2018

@author: nishithvn
"""

import Tkinter
from tkFileDialog import askdirectory, askopenfilename
import ntpath, time, tkFont
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from PIL import Image

class GUI(object):
    def __init__(self):
        """
        Class which render the GUI for the Logistics Regression Classifier
        Allows the user to import datasets, choose where the output directory where the results
        have to be saved, classify them with the Logistics Regression Classifier and see the results
        """
        self.init_method()

    def init_method(self):
        """
        the method which renders the initial frame for the APP
        """
        self.main_window = Tkinter.Tk()
        self.main_window.wm_title("NUIG, Machine Learning (CT475)")
        self.main_window.wm_attributes('-alpha', 0.43)
        self.main_window.resizable(width=False, height=False)
        self.initial_width = 700
        self.initial_height = 550
        self.arial_button_font = tkFont.Font(family='Arial', size=12, weight=tkFont.BOLD)
        self.main_window.geometry('{}x{}'.format(self.initial_width, self.initial_height))
        image_sigmoid = Tkinter.PhotoImage(file="sigmoid.gif")
        Tkinter.Label(self.main_window, image=image_sigmoid).pack()
        self.import_button = Tkinter.Button(self.main_window,  height=2, width=20,borderwidth=2,
                                           font=self.arial_button_font, text="Import",command=self.data_import)
        self.import_button.place(relx=.7, rely=.7, anchor="c")
        label_text = Tkinter.StringVar()
        label = Tkinter.Label(self.main_window, textvariable=label_text, font=("Arial", 22),bg="white", relief="groove")
        label_text.set("Logistic Regression")
        label.place(relx=.2, rely=.2, anchor="c")
        self.classification_details_container = Tkinter.Frame()
        self.classification_details_label = Tkinter.Label()
        self.main_window.mainloop()

    def data_import(self):
        """
        Action for when import button is pressed
        Allows import of only csv files
        """
        opts = {'filetypes': [('CSV files', '.csv')]}
        self.filepath = askopenfilename(**opts)  # show an "Open" dialog box and return the path to the selected file
        self.filename = ntpath.basename(self.filepath)
        if self.filename != "":
            self.label_text = Tkinter.StringVar()
            self.import_button.destroy()
            self.label = Tkinter.Label(self.main_window, textvariable=self.label_text, font=("Arial", 22), bg="white", relief="solid")
            self.label_text.set("Successfully imported")
            self.label.place(relx=.7, rely=.7, anchor="c")
            float_width = float(self.initial_width)
            float_height = float(self.initial_height)
            self.classify_button = Tkinter.Button(self.main_window, text=("Classify "),borderwidth=2,
                                                  font=self.arial_button_font,command=self.classify)
            self.classify_button.place(relx=(float_width - 20.0) / float_width, rely=(float_height - 20.0) / float_height, anchor="se")

    def classify(self):
        """
        Action for when classify button is pressed
        Allow user to choose the output directory when the classify button is pressed
        """
        self.label_text.set("Choose output directory")
        time.sleep(2.5)
        self.output_dir = askdirectory()
        if self.output_dir != "":
            self.import_button.destroy()
            self.classify_button.configure(state="disabled")
            self.label_text.set("Classifying " + self.filename)
            self.main_window.after(500, self.logistics_model)

    def read_data(self):
        """
        Function to read data from csv file,
        assuming the last column to be the independant variable 
        and all other variables to be dependant variables
        """
        df = pd.read_csv(self.filepath, header=None)
        y = df.iloc[:,-1]
        X = df[df.columns[:-1]]
        # convert from pandas dataframe to numpy array
        X = X.values
        y = y.values
        return (X, y)
    
    def normalize(self, X):
        """the depednant variables are z normlized, with a mean of zero and sd of 1"""
        return(X - np.mean(X, axis=0))/np.std(X, axis=0) 

    def train_test_split(self,X, y, test_size):
        """
        Splits the dataset to return test and training samples randomly
        """
        p = np.random.permutation(len(y))
        split_index = int(round(len(y)*test_size))
        test = p[:split_index]
        train = p[split_index:]
        return (X[train], X[test], y[train], y[test])
    
    def logistics_model(self):
        """
        the main function which calls the LogisticsClassifier
        and evaluates the classifier
        """
        X, y = self.read_data()
        X = self.normalize(X)
        (X_train, X_test, y_train, y_test) = self.train_test_split(X, y, test_size = .33)
        classifier = LogisticRegression(tolerance=1e-5, max_iter=100000, learning_rate=.025)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        confusion_matrix = self.confusion_matrix(y_test, predictions)
        print(confusion_matrix)
        self.cross_validation(X,y, 10);
        self.main_window.mainloop()

    def confusion_matrix(self, y_test, y_pred) :
        """
        the function generates a confusion matrix based on the predicted and the actual values
        """
        actual = pd.Series(y_test,name= 'Actual')
        predicted = pd.Series(y_pred, name = 'Predicted')
        output = pd.concat([actual, predicted], axis=1)
        output.to_csv(self.output_dir+'/output.csv', encoding='utf-8')
        return pd.crosstab(actual, predicted)
        
    def cross_validation(self, X, y, n_folds= 10):
        """
        the function performs cross validation n times based on the given input parameter
        """
        out = open(self.output_dir+'/output-cross-validation-txt','w')
        test_scores = np.array([])
        for i in range(n_folds) :
            out.write(str(i+1) + 'Split \n')
            (X_train, X_test, y_train, y_test) = self.train_test_split(X, y, test_size = .1)
            logit = LogisticRegression(tolerance=1e-5, max_iter=100000, learning_rate=.025)
            logit.fit(X_train, y_train)
            pred_test = logit.predict(X_test)
            out.write('Actual labels\n')
            y_test.tofile(out, sep=',')
            out.write('\nPredicted labels\n')
            pred_test.tofile(out, sep=',')
            test_score = np.mean(y_test == pred_test)
            out.write('\nAccuracy: ' + str(int(100*test_score)) + '%\n')
            test_scores = np.r_[test_scores, test_score]

        print 'test accuracies', test_scores.round(2)
        print 'mean test accuracy %.02f, standard deviation %.02f' %(np.mean(test_scores), np.std(test_scores))
        out.write('\nmean test accuracy %.02f, standard deviation %.02f' %(np.mean(test_scores), np.std(test_scores)))
        out.close()
        self.label_text.set("Classification complete!\nMean accuracy: " + str(round(100*np.mean(test_scores), 2)) + "%")
        plt.plot(test_scores)
        plt.savefig(self.output_dir+'/10fold-plot.png')
        img = Image.open(self.output_dir+'/10fold-plot.png')
        img.show()
