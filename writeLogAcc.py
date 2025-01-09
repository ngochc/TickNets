# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:35:11 2022

@author: tuann
"""
import os.path
import csv

from datetime import datetime


def writeLogAcc(filename='LogAcc1.txt', strtext=''):
    if not os.path.exists(filename):
        file1 = open(filename, "w")
    else:
        file1 = open(filename, "a")
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    strtext = dt_string + ' ' + strtext + '\n'
    file1.writelines(strtext)
    file1.close()  # to change file access modes


def log_results_to_csv(file_path, current_epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Log the results to a CSV file.
    """
    mode = open(file_path, 'w' if not os.path.exists(file_path) else 'a')

    result_row = [
        current_epoch + 1,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy
    ]

    with open(file_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_row)
