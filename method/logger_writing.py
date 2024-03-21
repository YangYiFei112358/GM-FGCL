#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 20:50
# @Author  : YOUR-NAME
# @FileName: logger_writing.py
# @Software: PyCharm


import sys
import numpy as np
from datetime import datetime

Run_Time = datetime.today().date()


class Logger(object):
    def __init__(self, filename='./Run_Recording/Run_Recording_{}_'.format(Run_Time), stream=sys.stdout):
        self.terminal = stream
        self.version = int(np.loadtxt('./Run_Recording/version.txt', delimiter=',')[0])
        self.log = open(filename + str(self.version) + '.log', 'a+')
        self.run = 1

    def write(self, message):
        message = message
        self.terminal.write(message)
        self.log.write(message)

    def show_version(self):
        print("_____________________Version=={}_____________________".format(self.version))
        np.savetxt('./Run_Recording/version.txt', [self.version + self.run, self.version], delimiter=',')

    def flush(self):
        pass
