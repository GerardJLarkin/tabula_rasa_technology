# create database to store patterns, and index them
from time import perf_counter, sleep, gmtime, strftime, localtime
import numpy as np
import random
from operator import itemgetter
import math
import datetime

def pattern_store(patoms):
    return None