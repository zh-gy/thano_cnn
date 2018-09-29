import os
import pandas as pd
import re
import jieba
import shutil
import collections
import numpy as np
from numpy import random
import json
import tensorflow as tf
from reduce_plot import reduce_plot

from creat_train_data import *
from train_wordvector import *
from Attention_classfier import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

