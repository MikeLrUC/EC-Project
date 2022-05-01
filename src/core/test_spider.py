from utils import *
import pandas as pd
import numpy as np

df = pd.DataFrame(np.loadtxt("spider.txt"), columns=["Fake", "Real"],)
Statistics.analyse(df, True)