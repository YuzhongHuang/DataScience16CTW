import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from saliency import saliency

ax = sns.heatmap(saliency.load("./CDR_saliency/cor.pkl").mat)