import numpy as np
import matplotlib.pyplot as plt

modelName =  "vanUNet"

def perfLoader(modelName):

    trainLoss = np.genfromtxt(modelName + "_train_loss.csv",  delimiter=",")
    valLoss= np.genfromtxt(modelName + "_val_loss.csv",  delimiter=",")
    time = np.genfromtxt(modelName + "_time.csv",  delimiter=",")

    return  trainLoss,valLoss,time


# using loadtxt()

mod1Train, mod1Val,mod1time = perfLoader(modelName)

# drop nan ind
mod1Val = mod1Val[~np.isnan(mod1Val)]



minVal = np.argmin(mod1Val)


gap = 1
startOffset = 5
# take values only up to min val
mod1Train = mod1Train[startOffset:minVal+1000:gap]
mod1Val =  mod1Val[startOffset:minVal+1000:gap]


# xpoints = list(range(0,len(mod1Train)))
# plt.plot(xpoints, mod1Train,'r')


xpoints = list(range(startOffset,len(mod1Val)+startOffset))
plt.plot(xpoints, mod1Val,'b')
plt.show()


print("end")