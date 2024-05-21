
import matplotlib.pyplot as plt
import  numpy as np
import seaborn as sns


def perfLoader(modelName):

    trainLoss = np.genfromtxt(modelName + "_train_loss.csv",  delimiter=",")
    valLoss= np.genfromtxt(modelName + "_val_loss.csv",  delimiter=",")
    time = np.genfromtxt(modelName + "_time.csv",  delimiter=",")

    return  trainLoss,valLoss,time


models = ["noResNes","binnedCond", "simpCond"]
model_dims = [ 32,64,128,256,256 + 128]


modelName = "transformerSimpModel"

tit = modelName  + " test error "
maxDepth =4
grid = np.zeros( (maxDepth ,len(model_dims )) )

for i,depth in enumerate(range(1, maxDepth )):
    for j,dim in enumerate(model_dims):


        fileName = "model_perf/" + modelName + "model_dim=" + str(dim)  + "depth=" + str(depth)

        trainLoss, valLoss, time = perfLoader( fileName)
        grid[i][j] = min(valLoss)


sns.set_theme()
plt.figure(figsize=(8, 6))
sns.heatmap(grid, cmap="viridis", linewidths=0.5, yticklabels= model_dims, vmin=0
            )
plt.xlabel('cond dim ')
plt.ylabel('model dim')

plt.title(tit)
plt.show()

print(" min error "  + str(np.min(grid)))


models = ["noResNes","binnedCond", "simpCond"]
model_dims = [ 32,64,128,256,256 + 128]
cond_dims = [1,4,16,32,64]



for modelName in models:

    tit = modelName  + " test error "
    grid = np.zeros( (len(model_dims),len(cond_dims)) )

    for i,dim in  enumerate(model_dims):
        for j,cond_dim in enumerate(cond_dims):


            fileName = "model_perf/" + modelName + "model_dim=" + str(dim) + "cond_dim=" + str(cond_dim)

            trainLoss, valLoss, time = perfLoader( fileName)
            grid[i][j] = min(valLoss)


    sns.set_theme()
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, cmap="viridis", linewidths=0.5, xticklabels=cond_dims, yticklabels= model_dims, vmin=0
                )
    plt.xlabel('cond dim ')
    plt.ylabel('model dim')

    plt.title(tit)
    plt.show()

    print(" min error "  + str(np.min(grid)))

            # plt.savefig("../figs/dis" + tit + ".png", dpi=300, bbox_inches='tight', format='png')


print("end")



