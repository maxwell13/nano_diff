import  pandas as pd
from itertools import permutations, product
import matplotlib.pyplot as plt
import  numpy as np
import seaborn as sns

def mulitPeakCheck(oneRow,colorCenter,margin):
    centers = oneRow[['Peak 1 b', 'Peak 2 b', 'Peak 3 b', 'Peak NIR b']]


    EC = 1e9 * (4.135667516E-15) * (299792458) /   colorCenter
    ES = EC - margin
    EE = EC + margin

    inDis = False

    for cent in centers:
        if cent > ES and cent < EE:
            inDis = True

    return inDis


def singPeakCheck(oneRow,colorCenter,margin):
    centers = oneRow[['Peak 1 b', 'Peak 2 b', 'Peak 3 b', 'Peak NIR b']]


    EC = 1e9 * (4.135667516E-15) * (299792458) /   colorCenter
    ES = EC - margin
    EE = EC + margin

    inDis = False

    for cent in centers:
        if cent > ES and cent < EE:
            inDis = True
        else:
            return False

    return inDis




def occCounts(seqs):
    occ  = {}
    for stepSize in range(1,3):
        for ind in range(len(seqs)-stepSize+1):
            lets = seqs[ind:ind+stepSize]
            if lets in occ:
                occ[lets]+=1
            else:
                occ[lets]=1

    return  occ


def freqAgg(freqs):
    totFreq = {}
    for freqDic in freqs:
        for letComb in freqDic:
            if letComb in totFreq:
                totFreq[letComb] += freqDic[letComb]
            else:
                totFreq[letComb] = freqDic[letComb]
    return  totFreq


def combReturner(length):
    letters = "ACTG"
    letPerms = list(  product(letters,repeat=length))
    letFormated = [0] * len( letPerms)
    for i, ele in enumerate(  letPerms):
        letFormated[i] = ''.join(ele)
    return letFormated


def basicBar(lets,freqs,methodName):

    refindLets = []
    for let in lets:
        if let in freqs:
            refindLets.append(let)
    lets = refindLets

    tot = sum([freqs[x] for x in lets])

    # creating the bar plot
    plt.bar(lets, [freqs[x]/tot for x in lets], color='maroon',
            width=0.4)

    plt.xlabel("Letter Comb")
    plt.ylabel("Freq")
    plt.title(methodName)








def plotColor(df,color):

    if color =="green":
        colorCenter = 500

    if color == "red":
        colorCenter = 620

    if color == "far_red":
        colorCenter = 670

    if color == "nir":
        colorCenter = 950

    if color == "nir_2":
        colorCenter = 1100



    # gets letter combiations
    singLet = combReturner(1)
    twoLet =  combReturner(2)



    # 80/800 = 0.1 -> x/2 = 0.1  -> x = .2
    margin = .2



    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    freqs = GTMulti.apply( occCounts)
    totFreq = freqAgg(freqs)


    basicBar(singLet ,totFreq,color + " GT M")
    basicBar(twoLet ,totFreq,color + " GT M")


    GTSing = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1)]['Sequence']
    freqs = GTSing.apply( occCounts)
    totFreq = freqAgg(freqs)

    basicBar(singLet ,totFreq, color + " GT S")
    basicBar(twoLet ,totFreq,color + " GT S")



    file_path = color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    freqs = df.iloc[:,0].apply( occCounts)
    totFreq = freqAgg(freqs)

    basicBar(singLet ,totFreq,color + " Diff Gen")
    basicBar(twoLet ,totFreq, color + " Diff Gen")


def plotColorSing(df,color):

    if color =="green":
        colorCenter = 500

    if color == "red":
        colorCenter = 620

    if color == "far_red":
        colorCenter = 670

    if color == "nir":
        colorCenter = 950

    if color == "nir_2":
        colorCenter = 1100



    # gets letter combiations
    allLets = combReturner(1)
    allLets.extend(combReturner(2))

    def seriesToRelFreq(freqSeries):
        freqs = freqSeries.apply(occCounts)
        totFreq = freqAgg(freqs)

        tot = 0
        rawFreq = np.zeros(len(allLets))
        for i,let in enumerate(allLets):
            if let in totFreq:
                rawFreq[i] = totFreq[let]
                tot += rawFreq[i]

        if tot ==0:
            return  rawFreq
        else:
            return  rawFreq/tot


    # 80/800 = 0.1 -> x/2 = 0.1  -> x = .2
    margin = .2

    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    GTMultiFreq = seriesToRelFreq(GTMulti)


    GTSing = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1)]['Sequence']
    GTMultiSing = seriesToRelFreq(GTSing)


    file_path = color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    Dis = seriesToRelFreq(df.iloc[:,0])

    X_axis = np.arange(len( allLets))

    plt.bar(X_axis - 0.1, GTMultiFreq, 0.4, label='GT multi')
    plt.bar(X_axis + 0.1, GTMultiSing, 0.4, label='GT Sing')
    plt.bar(X_axis + 0.2, Dis, 0.4, label='Diff')

    plt.xticks(X_axis,  allLets)
    plt.xlabel("Let")
    plt.ylabel("Relative freq")
    plt.title(color)
    plt.legend()
    plt.show()


def seriesToRelFreq(freqSeries,allLets):
    freqs = freqSeries.apply(occCounts)
    totFreq = freqAgg(freqs)

    tot = 0
    rawFreq = np.zeros(len(allLets))
    for i,let in enumerate(allLets):
        if let in totFreq:
            rawFreq[i] = totFreq[let]
            tot += rawFreq[i]

    if tot ==0:
        return  rawFreq
    else:
        return  rawFreq/tot


def plotDifWaves(seriesList,lets,plotName=None):


    DisGreen = seriesToRelFreq(seriesList[0],lets)
    DisRed = seriesToRelFreq(seriesList[1],lets)
    DisFarRed = seriesToRelFreq(seriesList[2],lets)
    DisNir = seriesToRelFreq(seriesList[3],lets)
    DisNir2 = seriesToRelFreq(seriesList[4],lets)

    X_axis = np.arange(len(lets))

    plt.bar(X_axis + -0.3, DisGreen, 0.15, color='green', label='Green')
    plt.bar(X_axis + - 0.15, DisRed, 0.15, color='red', label='Red')
    plt.bar(X_axis, DisFarRed, 0.15, color='darkred', label='Far Red')
    plt.bar(X_axis + 0.15, DisNir, 0.15, color='purple', label='NIR')
    plt.bar(X_axis + 0.3, DisNir2, 0.15, color='black', label='NIR 2')

    plt.xticks(X_axis,lets)
    plt.xlabel("Let")
    plt.ylabel("Relative freq")
    plt.title(plotName)
    plt.legend()
    plt.show()



def plotAllDiff(model):



    # gets letter combiations
    allLets = combReturner(1)
    singLets = allLets.copy()
    twoLets = combReturner(2)
    allLets.extend(twoLets )

    seriesList = []

    color ="green"


    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:,0])

    color = "red"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "far_red"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "nir"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "nir_2"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    plotName= " diffusion "
    plotDifWaves(seriesList, singLets, plotName=plotName)
    plotDifWaves(seriesList, twoLets, plotName=plotName)



def plotAllMulti(df):


    # gets letter combiations
    allLets = combReturner(1)
    singLets = allLets.copy()
    twoLets = combReturner(2)
    allLets.extend(twoLets )


    # 80/800 = 0.1 -> x/2 = 0.1  -> x = .2
    margin = .2
    # 5/800 = 0.00625 -> x/2 = 0.00625  -> x = .0125
    margin = .0125

    color = "green"
    colorCenter = 500
    seriesList =[]
    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti )

    color = "red"
    colorCenter = 620
    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)


    color = "far_red"
    colorCenter = 670
    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)


    color = "nir"
    colorCenter = 950
    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)

    color = "nir_2"
    colorCenter = 1100
    GTMulti = df.loc[ df.apply(mulitPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)

    plotName = " GT multi peak "
    plotDifWaves(seriesList, singLets, plotName=plotName)
    plotDifWaves(seriesList, twoLets, plotName=plotName)


def plotAllSing(df):


    # gets letter combiations
    allLets = combReturner(1)
    singLets = allLets.copy()
    twoLets = combReturner(2)
    allLets.extend(twoLets )


    # 80/800 = 0.1 -> x/2 = 0.1  -> x = .2
    margin = .4

    color = "green"
    colorCenter = 500
    seriesList =[]
    GTMulti = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti )

    color = "red"
    colorCenter = 620
    GTMulti = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)


    color = "far_red"
    colorCenter = 670
    GTMulti = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)


    color = "nir"
    colorCenter = 950
    GTMulti = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)

    color = "nir_2"
    colorCenter = 1100
    GTMulti = df.loc[ df.apply(singPeakCheck,args=(colorCenter,margin), axis=1) ]['Sequence']
    seriesList.append(GTMulti)

    plotName = " GT sing peak"
    plotDifWaves(seriesList, singLets, plotName=plotName)
    plotDifWaves(seriesList, twoLets, plotName=plotName)




def posOcc(lets):

    charToInt = {"A": 1, "C": 2, "G": 3, "T": 4}
    orderTrans = {1:1,2:2,3:4,4:3}
    occArr = np.zeros( (4,len(lets)) )
    for i,let in enumerate(lets):
        position =   orderTrans[charToInt[let]]-1
        occArr[position][i]=1

    return  occArr


def plotOneHeatmap(data,tit):

    sns.set_theme()
    plt.figure(figsize=(8, 2.5))
    positions = [ "P" + str(i) for i in range(1,11)]
    nucleotides = ["A","C","T","G"]
    sns.heatmap(data, cmap="viridis", linewidths=0.5, xticklabels=positions, yticklabels=nucleotides, vmin=0, vmax=0.5)
    plt.xlabel('Position')
    plt.ylabel('Nucleobase')

    plt.title(tit)
    plt.savefig("../figs/dis" + tit +".png", dpi=300, bbox_inches='tight', format='png')

    # plt.show()



def plotHeatmaps(seriesList,lets,plotName=None):


    DisGreen = sum(seriesList[0].apply( posOcc))
    plotOneHeatmap(DisGreen/100, plotName + "Green")

    DisRed = sum(seriesList[1].apply( posOcc))
    plotOneHeatmap(DisRed/100,plotName + "Red")

    DisFarRed =  sum(seriesList[2].apply( posOcc))
    plotOneHeatmap(DisFarRed/100,plotName + "Far Red")

    DisNir = sum(seriesList[3].apply( posOcc))
    plotOneHeatmap(DisNir /100,plotName + "Nir")

    DisNir2 =  sum(seriesList[4].apply( posOcc))
    plotOneHeatmap(DisNir2/100,plotName + "Nir2")



def heatMapDiff(model):




    # gets letter combiations
    allLets = combReturner(1)
    singLets = allLets.copy()
    twoLets = combReturner(2)
    allLets.extend(twoLets )

    seriesList = []

    color ="green"


    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:,0])

    color = "red"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "far_red"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "nir"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    color = "nir_2"
    file_path =model  + "" + color + "_seqs.txt"
    df = pd.read_csv(file_path,header=None).squeeze(axis=0)
    seriesList.append(df.iloc[:, 0])

    plotName= model
    plotHeatmaps(seriesList, singLets, plotName=plotName)
    plotHeatmaps(seriesList, twoLets, plotName=plotName)


model = "simpCond"
model  = "binnedCond"
heatMapDiff(model)
model = "simpCond"
heatMapDiff(model)


file_path = "../data/seq_with_dis.csv"
df = pd.read_csv(file_path)

# plotAllSing(df)

# plotAllDiff(model)
#
# plotAllMulti(df)

# # plotColor(df,"")
# plotColorSing(df,"green")
# #
# # plotColorSing(df,"red")
#
# plotColorSing(df,"nir")
# # plotColorSing(df,"nir_2")
#
#


## just to verify
# count = 0
# for row in GTMulti:
#     for let in row:
#         if let=="A":
#             count+=1



print("end")