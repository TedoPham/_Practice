import myFunctions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

fileName = r"./Data.txt"
    
maxIter = 500  #max interation to find centroid within a cluster
n_comp = 90    #reduce dimentionality
k = 10 #kclusters

SSEs = []
SSERs = []

data = np.loadtxt(open(fileName, "rb"), delimiter=",")   
data[data < 100] = 0    #remove blurr pixels
data[data >= 100] = 255

#================================================================== 
pca =  PCA(n_comp)
X = pca.fit_transform(data)

tsne = TSNE(2,perplexity=60, random_state=0, learning_rate=1000, early_exaggeration=20)
X = tsne.fit_transform(X)

# =============================================================================
# #clusters representation after dimension reduction
# plt.scatter(X[:,0],X[:,1])
# =============================================================================

cents,clusters,nClusters = myFunctions.splitAndCenter(X,k,maxIter) 
# =============================================================================
# centsR,clustersR,nClustersR = myFunctions.splitAndCenterRandom(X,k,maxIter) 
# =============================================================================

labels = myFunctions.putLabel(clusters,X)
# =============================================================================
# labels2 = myFunctions.putLabel(clustersR,X)
# =============================================================================

SSE = myFunctions.SSE(cents,clusters)
# =============================================================================
# SSER = myFunctions.SSE(centsR,clustersR)
# =============================================================================

print('SSE %f' %SSE)
# =============================================================================
# print('SSER %f' %SSER)
# =============================================================================

# =============================================================================
# for i in range(0,len(clusters)):
#     print('Cluser size: %d, SSE: %f' %(len(clustersR[i]),myFunctions.SSE([cents[i]],[clusters[i]])))
# for i in range(0,len(clustersR)):
#     print('Cluser(random) size: %d, SSE: %f' %(len(clustersR[i]),myFunctions.SSE([centsR[i]],[clustersR[i]])))
# =============================================================================

###myFunctions.writeListToFile(labels,"Image_clustering.txt")
    
plt.scatter(X[:,0],X[:,1])
for i,type in enumerate(cents):
    x = np.asarray(cents)[i,0]
    y = np.asarray(cents)[i,1]
    plt.scatter(x, y, marker='o')
    plt.text(x+0.3, y+0.3, i, fontsize=9, color='red')
plt.show()
    
for i in range(0,10):
    myFunctions.showImage(data,labels,i)
    myFunctions.showFirstFew(X,labels,i,50)
    
