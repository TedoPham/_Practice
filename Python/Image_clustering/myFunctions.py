import numpy as np
import csv
import random
import copy
from sklearn.metrics.pairwise import euclidean_distances

#split cluster into 2, return centroids in list
#k should be 2 or 3 
def splitCluster(cluster,k=2):
    cents = []
    if(type(cluster) is list):  
        cl_array = np.asarray(cluster,dtype=float)
    else:
        cl_array = copy.deepcopy(cluster)
        
    mid = cl_array.mean(0)
    
    #pick farthest distance from mid
    distance = euclidean_distances(cl_array, [mid])
    Sort_Indices = (-distance).argsort(axis=0)
    temp = np.vstack((mid,cl_array[ Sort_Indices[0][0] ]))  
    cents.append( temp.mean(0) )
    
    #pick farthest distance from first centroid
    distance = euclidean_distances(cl_array, [cents[0]])
    Sort_Indices = (-distance).argsort(axis=0)
    temp = np.vstack((mid,cl_array[ Sort_Indices[0][0] ]))
    cents.append(temp.mean(0))
    
    if(k == 3):
        cents.append(mid)
    
    return  cents
             
#move points in cents to the mean of the cluster
def findMean(cents,cluster):
    if(type(cluster) is list):  
        cl_array = np.asarray(cluster,dtype=float)
        cl = cluster.copy()
    else:
        cl_array = copy.deepcopy(cluster)
        cl = cluster.tolist()
        
    #create empty cluster list
    clters = []
    for i in range(0,len(cents)):
        clters.append([])
    
    #calculate distances from all points in cluster to each centroid    
    distances = []
    for i in range(0,len(cents)):
        distances.append(euclidean_distances(cl_array, [cents[i]]))

    #divide cluster by the nearest distance to each centroid
    for i in range(0,len(cl_array)):
        smallest = distances[0][i][0]
        u = 0
        for o in range(1,len(cents)):
            if distances[o][i][0] < smallest:
                u = o
                smallest = distances[o][i][0]
# =============================================================================
#         clters[u].append(cl_array[i])
# =============================================================================
        clters[u].append(cl[i])
        
    #calculate new means of each cluster
    newCents = []
    for i in range(0,len(clters)):
        cl_array = np.asarray(clters[i],dtype=float)
        m = cl_array.mean(0)
        newCents.append(m)
    
    return newCents,clters

def center(cents,cluster,maxIter):
    newCents = []
    clters = []
    i = 0
    for x in range(0,maxIter):
       newCents,clters = findMean(cents,cluster)
       if(np.array_equal(newCents,cents)):      #stop if centroid doesn't change
           break
       else:
           cents = newCents.copy()
           
       i = x
# =============================================================================
#     print('Centered after %d iteration' %(i+1))
# =============================================================================
    return newCents,clters

#min split is k=2    
def splitAndCenter(cluster,k,maxIter=0):   
    cents = splitCluster(cluster)
    cents,clters = center(cents,cluster,maxIter)
    
    nCluster = 2 
    
    while(nCluster < k):
        index = 0
        
        #find the most spread cluster and split it
        largestError = SSE([cents[0]],[clters[0]])
        for i in range(1, len(cents)):
            e = SSE([cents[i]],[clters[i]])
            if(largestError < e):
                largestError = e
                index = i
    
        #spliting the sub cluster smaller
        subCluster = clters[index]
        subCents = splitCluster(subCluster)
        tempCents,tempClters = center(subCents,subCluster,maxIter)
       
        # delete old clusters form 
        del cents[index]
        del clters[index]
       
        # add new splitted clusters
        cents.append(tempCents[0])
        cents.append(tempCents[1])
        clters.append(tempClters[0])
        clters.append(tempClters[1])
          
        nCluster += 1
        
    cents,clters = center(cents,cluster,maxIter)    
    return cents,clters,nCluster
      
def SSE(cents,clusters):
    sse = 0;
    for i in range(0,len(cents)):
        d = euclidean_distances(clusters[i], [cents[i]])
        d = d*d
        sse += d.sum()
    return sse;

#assign labels to each entry data
def putLabel(clusters,data):
    label = [None]*len(data)

# =============================================================================
#     data_a = np.asarray(data,dtype=float)
# =============================================================================
    data_a = data.tolist()
    
    for i in range(0,len(data)):
        for o in range(0,len(clusters)):
# =============================================================================
#             if (np.array_equal(data_a[i],clusters[o])):
# =============================================================================
# =============================================================================
#             if (((data_a[i]==clusters[o]).all(axis=1)).any() ):
# =============================================================================
              if(data_a[i] in clusters[o] ):  
                label[i] = o
                break 
         
    return label        
    
def splitAndCenterRandom(cluster,k=2,maxIter=100):
    cents = []
    if(type(cluster) is list):  
        cl_array = np.asarray(cluster,dtype=float)
    else:
        cl_array = copy.deepcopy(cluster)
    
# =============================================================================
#     for i in range(0,k):
#         r = random.randint(0,len(cluster)-1)
#         cents.append(cl_array[r])
# =============================================================================
    r = random.sample(range(1, len(cluster)-1), k)
    for i in range(0,k):
         cents.append(cl_array[r[i]])    
         
    cents,clters = center(cents,cluster,maxIter)
    
    return cents,clters,k

def mergClusters(clusters,cents,k):
    cl = copy.deepcopy(clusters)
    c = copy.deepcopy(cents)

    while(len(cl) > k):
        d = euclidean_distances(c, c)
        
        SI = (d).argsort(axis=None)
        
        #getting closest pair of cluster
        p1 = SI[len(cl)] % len(cl)
        p2 = int(SI[len(cl)] / len(cl))
        
        print('%d %d' %(p1,p2))
        #merge clusters
        cl[p1] += cl[p2]
        c[p1] = (c[p1] + c[p2]) / 2
        
        del(c[p2])
        del(cl[p2])
        for i in range(0,len(cl)):
            print('%d Cluser size: %d, SSE: %f' %(i,len(cl[i]),SSE([cents[i]],[cl[i]])))
    
    return cl  
        
def pull(cents,data):
    clusters = [ [] for i in range(0,len(cents))]
    labels = []
    for i in range(0,len(data)):
        d = euclidean_distances([data[i]], cents)
        SI = (d).argsort(axis=None)       
        (clusters[SI[0]]).append((data[i]).tolist())
        labels.append(SI[0])
        
    return clusters, labels    

def writeListToFile(lis,fileName):
    with open(fileName, 'w') as f:
        for item in lis:
            f.write("%s\n" % item)
            
import matplotlib.pyplot as plt            
def showImage(data,label_list, label):
    n_image = 50
    imgs = []
    for i in range(0,len(data)):
        if len(imgs) <= n_image and label_list[i] == label:
            imgs.append(data[i])
    
    fig=plt.figure(figsize=(8, 8))
    columns = int(n_image / 5)
    rows = int(n_image/columns)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.reshape(imgs[i-1],(28,28) ) )

def showFirstFew(data,label_list, label, n):
    dots = []
    for i in range(0,len(data)):
        if len(dots) <= n and label_list[i] == label:
            dots.append(data[i])
    x = np.asarray(dots)[:,0]
    y = np.asarray(dots)[:,1]
    plt.scatter(x, y, marker='x')      