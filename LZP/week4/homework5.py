import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#计算欧氏距离
def olDis(dataset,centroids,k):
    clalist=[]
    for data in dataset:
        diff=np.tile(data,(k,1))-centroids
        #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff =diff**2    #平方
        squaredDist = np.sum(np.square(diff),axis=1)    #和(axis-1表示行）
        distance=squaredDist**0.5 #开根号
        clalist.append(distance)
    clalist=np.array(clalist)   #返回一个每个点到质点的距离Len(dataset)*k的数组
    return clalist

#计算质心
def classify(dataset,centroids,k):
    #计算样本到质心的距离
    clalist = olDis(dataset,centroids,k)
    #分组并计算新的质心
    minDistIndices=np.argmin(clalist,axis=1)     #axis=1 表示求出每行的最小值的下标
    newCentroids=pd.DataFrame(dataset).groupby(minDistIndices).mean()
    newCentroids=newCentroids.values
    #计算变化量
    changed =newCentroids-centroids
    return changed,newCentroids

#使用 k-means算法分类
def kmeans(dataset,k):
    #随机取质心
    centroids=random.sample(dataset,k)
    #更新质心，直到变化量为0
    changed,newCentroids=classify(dataset,centroids, k)
    while np.any(changed != 0):
        changed,newCentroids=classify(dataset,newCentroids, k)
    centroids=sorted(newCentroids.tolist())     #tolist()将矩阵转换成列表 sorted()排序
    #根据质心计算每个集群
    cluster =[]
    clalist =olDis(dataset,centroids,k)     #调用欧拉距离
    minDistIndices=np.argmin(clalist,axis=1)
    for i in range(k):
        cluster.append([])
    for i,j in enumerate(minDistIndices):       #enumerate()可同时遍历索引和遍历元素
        cluster[j].append(dataset[i])
    return centroids,cluster

#创建数据集
def creatDataset():
    return [[1,1],[1,2],[2,1],[2,2],[6,4],[6,3],[5,4]]


if __name__ == '__main__':
    dataset=creatDataset()
    centroids,cluster=kmeans(dataset,2)
    print("集群为：%s",cluster)
    print("质心为：%s",centroids)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0],dataset[i][1],c='blue',marker='o',s=100)

    for j in range(len(centroids)):
        plt.scatter(centroids[j][0],centroids[j][1],c='red',marker='*',s=100)

    plt.show()






