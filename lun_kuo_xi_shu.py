from  sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
#定义轮廓系数函数，方便其他文件调用
def chioce_best (features_numpy,min_number=2,max_number=10,type = "mini",min_esp = 0.1,max_esp =0.2 ):
    values = []
    #minibach聚类
    if  type == "mini":
        for cluster_number in range(min_number,max_number):
            minibach_program = MiniBatchKMeans(n_clusters=cluster_number,batch_size=5000)
            cluster_result = minibach_program.fit_predict(features_numpy)
            value = silhouette_score(X=features_numpy,labels=cluster_result)
            values.append(value)
        #可视化输出
        plt.plot(np.arange(min_number, max_number), values, c="b")
        plt.grid
        plt.title(f"Silhouette Scores for minibach")
        plt.xlabel("group_number")
        plt.show()
    #谱聚类
    if type == "pu":
        for cluster_number in range(min_number,max_number):
            spectral_program = SpectralClustering(n_clusters=cluster_number)
            spectral_program.fit(features_numpy)
            clustering_result = spectral_program.labels_
            value = silhouette_score(X=features_numpy, labels=clustering_result_result)
            values.append(value)
        #可视化输出
        plt.plot(np.arange(min_number,max_number),values,c="b")
        plt.xlabel("group_number")
        plt.title(f"Silhouette Scores for SpectralClustering")
        plt.grid
        plt.show()
    #DBSCAN
    if type == "db":
        for sample_numbers  in range(min_number,max_number):
            for epsnumber in range (10*min_esp,10*max_esp):#循环双参数
                epsn = epsnumber/10
                density_program = DBSCAN(eps=epsn, min_samples=sample_numbers)
                density_program.fit(features_numpy)
                clustering_result = density_program.labels_
                # 检查是否至少有两个聚类
                if len(np.unique(clustering_result)) < 2:
                    print("Error: Less than two clusters after filtering noise. Cannot compute silhouette score.")
                    value = np.nan
                else:
                    # 计算轮廓系数
                    value = silhouette_score(features_numpy, clustering_result)
                mid = [sample_numbers,epsn,value]
                values.append(mid)
                print(f'sample numbers:{mid[0]},epsnumber:{mid[1]},score:{mid[2]}')
        values = np.array(values)
        #可视化输出
        plt.scatter(values[:,0],values[:,1],c=values[:,2],cmap="jet",s=8)
        plt.colorbar(label="Silhouette Score")
        plt.xlabel("min_samples")
        plt.ylabel("eps")  # y轴是占位符，实际上只用于可视化
        plt.title(f"Silhouette Scores for DBSCAN")
        plt.colorbar
        plt.show()