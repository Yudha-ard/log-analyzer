import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import joblib
import os

def saveResults(results, resultDir, fileName):
    # Save results as CSV
    blockCodeCounts = pd.DataFrame(list(results['blockCodeCounts'].items()), columns=['Block Code', 'Count'])
    blockCodeCounts.to_csv(os.path.join(resultDir, f'{fileName}_block_code_counts.csv'), index=False)
    
    clusterDistribution = pd.DataFrame(results['clusterDistribution']).fillna(0)
    clusterDistribution.to_csv(os.path.join(resultDir, f'{fileName}_cluster_distribution.csv'))

    documentsPerCluster = pd.DataFrame([
        {'Cluster': cluster, 'Document': doc, 'Log Line': logLine}
        for cluster, docs in results['documentsPerCluster'].items()
        for doc, logLine in docs
    ])
    documentsPerCluster.to_csv(os.path.join(resultDir, f'{fileName}_documents_per_cluster.csv'), index=False)

    # Save model
    joblib.dump(results['som'], os.path.join(resultDir, f'{fileName}_som_model.pkl'))

    # Save silhouette score
    silhouetteDf = pd.DataFrame({'Silhouette Score': [results['silhouette']]})
    silhouetteDf.to_csv(os.path.join(resultDir, f'{fileName}_silhouette_score.csv'), index=False)

    # Plot and save distance map with coordinates
    distance_map = results['som'].distance_map()
    plt.figure(figsize=(10, 10))
    plt.imshow(distance_map, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f'SOM Distance Map for {fileName}')

    # Annotate each point with its coordinates
    for i in range(distance_map.shape[0]):
        for j in range(distance_map.shape[1]):
            plt.text(j, i, f'({i},{j})', color='black', ha='center', va='center', fontsize=8)

    plt.savefig(os.path.join(resultDir, f'{fileName}_distance_map.png'))
    plt.close()

def processFile(tfidfFile, logFile, resultDir, fileName):
    dataDf = pd.read_csv(tfidfFile)
    with open(logFile, 'r') as file:
        logLines = file.readlines()
    dataDf['document'] = [f"doc_{i+1}" for i in range(len(dataDf))]
    dataDf['log_line'] = logLines[:len(dataDf)]
    dataArray = dataDf.drop(columns=['document', 'log_line']).to_numpy()
    scaler = StandardScaler()
    dataArrayScaled = scaler.fit_transform(dataArray)
    somSize = (10, 10)
    som = MiniSom(x=somSize[0], y=somSize[1], input_len=dataArrayScaled.shape[1], sigma=2.6, learning_rate=0.5)
    som.train_batch(dataArrayScaled, 10)
    
    def getSomCluster(som, data):
        clusters = [som.winner(x) for x in data]
        return clusters

    clusters = getSomCluster(som, dataArrayScaled)
    dataDf['cluster'] = [f'{i}-{j}' for i, j in clusters]
    blockCodeMapping = {
        'XSS': ['alert', 'confirm'],
        'SQLI': ['union', 'select'],
        'CRLF': ['location', 'cookie']
    }

    def calculateBlockCodeCounts(dataDf, blockCodeMapping):
        counts = defaultdict(int)
        for block, cols in blockCodeMapping.items():
            counts[block] = dataDf[cols].sum(axis=1).apply(lambda x: x > 0).sum()
        return counts

    blockCodeCounts = calculateBlockCodeCounts(dataDf, blockCodeMapping)

    def calculateClusterDistribution(dataDf, blockCodeMapping):
        clusterDistribution = defaultdict(lambda: defaultdict(int))
        for block, cols in blockCodeMapping.items():
            mask = dataDf[cols].sum(axis=1) > 0
            clusters = dataDf[mask]['cluster']
            for cluster in clusters:
                clusterDistribution[block][cluster] += 1
        return clusterDistribution

    clusterDistribution = calculateClusterDistribution(dataDf, blockCodeMapping)

    def getDocumentsPerCluster(dataDf):
        documentsPerCluster = defaultdict(list)
        for _, row in dataDf.iterrows():
            documentsPerCluster[row['cluster']].append((row['document'], row['log_line']))
        return documentsPerCluster

    documentsPerCluster = getDocumentsPerCluster(dataDf)

    def calculateSilhouetteScore(data, clusters):
        clusterLabels = [f'{i}-{j}' for i, j in clusters]
        try:
            return silhouette_score(data, clusterLabels, metric='euclidean')
        except ValueError:
            return None

    silhouette = calculateSilhouetteScore(dataArrayScaled, clusters)

    results = {
        'blockCodeCounts': blockCodeCounts,
        'clusterDistribution': clusterDistribution,
        'documentsPerCluster': documentsPerCluster,
        'silhouette': silhouette,
        'som': som
    }

    saveResults(results, resultDir, fileName)
    return results, dataDf

def main():
    tfidfFiles = [f'tfidf-{i+1}.csv' for i in range(15)]
    logFiles = [f'accesslog-{i+1}.log' for i in range(15)]
    resultDir = 'results'

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    for tfidfFile, logFile in zip(tfidfFiles, logFiles):
        fileName = os.path.basename(tfidfFile).replace('.csv', '')
        processFile(tfidfFile, logFile, resultDir, fileName)

if __name__ == "__main__":
    main()
