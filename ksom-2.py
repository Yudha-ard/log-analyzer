import pandas as pd
from minisom import MiniSom
from sklearn.metrics import silhouette_score
import os

def saveResults(results, resultDir, fileName):
    silhouetteDf = pd.DataFrame({'Silhouette Score': [results['silhouette']]})
    silhouetteDf.to_csv(os.path.join(resultDir, f'{fileName}_silhouette_score.csv'), index=False)

def processFile(tfidfFile, logFile, resultDir, fileName, sigma, learning_rate, iterations):
    dataDf = pd.read_csv(tfidfFile)
    with open(logFile, 'r') as file:
        logLines = file.readlines()
    dataDf['document'] = [f"doc_{i+1}" for i in range(len(dataDf))]
    dataDf['log_line'] = logLines[:len(dataDf)]
    dataArray = dataDf.drop(columns=['document', 'log_line']).to_numpy()
    
    somSize = (10, 10)
    som = MiniSom(x=somSize[0], y=somSize[1], input_len=dataArray.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.train_batch(dataArray, iterations)
    
    def getSomCluster(som, data):
        clusters = [som.winner(x) for x in data]
        return clusters

    clusters = getSomCluster(som, dataArray)
    dataDf['cluster'] = [f'{i}-{j}' for i, j in clusters]

    def calculateSilhouetteScore(data, cluster_labels):
        try:
            if len(set(cluster_labels)) > 1:
                return silhouette_score(data, cluster_labels, metric='euclidean')
            else:
                return None
        except ValueError as e:
            print(f"Silhouette Score Calculation Error: {e}")
            return None

    silhouette = calculateSilhouetteScore(dataArray, [f'{i}-{j}' for i, j in clusters])

    results = {
        'silhouette': silhouette,
        'som': som
    }

    saveResults(results, resultDir, fileName)
    return {
        'fileName': fileName,
        'sigma': sigma,
        'learning_rate': learning_rate,
        'iterations': iterations,
        'silhouette_score': silhouette
    }

def evaluateHyperparameters(tfidfFile, logFile, resultDir, fileName, sigma_values, learning_rate_values, iteration_values):
    best_results = []
    for iterations in iteration_values:
        print(f"Evaluating sigma={sigma_values}, learning_rate={learning_rate_values}, iterations={iterations}")
        result = processFile(tfidfFile, logFile, resultDir, fileName, sigma_values, learning_rate_values, iterations)
        best_results.append(result)
    return best_results

def main():
    tfidfFiles = [f'tfidf-{i+1}.csv' for i in range(15)]
    logFiles = [f'accesslog-{i+1}.log' for i in range(15)]
    resultDir = 'results/'

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    sigma_values = 0.3
    learning_rate_values = 0.05
    iteration_values = [2, 3, 6, 8, 10, 11, 14, 18, 21]

    best_results = []
    for tfidfFile, logFile in zip(tfidfFiles, logFiles):
        result = evaluateHyperparameters(tfidfFile, logFile, resultDir, os.path.basename(tfidfFile).replace('.csv', ''), sigma_values, learning_rate_values, iteration_values)
        best_results.extend(result)

    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv(os.path.join(resultDir, 'best_results.csv'), index=False)
    print("Hyperparameter evaluation completed. Results saved to best_results.csv")

if __name__ == "__main__":
    main()
