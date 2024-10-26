from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def Feature_Correlation_Scores(original_df, reduced_df):
    """
    Takes the DataFrame with original input features, and dataframe after dimensionality reduction, 
    and computes the FCS and FCSS scores.
    To compute nonzero correlations, and increase sparcity while doing so, all correlations are rounded to one decimal point.
    Furthermore, for any original feature with only one unique value, the correlation of that feature with all reduced components is set to 0.
    
    """
    reduced_components = [i for i in reduced_df.columns if i !='target']
    original_features = [i for i in original_df.columns if i !='target']
    max_correlations = []
    nonzero_count = []
    for component in reduced_components:
        correlations = {}
        for feature in original_features:
            if len(original_df[feature].unique())>1:
                corr, p = spearmanr(reduced_df[component], original_df[feature])
                correlations[feature] = corr
            elif len(original_df[feature].unique())<=1:
                correlations[feature] = 0
        max_correlations.append(max(correlations.values()))
        nonzero_count.append(sum([round(correlation,1) != 0 for correlation in correlations.values()]))

    fcs = sum(max_correlations)/len(reduced_components)
    fcss = 1- sum(nonzero_count)/(len(reduced_components)*len(original_features))

    return fcs, fcss



def DBI_beam(subgroups, df): 
    """
    Takes the subgroups found by the subgroup detection algorithms, and the original dataframe, and computes the DBI.
    Centroids are the mean of the subgroup, and euclidean distance is used to compute distance between values and centroids, 
    and distance between centroids.
    """
    features = [feature for feature in df.columns if feature != 'target']
    centroids = []
    subgroup_cohesion = []
    for subgroup_index in range(len(subgroups)):
        subgroup_df = df[df.eval(str(subgroups[subgroup_index][1]).replace("', '", ' and ').replace("['", '').replace("']", ''))][features].astype(float)
        centroid = subgroup_df.mean()
        centroids.append(centroid)
        avg_distance_to_centroid = np.linalg.norm(subgroup_df-centroid, axis=1).mean() 
        subgroup_cohesion.append(avg_distance_to_centroid)
    
    # for subgroup_index in range(len(subgroups)): #TO CHECK SUBGROUPS FOR EXTREMELY LARGE DB INDEX
    #     condition = str(subgroups[subgroup_index][1]).replace("', '", ' and ').replace("['", '').replace("']", '')
    #     print(f"Subgroup {subgroup_index} condition: {condition}")
    #     print(f"Subgroup {subgroup_index} size: {len(df[df.eval(condition)])}")
    
    maxima = []
    for i in range(len(centroids)):
        k=0
        for j in range(len(centroids)):
            if i!=j:
                centroid_dist = np.linalg.norm(centroids[i]-centroids[j])
                if centroid_dist > 0:
                    value = subgroup_cohesion[i] + subgroup_cohesion[j]/centroid_dist
                elif centroid_dist == 0:
                    value = subgroup_cohesion[i] + subgroup_cohesion[j]/np.finfo(float).eps
                if value > k:
                    k = value
        maxima.append(k)
    dbi = np.mean(maxima)


    return dbi

def DBI_ps(subgroups, df): 
    """
    Takes the subgroups found by the subgroup detection algorithms, and the original dataframe, and computes the DBI.
    Centroids are the mean of the subgroup, and euclidean distance is used to compute distance between values and centroids, 
    and distance between centroids.
    """

    features = [feature for feature in df.columns if feature != 'target']
    centroids = []
    subgroup_cohesion = []
    for subgroup_index in range(len(subgroups)):
        oper = str(subgroups["subgroup"][subgroup_index])
        oper = oper.replace("AND", "&")
        if oper.find(":") >= 0 :
            newOpers = []
            splitOper = oper.split(" & ")
            for j in range(len(splitOper)-1, -1, -1) :
                dpIndex = splitOper[j].find(":")
                if dpIndex >= 0 :
                    attr = splitOper[j][:dpIndex]
                    brIndex = splitOper[j].find("[")
                    dpIndex2 = splitOper[j].find(":", dpIndex+1)
                    lb = splitOper[j][brIndex+1:dpIndex2]
                    ub = splitOper[j][dpIndex2+1:-1]
                    newOpers.append(attr+">="+lb)
                    newOpers.append(attr+"<="+ub)
                    del splitOper[j]
            splitOper += newOpers
            oper = " & ".join(splitOper)
        subgroup_df = df[df.eval(oper)][features].astype(float)
        # subgroup_df = df[df.eval(str(subgroups.loc[subgroup_index, "subgroup"]).replace('AND', 'and'))].astype(float)
        centroid = subgroup_df.mean()
        centroids.append(centroid)
        avg_distance_to_centroid = np.linalg.norm(subgroup_df-centroid, axis=1).mean() 
        subgroup_cohesion.append(avg_distance_to_centroid)

    maxima = []
    for i in range(len(centroids)):
        k=0
        for j in range(len(centroids)):
            if i!=j:
                centroid_dist = np.linalg.norm(centroids[i]-centroids[j])
                if centroid_dist > 0:
                    value = subgroup_cohesion[i] + subgroup_cohesion[j]/centroid_dist
                elif centroid_dist == 0:
                    value = subgroup_cohesion[i] + subgroup_cohesion[j]/np.finfo(float).eps
                if value > k:
                    k = value
        maxima.append(k)
    dbi = np.mean(maxima)


    return dbi
