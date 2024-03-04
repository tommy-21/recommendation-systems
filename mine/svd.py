##============================================
##============================================
## Singular Value Decomposition
##============================================
##============================================
# remplace les valeurs manquantes de façon naïve, puis retourne la matrice de rang k la plus proche_

# fonctions pour la recommandation
# * svd.complete(M_train, k, ___)
# * svd.recommend(M_train, id_user, new=True, k=10, ___)

import numpy as np

##============================================
## utile : replaceNA_with_zeros(M.train)
##============================================
def replaceNA_with_zeros(M_train):
    res = M_train.copy()
    res[np.isnan(res)] = 0
    return res


##============================================
## svd.complete(M_train, k)
##============================================
def complete(M_train, k, replaceNA_fn=replaceNA_with_zeros):
    res = np.zeros(M_train.shape)
    M_train = replaceNA_fn(M_train) # replace NA with zeros
    U, s, V = np.linalg.svd(M_train, full_matrices=False) # singular value decomposition
    s = np.diag(s) # diagonal matrix
    res = np.dot(U[:, :k], np.dot(s[:k, :k], V[:k, :])) # rank k approximation of M_train
    return res


##============================================
## svd.recommend(M_train, id_user, new=True, k=10, ___)
##============================================
def recommend(M_train, id_user, new=True, k=10, replaceNA_fn=replaceNA_with_zeros):
    M_completed = complete(M_train, k, replaceNA_fn) # complete the traning data matrix using SVD
    scores = M_completed[id_user] # get the estimated ratings of our user

    if new: # if we want to recommend a new movie
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0] 
        rec_ind_in_unknown = np.argmax(scores[inds_unknown]) 
        return inds_unknown[rec_ind_in_unknown]
    else: # else, we just return the index of the movie with the highest estimated rating
        return np.argmax(scores)
        

