##============================================
##============================================
## kppv
##============================================
##============================================
# Recommande des films en utilisant la stratégie des k plus proches utilisateurs
# La similarité est claculée selon le critère *cosinus*

# * popularity.recommend(M_train, id_user, new) : recommande un film
# * popularity.complete(M_train) : complète la matrice


# utilise les sous-fonctions
# * cosinus(M_train, u1, u2)
# * complete_a_user_knn(M_train, id_user, k)

import numpy as np

##============================================
## cosinus(M_train, u1, u2)
##============================================
def cosinus(M_train, u1, u2):
    # films en commun
    inds_movie = np.where(np.sum(np.isnan(M_train[[u1, u2], ]), axis=1) == 0)[0]
    
    # cosinus
    if len(inds_movie) != 0:
        n1 = M_train[u1, inds_movie]
        n2 = M_train[u2, inds_movie]
        cos = sum(n1*n2) / np.sqrt(sum(n1**2)) / np.sqrt(sum(n2**2))
        return cos

    else:
        return 0



##============================================
## complete_a_user_knn(M_train, id_user, k)
##============================================
def complete_a_user(M_train, id_user, k):
    scores = np.zeros(M_train.shape[1])
    for id_item in range(M_train.shape[1]):
        inds_known = np.where(~np.isnan(M_train[:, id_item]))[0]

        if len(inds_known) > 0 :
            sims = np.array([cosinus(M_train, id_user, u) for u in inds_known])

            if len(inds_known) > k :
                ind = np.argsort(sims)
                inds_known = inds_known[ind][:k]
                sims = sims[ind][:k]

            if sum(abs(sims)) != 0 :
                rates = M_train[inds_known, id_item]

                mean_rates = np.nanmean(M_train[inds_known, :], axis=1)

                scores[id_item] = np.nanmean(M_train[id_user, :]) + np.sum(sims*(rates-mean_rates))/sum(abs(sims))
            else :
                scores[id_item] = np.nanmean(M_train[id_user, :])

        else :
            scores[id_item] = np.nanmean(M_train[id_user, :])

    return scores


##============================================
## knn.recommend(M_train, id_user, new=True, k=10)
##============================================
def recommend(M_train, id_user, new=True, k=10):
    scores = complete_a_user(M_train, id_user, k)
  
    if new:
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]
        rec_ind_in_unknown = np.argmax(scores[inds_unknown])
        return inds_unknown[rec_ind_in_unknown]

    else :
        return np.argmax(scores)



##============================================
## knn.complete
##============================================
def complete(M_train, k):
    # selectionne les utilisateurs qui ont note i
    M_completed = np.zeros(M_train.shape)
    for id_user in range(M_train.shape[0]):
        M_completed[id_user, :] = complete_a_user(M_train, id_user, k)
    return M_completed

