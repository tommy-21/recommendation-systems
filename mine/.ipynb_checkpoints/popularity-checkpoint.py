##============================================
##============================================
## Populaire
##============================================
##============================================
# recommande le produit le plus populaire

# * popularity.recommend(M_train, id_user, new) : recommande un film
# * popularity.complete(M_train) : complete la matrice

import numpy as np


##============================================
# popularity.recommend(M_train, id_user, new)
##============================================
def recommend(M_train, id_user, new=True):
    '''
    M_train : training data as a matrix where rows are users and columns are movies.
    id_user : user for which recommendation should be done
    '''
    scores = np.nanmean(M_train, axis=0) # compute the average of the rates given by each user
    if new:
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0] # find indexes of  movies where there is no rating by our user 
        rec_ind_in_unknown = np.argmax(scores[inds_unknown]) # index of the movies with the best average rating in the list of the unknown id
        return inds_unknown[rec_ind_in_unknown] # return the actual id of that movie
    else:
        return np.argmax(scores) # return the id of the movies with the best average rating 


##============================================
# popularity.complete(M_train)
##============================================
def complete(M_train):
  scores = np.nanmean(M_train, axis=0)
  scores[np.isnan(scores)] = 0
  to_complete = np.ones((M_train.shape[0], 1)) @ scores.reshape((1, -1))
  M_completed = M_train.copy()
  M_completed[np.isnan(M_train)] = to_complete[np.isnan(M_train)]
  return M_completed



