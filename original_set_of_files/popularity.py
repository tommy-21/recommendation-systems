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
  scores = np.nanmean(M_train, axis=1)
  if new:
    inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0]
    rec_ind_in_unknown = np.argmax(scores[inds_unknown])
    return inds_unknown[rec_ind_in_unknown]
  else:
    return np.argmax(scores)


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



