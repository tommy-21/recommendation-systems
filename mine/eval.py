##============================================
##============================================
## fonctions utiles à l'évaluation des algorithmes
##============================================
##============================================

# * get_train_val(M, prop=0_8)
# * RMSE(M_completed, M_star)
# * quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)



import numpy as np
from time import time
import pandas as pd

##============================================
## get_train_val(M, prop=0_8)
##============================================
def get_train_val(M, prop=0.8):
  n, m = M.shape
  M_train = np.nan * np.ones((n, m), dtype=float)
  M_validation = M.copy()
  
  for id_user in range(n):
    inds_star = np.where(~np.isnan(M[id_user, :]))[0]
    if len(inds_star)==1:
      inds = inds_star
    else:
      inds = np.random.choice(inds_star, max(1, int(prop*len(inds_star))), replace=False)
    M_train[id_user, inds] = M[id_user, inds]
    M_validation[id_user, inds] = np.nan
  
  return (M_train, M_validation)




##============================================
## RMSE(M_completed, M_star)
##============================================
def RMSE(M_completed, M_star):
  inds = ~np.isnan(M_star)
  return np.sqrt(np.mean((M_completed[inds] - M_star[inds])**2))



##============================================
## MAE(M_completed, M_star)
##============================================
def MAE(M_completed, M_star):
  inds = ~np.isnan(M_star)
  return np.mean(np.abs(M_completed[inds] - M_star[inds]))



##============================================
## precision_for_n_rec(M_completed, M_star)
##============================================
def precision_for_n_rec(M_completed, M_star, n=5, th=2.5):
  n_user, _ = M_star.shape
  n_total = n_user
  precision = 0
  for id_user in range(n_user):
    inds_star = np.where((~np.isnan(M_star[id_user, :])) & (M_star[id_user, :] > th))[0]
    candidate_items = np.where(~np.isnan(M_star[id_user, :]))[0]
    if len(inds_star) > 0:
      inds_rec = np.argsort(M_completed[id_user, :])
      inds_rec = inds_rec[np.isin(inds_rec, candidate_items)]
      inds_rec = inds_rec[-n:]
      precision += len(set(inds_star) & set(inds_rec)) / n
    else:
      n_total -= 1
  return precision / n_total


##============================================
## recall_for_n_rec(M_completed, M_star)
##============================================
def recall_for_n_rec(M_completed, M_star, n=5, th=2.5):
  n_user, _ = M_star.shape
  n_total = n_user
  recall = 0
  for id_user in range(n_user):
    inds_star = np.where((~np.isnan(M_star[id_user, :])) & (M_star[id_user, :] > th))[0]
    candidate_items = np.where(~np.isnan(M_star[id_user, :]))[0]

    if len(inds_star) > 0:
      inds_rec = np.argsort(M_completed[id_user, :])
      inds_rec = inds_rec[np.isin(inds_rec, candidate_items)]
      inds_rec = inds_rec[-n:]
      recall += len(set(inds_star) & set(inds_rec)) / len(inds_star)
    else:
      n_total -= 1
  return recall / n_total



##============================================
## quantitative_comparison(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)
##============================================

def quantitative_comparison(scoring_fn, M_star, recommenders, prop=0.8, nrep=10):
  scores = np.zeros((len(recommenders), nrep))
  scores_train = np.zeros((len(recommenders), nrep))
  computation_time = np.zeros((len(recommenders), nrep))
  for id_rep in range(nrep):
    M_train, M_validation = get_train_val(M_star, prop)
    for id_rec in range(len(recommenders)):
      ptm = time()
      M_completed = recommenders[id_rec]['fn'](M_train)
      computation_time[id_rec, id_rep] = (time() - ptm)
      scores[id_rec, id_rep] = scoring_fn(M_completed, M_validation)
      scores_train[id_rec, id_rep] = scoring_fn(M_completed, M_train)


  return pd.DataFrame({
          'recommender': [rec['label'] for rec in recommenders],
          'validation score': np.mean(scores, axis=1),
          'training score': np.mean(scores_train, axis=1),
          'computation time': np.mean(computation_time, axis=1)
          })


##============================================
## quantitative_comparison_v2(scoring_fn, M_star, recommmenders, prop=0_8, nrep=10)
##============================================
def quantitative_comparison_v2(scoring_fn, M_star, recommenders, prop=0.8, nrep=10):
  scores = np.zeros((len(recommenders), nrep))
  scores_train = np.zeros((len(recommenders), nrep))
  scores_mae = np.zeros((len(recommenders), nrep))
  computation_time = np.zeros((len(recommenders), nrep))
  precision_scores = np.zeros((len(recommenders), nrep))
  recall_scores = np.zeros((len(recommenders), nrep))
  for id_rep in range(nrep):
    M_train, M_validation = get_train_val(M_star, prop)
    for id_rec in range(len(recommenders)):
      ptm = time()
      M_completed = recommenders[id_rec]['fn'](M_train)
      computation_time[id_rec, id_rep] = (time() - ptm)
      scores[id_rec, id_rep] = scoring_fn(M_completed, M_validation)
      scores_train[id_rec, id_rep] = scoring_fn(M_completed, M_train)
      scores_mae[id_rec, id_rep] = MAE(M_completed, M_validation)
      precision_scores[id_rec, id_rep] = precision_for_n_rec(M_completed, M_validation)
      recall_scores[id_rec, id_rep] = recall_for_n_rec(M_completed, M_validation)


  return pd.DataFrame({
          'recommender': [rec['label'] for rec in recommenders],
          'RMSE validation': np.mean(scores, axis=1),
          'RMSE training': np.mean(scores_train, axis=1),
          'MAE validation': np.mean(scores_mae, axis=1),
          'Precision': np.mean(precision_scores, axis=1),
          'Recall': np.mean(recall_scores, axis=1),
          'computation time': np.mean(computation_time, axis=1)
          })
