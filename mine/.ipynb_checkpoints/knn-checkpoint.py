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
    '''
    Compute the cosine similarity between two user given the train data
    
    M_train : training data as a matrix where rows are users and columns are movies.
    u1 : first user
    u2 : second user
    '''
    # films en commun
    inds_movie = np.where(np.sum(np.isnan(M_train[[u1, u2], ]), axis=0) == 0)[0] # find indexes of movies that both of the users rated  
    
    # cosinus
    if len(inds_movie) != 0: # if found
        n1 = M_train[u1, inds_movie] # vectors formed by the values of the ratings of these movies for the user 1
        n2 = M_train[u2, inds_movie] # vectors formed by the values of the ratings of these movies for the user 2
        cos = sum(n1*n2) / np.sqrt(sum(n1**2)) / np.sqrt(sum(n2**2)) # the cosine similarity formula is applied for these two vectors 
        return cos

    else: # if not, 
        return 0 # the cosine similarity is 0



##============================================
## complete_a_user_knn(M_train, id_user, k)
##============================================
def complete_a_user(M_train, id_user, k):
    '''
    Estimate the values of a specific user's ratings using user-based collaborative filtering given training data
    
    M_train : training data as a matrix where rows are users and columns are movies. The matrix to be completed
    k : number of nearest neighbours
    '''
    scores = np.zeros(M_train.shape[1]) # initialize the rate vector to the null vector
    for id_item in range(M_train.shape[1]): # for each movie
        inds_known = np.where(~np.isnan(M_train[:, id_item]))[0] # retrieve users that already rated this specific movie

        if len(inds_known) > 0 : # if there are such users (at least one)
            sims = np.array([cosinus(M_train, id_user, u) for u in inds_known]) # compute cosine similarities between our user and each of them

            if len(inds_known) > k : # if there is more than k users 
                ind = np.argsort(sims) # sort the similarity vector
                inds_known = inds_known[ind][:k] # keep only the k users with highest similarity with our user (the famous k nearest neighbours)  
                sims = sims[ind][:k] # and their similarity value

            if sum(abs(sims)) != 0 : # if there is at least one non-zero value in the similarities computed (k similarities at most) 
                rates = M_train[inds_known, id_item] # take the ratings of our k nearests neighbours for the movie

                mean_rates = np.nanmean(M_train[inds_known, :], axis=1) # 

                scores[id_item] = np.nanmean(M_train[id_user, :]) + np.sum(sims*(rates-mean_rates))/sum(abs(sims))
            else : # if all of them are zeros, 
                scores[id_item] = np.nanmean(M_train[id_user, :]) # the estimated rating is the mean of our user's given ratings

        else : # if there is no rating yet for the movie
            scores[id_item] = np.nanmean(M_train[id_user, :]) # the estimated rating is the mean of our user's given ratings

    return scores


##============================================
## knn.recommend(M_train, id_user, new=True, k=10)
##============================================
def recommend(M_train, id_user, new=True, k=10):
    '''
    Recommend a movie for a specified user, given the train data and a number of neareast neighbours
    
    M_train : training data as a matrix where rows are users and columns are movies.
    id_user : user for which recommendation should be done
    new : new user or not 
    k: number of nearest neighbours
    '''
    scores = complete_a_user(M_train, id_user, k) # compute estimated ratings of the user each of the movies
  
    if !new: # if the user is not new, that means he may already have rated some movies. Thus we are going to 
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0] # first, find indexes of the movies for which there is no rating yet by our user 
        rec_ind_in_unknown = np.argmax(scores[inds_unknown]) # find the index of the movie with the best estimated rating in the list of the non-rated movies id
        return inds_unknown[rec_ind_in_unknown] # return the actual id of that movie

    else: # Just recommend the movies with the best estimated rating since the user is new and haven't rated any movies before
        return np.argmax(scores)



##============================================
## knn.complete
##============================================
def complete(M_train, k):
    '''
    Complete a matrix of user-to-items ratings by estimating non-existant ratings using user-based collaborative filtering
    
    M_train : training data as a matrix where rows are users and columns are movies. The matrix to be completed
    k : number of nearest neighbours
    '''
    # selectionne les utilisateurs qui ont note i
    M_completed = np.zeros(M_train.shape)
    for id_user in range(M_train.shape[0]):
        M_completed[id_user, :] = complete_a_user(M_train, id_user, k)
    return M_completed

