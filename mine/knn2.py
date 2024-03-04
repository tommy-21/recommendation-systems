
import numpy as np

##============================================
## cosinus_item(M_train, i1, i2)
##============================================
def cosinus_item(M_train, i1, i2):
    '''
    Compute the cosine similarity between two items given the train data
    
    M_train : training data as a matrix where rows are users and columns are items (movies).
    i1 : first item
    i2 : second item
    '''
    # users in common
    inds_user = np.where(np.sum(np.isnan(M_train[:, [i1, i2]]), axis=1) == 0)[0] # find indexes of users who rated both of the items
    
    # cosinus
    if len(inds_user) != 0: # if found
        n1 = M_train[inds_user, i1] # vectors formed by the values of the ratings of these users for the item 1
        n2 = M_train[inds_user, i2] # vectors formed by the values of the ratings of these users for the item 2
        cos = sum(n1*n2) / np.sqrt(sum(n1**2)) / np.sqrt(sum(n2**2)) # the cosine similarity formula is applied for these two vectors 
        return cos
    else: # if not, 
        return 0 # the cosine similarity is 0


##============================================
## complete_an_item(M_train, id_item, k)
##============================================
def complete_an_item(M_train, id_item, k):
    '''
    Estimate the values of a specific item's ratings using item-based collaborative filtering given training data
    
    M_train : training data as a matrix where rows are users and columns are items (movies).
    id_item : item for which ratings should be estimated
    k : number of nearest items
    '''
    # Implement the logic to estimate ratings for an item based on k nearest items
    # Similar to complete_a_user but for items
    scores = np.zeros(M_train.shape[0])
    for id_user in range(M_train.shape[0]): # for each user
        inds_known = np.where(~np.isnan(M_train[id_user, :]))[0] # retrieve movies already rated by this specific user
    
        if len(inds_known) > 0 : # if there are such movies (the user is not new)
            sims = np.array([cosinus_item(M_train, id_item, i) for i in inds_known]) # compute cosine similarities between our movie and each of them

            if len(inds_known) > k : # if there is more than k movies
                ind = np.argsort(sims) # sort the similarity vector
                inds_known = inds_known[ind][:k] # keep only the k movies with highest similarity with our movie (the famous k nearest neighbours)
                sims = sims[ind][:k] # and their similarity value

            if sum(abs(sims)) != 0 : # if there is at least one non-zero value in the similarities computed (k similarities at most)
                rates = M_train[id_user, inds_known]

                mean_rates = np.nanmean(M_train[:, inds_known], axis=0) # compute the mean of the ratings of our k nearests neighbours for each of them

                scores[id_user] = np.nanmean(M_train[:, id_item]) + sum(sims*(rates-mean_rates)) / sum(abs(sims)) # compute the estimated rating for the movie
            else : # if all of them are zeros,
                scores[id_user] = np.nanmean(M_train[:, id_item])

        else: # if not (maybe the user is new)
            scores[id_user] = np.nanmean(M_train[:, id_item]) # the estimated rating is the mean of the ratings of the movie
    
    return scores


##============================================
## item_recommend(M_train, id_user, k=10)
##============================================
def recommend(M_train, id_user, new=True, k=10):
    '''
    Recommend an item for a specified user, given the train data and a number of nearest items
    
    M_train : training data as a matrix where rows are users and columns are items.
    id_user : user for which recommendation should be done
    k: number of nearest items
    '''
    # Compute estimated ratings for all items for the user
    scores = complete(M_train, k)[id_user, :]

    if not new: # if the user is not new, that means he may already have rated some items. Thus we are going to
        inds_unknown = np.where(np.isnan(M_train[id_user, :]))[0] # first, find indexes of the items for which there is no rating yet by our user
        rec_item_in_unknown = np.argmax(scores[inds_unknown]) # then, find the item with the highest estimated rating among these items
        return inds_unknown[rec_item_in_unknown]
    
    else: # if the user is new, we are going to recommend the item with the highest estimated rating
        return np.argmax(scores)

##============================================
## item_complete(M_train, k)
##============================================
def complete(M_train, k):
    '''
    Complete a matrix of user-to-items ratings by estimating non-existent ratings using item-based collaborative filtering
    
    M_train : training data as a matrix where rows are users and columns are items. The matrix to be completed
    k : number of nearest items
    '''
    # Estimate ratings for all items using item-based collaborative filtering
    M_completed = np.zeros(M_train.shape)
    for id_item in range(M_train.shape[1]):
        M_completed[:, id_item] = complete_an_item(M_train, id_item, k)
    return M_completed
