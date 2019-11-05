import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    
    matches = np.zeros((features1.shape[0], 2)).astype(np.int32)
    confidences = np.zeros(features1.shape[0]).astype(np.int32)
    
    match_cnt = 0
    for i in range(features1.shape[0]):
        distances = np.sum((features1[i] - features2) ** 2, axis=1)
        closest_match = np.argsort(distances)[0]
        distances = np.sort(distances)
        
        # empirical value 0.68
        if (distances[0] / distances[1] < 0.68):
            matches[match_cnt] = i, closest_match
            confidences[match_cnt] = distances[0]
            match_cnt += 1


    
    
    
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
