def HarrisDetector(img,k = 0.04):
    '''
    Args:
    
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector

    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
   '''
    pass 


def SuppressNonMax(Rvals, numPts):
    '''
    Args:
    
    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:

     x: A numpy array of shape (N,) containing x-coordinates of interest points
     y: A numpy array of shape (N,) containing y-coordinates of interest points
     confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
   '''
    pass 


