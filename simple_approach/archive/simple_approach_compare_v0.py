# import warnings
# warnings.filterwarnings("ignore")
import numpy as np

## need to reconcile this

def compare(patom_one, patom_two):
    # 0,1,2,3,4,5,6,7,8,9,10,11,12,13: 
    # p0 norm_x, norm_y - 3, 4, p0 color - 5
    # p1 norm_x, norm_y - 10, 11 p1 color - 12
    # p0: [patom_id, x_vals, y_vals, norm_x, norm_y, colour, sequence_id]
    # p0: [patom_id, x_vals, y_vals, norm_x, norm_y, colour, sequence_id]
    m = patom_one.shape[0]
    n = patom_two.shape[0]

    one_repeat = np.repeat(patom_one, repeats=n, axis=0)  # repeat rows of new patom segment
    two_tile = np.tile(patom_two, (m, 1))  # tile rows of of ref array segment 

    # merge using hstack (concatenation along columns)
    cartesian = np.hstack((one_repeat, two_tile))
    
    # get length of segment after cartesian join between the new patom and the reference patom
    cart_segment_len = m * n
    # get the euclidean distance between all points in corresponding segments
    position_similarity = np.sqrt(sum(cartesian[:,10] - cartesian[:,3])**2 + 
        sum(cartesian[:,11] - cartesian[:,4])**2) / cart_segment_len # closer to zero more similar
    # get the absolute distance between the colours at each pixel location in corresponding segents
    colour_similarity = np.sum(np.absolute(cartesian[:,5] - cartesian[:,12])) / cart_segment_len # closer to zero more similar
    # compare number of pixels between patoms
    pixel_fill_similarity = np.absolute(m - n) / ((m + n) / 2) # closer to zero more similar
    
    # get patom ids for patoms being compared
    one_id = patom_one[0,0]; two_id = patom_two[0,0]

    # obtain final similarity score by weighting the similarity values, with position and fill being the top 2 ranked (40%)
    # followed by colour (20%)

    similarity_score = (position_similarity * 0.4) + (colour_similarity * 0.2) + (pixel_fill_similarity * 0.4)

    patom_similarity = [one_id, two_id, similarity_score]
    
    return patom_similarity