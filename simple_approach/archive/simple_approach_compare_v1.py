# import warnings
# warnings.filterwarnings("ignore")
import numpy as np

def compare(a: np.ndarray, b: np.ndarray):
    # p0: [patom_id, x_vals, y_vals, norm_x, norm_y, colour, sequence_id]
    # p0: [patom_id, x_vals, y_vals, norm_x, norm_y, colour, sequence_id]
    m = a.shape[0]
    n = b.shape[0]
    x1y1 = np.asarray(a[:,3:5],dtype=float); x2y2 = np.asarray(b[:,3:5],dtype=float)
    pos_diffs = x1y1[:, None, :] - x2y2[None, :, :]   # shape (n1, n2, 2)
    sq = pos_diffs**2
    sq_sum = np.sum(sq, axis=2).reshape(-1)
    position_similarity = np.sqrt(np.sum(sq_sum)) / (m * n)# closer to zero more similar
    # get the absolute distance between the colours at each pixel location in corresponding segents
    col1 = np.asarray(a[:,5],dtype=float); col2 = np.asarray(b[:,5],dtype=float)
    # broadcast to a full (n1, n2) matrix of differences
    col_diffs = col1[:, None] - col2[None, :]   # shape (n1, n2)
    colour_similarity = np.abs(col_diffs).sum() / (m * n) # closer to zero more similar
    # compare number of pixels between patoms
    pixel_fill_similarity = 1.0 if abs(m - n) / ((m + n) / 2) > 1.0 else abs(m - n) / ((m + n) / 2)  # closer to zero more similar
    # get patom ids for patoms being compared
    one_id = a[0,0]; two_id = b[0,0]

    # obtain final similarity score by weighting the similarity values, with position and fill being the top 2 ranked (40%)
    # followed by colour (20%)
    total_score = position_similarity + colour_similarity + pixel_fill_similarity
    if total_score == 0:
        similarity_score = 1/3
    else:
        similarity_score = (position_similarity * 0.4)/total_score + (colour_similarity * 0.2)/total_score + (pixel_fill_similarity * 0.4)/total_score
    
    return [one_id, two_id, similarity_score]
    