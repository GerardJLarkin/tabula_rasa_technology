# https://www.geeksforgeeks.org/computer-vision/fast-fourier-transform-in-image-processing
import numpy as np

# bin value of 64 as original image shape was 64x64. normalised patom values are less than this
# but bins with zero values are ignored
def radial_power_spectrum(array):
    # compute the 2d fourier transform and and recenter to make symmetric
    f = np.fft.fftshift(np.fft.fft2(array.astype(np.float32)))
    # calculate the power spectrum
    mag = np.abs(f)**2

    # set bins equal to the shape of the original array (arbitray)
    nbins=64
    # build radius map, centred at spectrum centre
    h, w = array.shape
    cy, cx = (h-1)/2.0, (w-1)/2.0
    # create open grids to match x, y coordinate number
    y, x = np.ogrid[:h, :w]
    # compute radial distance for each coordinate from the spectrum centre
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    # bin radial distances (normalised between 0 and 1)
    r_norm = r / r.max()
    # create bins 
    bins = np.linspace(0, 1.0, nbins+1)
    # flatten the normalised radial distance array and assign each pixel to a bin
    idx = np.digitize(r_norm.ravel(), bins) - 1
    # sum the values per bin
    sums = np.bincount(idx, weights=mag.ravel(), minlength=nbins)
    # count how many pixels fall into each bin
    counts = np.bincount(idx, minlength=nbins)
    # compute radial average
    radial = np.divide(sums, np.maximum(counts, 1), dtype=np.float32)

    # https://stackoverflow.com/questions/49538185/purpose-of-numpy-log1p
    radial = np.log1p(radial)
    # l1 normalise so all bins sum to 1
    radial /= radial.sum() + 1e-8

    return radial

def rps_distance(array1, array2):
    a = radial_power_spectrum(array1)
    b = radial_power_spectrum(array2)
    
    # calculate the chi-squared distance between corresponding bins from each patom
    return 0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-8))

# colour distance 
def colour_histogram(col):
    
    # flatten array
    col = np.asarray(col).reshape(-1)

    # create number of bins (arbitrary)
    bins=32
    
    # put colours into bins
    h, _ = np.histogram(col, bins=bins, range=(1, 255))
    h = h.astype(np.float32)
    
    # l1 normalise so all bins sum to 1
    h /= (h.sum() + 1e-8)

    return h

# colour distance 
def col_distance(arr_col1, arr_col2):
    a = colour_histogram(arr_col1)
    b = colour_histogram(arr_col2)
    
    # calculate the chi-squared distance between corresponding bins from each patom
    return 0.5 * np.sum((a - b)**2 / (a + b + 1e-12), dtype=np.float32)

def compare(array1, array2):
    
    id1 = array1[0,0]; id2 = array2[0,0]
    coords1 = array1[2:,:2]
    coords2 = array2[2:,:2]
    cols1 = array1[2:,2]
    cols2 = array2[2:,2]
    coordinate_dist = rps_distance(coords1, coords2)
    colour_dist = col_distance(cols1, cols2)

    score = coordinate_dist + colour_dist

    return [id1, id2, score]