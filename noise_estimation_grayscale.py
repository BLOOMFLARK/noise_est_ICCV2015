import numpy as np
import cv2
import sys


######################################
UPPER_BOUND_LEVEL = 0.0005
UPPER_BOUND_FACTOR = 3.1

M1 = 5
M2 = 5
M = M1 * M2
EIGENVALUE_COUNT = 7
EIGENVALUE_DIFF_THRESHOLD = 49.0
LEVEL_STEP = 0.05
MIN_LEVEL = 0.06
MAX_CLIPPED_PIXEL_COUNT = 2 # = 0.1 * M
#######################################


def pca_noise_level_estimation(image):

    block_info = compute_block_info(image)
    block_info = block_info[block_info[:, 0].argsort()]
    #block_info = sortrows( block_info, [1] ); ############
    
    sum1, sum2, subset_size = compute_statistics(image, block_info)
    
    upper_bound = compute_upperbound(block_info)
    prev_variance = 0
    variance = upper_bound
    
    for _ in range(10):
        if(abs(prev_variance - variance) < 1e-6):
            break
        prev_variance = variance
        variance = get_next_estimate(sum1, sum2, subset_size, variance, upper_bound)
    return variance


def clamp(x, a, b):
    y = x
    if x < a:
        y = a
        
    if x > b:
        y = b
    return y

# START STEP STOP!
def compute_block_info(image):
    # матрица (650px * 700px, 3) -матрицу нулей
    block_info = np.zeros((image.shape[0] * image.shape[1], 3))
        
    block_count = 0
    for y in range(image.shape[0] - M2):
        for x in range(image.shape[1] - M1):

            sum1 = 0.0;
            sum2 = 0.0;
            clipped_pixel_count = 0;
            for by in range(y, y + M2 - 1):
                for bx in range(x, x + M1 - 1):
                    #  значение пикселя это у данной картинки!
                    val = image[by, bx]
        
                    sum1 = sum1 + val
                    sum2 = sum2 + val * val
                        
                    if val == 0 or val == 255:
                        clipped_pixel_count = clipped_pixel_count + 1
                
            if clipped_pixel_count <= MAX_CLIPPED_PIXEL_COUNT:
    
                block_count = block_count + 1;
                    
                block_info[block_count, 0] = (sum2 - sum1 * sum1 / M) / M
                block_info[block_count, 1] = x
                block_info[block_count, 2] = y
    # block_count - 1
    block_info[block_count:image.shape[0] * image.shape[1], :] = []
    return block_info


def compute_statistics(image, block_info):

    sum1 = []
    sum2 = []
    subset_size = []
        
    subset_count = 0
    
    for p in range(0, MIN_LEVEL, -LEVEL_STEP):

        q = 0
        if p - LEVEL_STEP > MIN_LEVEL:
            q = p - LEVEL_STEP
            
        max_index = block_info.shape[0] - 1
        beg_index = clamp(round(q * max_index) + 1, 1, block_info.shape[0])
        end_index = clamp(round(p * max_index) + 1, 1, block_info.shape[0])

        curr_sum1 = np.zeros((M, 1)) #######
        curr_sum2 = np.zeros((M, M)) ########

        for k in range(beg_index,  end_index):
            curr_x = block_info[k, 1]
            curr_y = block_info[k, 2]
    
            #block = reshape(image(curr_y : curr_y + M2 - 1, curr_x : curr_x + M1 - 1), M, 1)
            curr_sum1 = curr_sum1 + block
            curr_sum2 = curr_sum2 + block * block


        subset_count = subset_count + 1
            
        sum1[:, :, subset_count - 1] = curr_sum1
        sum2[:, :, subset_count - 1] = curr_sum2
        subset_size[subset_count - 1] = end_index - beg_index
    for i in range(len(subset_size), 2, -1): #####?

        sum1[:, :, i-1] = sum1[:, :, i-1] + sum1[:, :, i]
        sum2[:, :, i-1] = sum2[:, :, i-1] + sum2[:, :, i]
        subset_size[i-1] = subset_size[i-1] + subset_size[i]

    return sum1, sum2, subset_size


def compute_upperbound(block_info):
    max_index = block_info.shape[0] - 1
    index = clamp(round(UPPER_BOUND_LEVEL * max_index) + 1, 1, block_info.shape[0])
    upper_bound = UPPER_BOUND_FACTOR * block_info[index, 0]

    return upper_bound


def apply_pca(sum1, sum2, subset_size):
        
    mean = sum1 / subset_size
    cov_matrix = sum2 / subset_size - mean * mean
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues.sort() 
    #eigen_value = sort( eig(cov_matrix) ) #############

    return eigenvalues

def get_next_estimate(sum1, sum2, subset_size, prev_estimate, upper_bound):
        
    variance = 0;
    for i in range(len(subset_size)):

        eigen_value = apply_pca(sum1[:, :, i], sum2[:, :, i], subset_size[i])

        variance = eigen_value[0]

        if variance < 1e-6:
            break

        diff = eigen_value[EIGENVALUE_COUNT - 1] - eigen_value[0]
        diff_threshold = EIGENVALUE_DIFF_THRESHOLD * prev_estimate / subset_size[i]^0.5

        if diff < diff_threshold and variance < upper_bound:
            break

    return variance


if __name__ == '__main__':
	# grayscale mode
    image = cv2.imread('data/cameraman-std=5.pgm', 0)
    image = image.astype(np.float)
    #if image.shape[2] != 1:
    #    sys.exit( 'Only grayscale 2D images can be processed.' )

    sigma = np.sqrt(pca_noise_level_estimation(image))
    print( 'expected noise standard deviation = 5.00\n' )
    print( 'computed noise standard deviation = %.2f\n', sigma )
