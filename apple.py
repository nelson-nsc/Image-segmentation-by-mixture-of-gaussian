import os 
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, transpose, matmul
from numpy.linalg import det, inv
import glob

def load_data(imgs, gts):
    '''load rgb images and ground truths; and
    calculate the prior for the likelihood
    '''
    data_true = []
    data_false = []
    for img, gt in zip(imgs, gts):
        # load images
        image = plt.imread(img)
        image = image / 255
        # ground truth
        gtruth = (plt.imread(gt)[:,:,2]).astype(int) # only needs one channel
        for i in range(gtruth.shape[0]):
            for j in range(gtruth.shape[1]):
                if gtruth[i, j] == 1.0:
                    data_true.append(image[i,j])
                else:
                    data_false.append(image[i,j])
    # reshape the data to d x n
    data_true = np.asarray(data_true).T
    data_false = np.asarray(data_false).T
    # calculate the prior
    pixels = data_false.shape[1] + data_true.shape[1]
    prior = [data_true.shape[1]/ pixels, data_false.shape[1]/ pixels]
    return data_true, data_false, prior
                                                              
                                                                                                       
def likelihood(data, model_estimate):
        """Calculate likelihood of seeing each pixel individually with our model estimate
        """
        nDim, nData = data.shape
        logs = np.zeros((model_estimate['k'], nData))

        for current_c in range(model_estimate['k']):
            current_weight = model_estimate['weight'][current_c]
            current_mean = model_estimate['mean'][:, current_c]
            current_cov = model_estimate['cov'][:, :, current_c]
            logs[current_c, :] = current_weight * \
            gauss_probability(data, current_mean, current_cov)

        # We only sum over the k's here as we want to keep the likelihood for each individual
        # pixel to calculate the posterior If we wanted the likelihood of all data, we would
        # multiply each value in likelihood together (not add, as not log!).
        likelihood = np.sum(logs, axis=0)

        return likelihood


def log_likelihood(data, model_estimate):
    """Calculate log likelihood of seeing the full image (all pixels) with our model estimate
    """

    n_data = data.shape[1]
    logs = np.zeros((model_estimate['k'], n_data))

    for current_c in range(model_estimate['k']):
        current_weight = model_estimate['weight'][current_c]
        current_mean = model_estimate['mean'][:, current_c]
        current_cov = model_estimate['cov'][:, :, current_c]
        logs[current_c, :] = current_weight * \
        gauss_probability(data, current_mean, current_cov)

    # We want the likelihood of seeing all these pixels, and hence we add over k and all pixels,
    # not just k. Add not multiply due to log likelihood
    log_likelihood = np.sum(np.log(np.sum(logs, axis=0)))

    return log_likelihood

def gauss_probability(data, mean, cov):
    """Calculate probability of observing data points under a gauss. distribution ~ [mean, cov]
    """
    data = data - mean[:, None]
    sig = np.linalg.inv(cov)
    power = (data.T.dot(sig) * data.T).sum(1)
    prob = (np.linalg.det(2 * np.pi * cov)**-0.5) * np.exp(-0.5 * power)
    # print(sig.shape, data.shape, power.shape)
    # print(prob.shape)
    return prob



def fit_gaussian(data, k):
    nDim, nData = data.shape
    mean = np.mean(data, axis=1)
    #cov = 1 / nData * (data - mean.reshape(nDim,1)) @ (data - mean.reshape(nDim,1)).T
    cov = 1 / nData * (data - mean[:, None]) @ (data - mean[:, None]).T
    postHidden = np.zeros(shape=(k, nData))

    mixGaussEst = dict()
    mixGaussEst['d'] = nDim
    mixGaussEst['k'] = k
    # initialise parameters
    mixGaussEst['weight'] = (1 / k) * np.ones(shape=(k))
    mixGaussEst['mean'] = 2 * np.random.randn(nDim, k)
    mixGaussEst['cov'] = (1 + 0.2 * np.random.normal(size=(k)))[None, None] * cov[:, :, None]

    logLike = log_likelihood(data, mixGaussEst)
    print('Log Likelihood Iter 0 : {:4.3f}\n'.format(logLike))

    nIter = 10

    for cIter in range(nIter):
        ### Expectation step ###

        for i in range(k):
            weight = mixGaussEst['weight'][i]
            mean = mixGaussEst['mean'][:,i]
            cov = mixGaussEst['cov'][:,:,i]
            postHidden[i,:] = weight * gauss_probability(data, mean, cov)
            postHidden /= np.sum(postHidden, axis=0) #+ sys.float_info.min # aviod singularity

        respon_k = np.sum(postHidden, axis=1)
        
        ### Maximization Step ###

        # for each constituent Gaussian
        for i in range(k):
            res = postHidden[i,:]
            mixGaussEst['weight'][i] = np.sum(res) / np.sum(postHidden)
            mixGaussEst['mean'][:,i] = (res*data).sum(1)/np.sum(res)
            mixGaussEst['cov'][:,:,i] = \
            ((((data - mixGaussEst['mean'][:, i][:, None])*res[None,:])@(data - mixGaussEst['mean'][:, i][:, None]).T)/respon_k[i])
            
        # calculate the log likelihood
        logLike = log_likelihood(data, mixGaussEst)
        print('Log Likelihood After Iter {} : {:4.3f}\n'.format(cIter, logLike))

    return mixGaussEst

def posterior(im_path, prior, data_true, data_false):
    # let's define priors for whether the pixel is skin or non skin
    prior_true, prior_false = prior
    # now run through the pixels in the image and classify them as being skin or
    im = plt.imread(im_path)
    imY, imX, imZ = im.shape
    like_true = likelihood(im.reshape(imY*imX,imZ).T, data_true).reshape(imY,imX)
    like_false = likelihood(im.reshape(imY*imX,imZ).T, data_false).reshape(imY,imX)
    # print(im.shape)
    posterior_true = like_true * prior_true / (like_true * prior_true + like_false * prior_false)
    # posterior_true_binary = (posterior_true > self._threshold).astype(int)
    plt.imshow(posterior_true)
    return posterior_true





if __name__ == "__main__":
    img_train = [
        './apples/Apples_by_kightp_Pat_Knight_flickr.jpg',
        './apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg',
        './apples/bobbing-for-apples.jpg'
    ]

    train_gt = [
        './apples/Apples_by_kightp_Pat_Knight_flickr.png',
        './apples/ApplesAndPears_by_srqpix_ClydeRobinson.png',
        './apples/bobbing-for-apples.png'
    ]

    data_true, data_false, prior = load_data(img_train, train_gt)
    mixGassEst_true = fit_gaussian(data_true, 3)
    mixGassEst_false = fit_gaussian(data_false, 3)
    post = posterior(img_train[0], prior, mixGassEst_true, mixGassEst_false)
    #plt.imshow()
    #print(data_true[:,0],data_true.shape)
