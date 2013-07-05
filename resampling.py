#Author: Timothy Anderton

import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse import lil_matrix
import scipy.stats
import time

def centers_to_bins(coord_centers):
    if len(coord_centers) == 0:
        return np.zeros(0)
    bins = np.zeros(len(coord_centers) + 1)
    bins[1:-1] = 0.5*(coord_centers[1:] + coord_centers[:-1])
    bins[0] = coord_centers[0] - (bins[1]-coord_centers[0])
    bins[-1] = coord_centers[-1] + 0.5*(coord_centers[-1] - coord_centers[-2])
    return bins

n_delts = 1024
z_scores = np.linspace(-6, 6, n_delts)
cdf_vals = scipy.stats.norm.cdf(z_scores)
min_z = z_scores[0]
max_z = z_scores[-1]
z_delta = (z_scores[1]-z_scores[0])
#gaussian_cdf_interpolator = interp1d(z_scores, scipy.stats.norm.cdf(z_scores))

def approximate_gaussian_cdf(zscore):
    if zscore > max_z-z_delta-1e-5:
        return 1.0
    elif zscore < min_z:
        return 0
    idx_val = (zscore-min_z)/z_delta
    base_idx = int(idx_val)
    alpha = idx_val-base_idx
    return cdf_vals[base_idx]*(1-alpha) + cdf_vals[base_idx+1]*alpha
    #return float(gaussian_cdf_interpolator(zscore))

class Density:
    
    def integrate(self, index, lb, ub):
        lower_val = self.get_density_integral(index, lb)
        upper_val = self.get_density_integral(index, ub)
        return upper_val-lower_val


class Gaussian_Density(Density):
    def __init__(self, coord_centers, widths, max_sigma = 5.0):
        self.centers = coord_centers
        self.widths = widths
        self.max_sigma = max_sigma

    def get_density_integral(self, index, coord):
        zscore = (coord-self.centers[index])/self.widths[index]
        return approximate_gaussian_cdf(zscore)

    def get_coordinate_density_range(self, index):
        lb = self.centers[index] - self.max_sigma*self.widths[index]
        ub = self.centers[index] + self.max_sigma*self.widths[index]
        return lb, ub

class Linear_Interpolated_Density(Density):
    
    def __init__(self, coord_centers, center_values):
        self.centers = coord_centers
        self.values  = center_values
        self.bins = centers_to_bins(coord_centers)
        
    def get_density_integral(self, index, coord):
        if index == 0:
            last_val = self.values[0]
        else:
            last_val = self.values[index-1]
        if index == len(self.centers)-1:
            next_val = self.values[-1]
        else:
            next_val = self.values[index+1]
        
        c_val = self.values[index]
        
        left_max_int = 0.5*(last_val+c_val)
        right_max_int = left_max_int + 0.5*(next_val+c_val)
        
        clb = self.bins[index] #current bin lower bound
        cub = self.bins[index+1] #current bin upper bound
        
        if coord < self.centers[index]:
            if coord <= clb:
                return 0
            else:
                alpha = (coord-clb)/(self.centers[index]-clb)
                return last_val*alpha + 0.5*(c_val-last_val)*alpha**2
        elif coord >= cub:
            return right_max_int
        else:
            alpha = (coord-self.centers[index])/(cub-self.centers[index])
            return left_max_int + c_val*alpha + 0.5*(next_val-c_val)*alpha**2
    
    def get_coordinate_density_range(self, index):
        return self.bins[index], self.bins[index+1]

class Box_Density(Density):
    def __init__(self, coord_centers):
        self.centers = coord_centers
        self.bins = centers_to_bins(coord_centers)
        
    def get_density_integral(self, index, coord):
        clb, cub = self.bins[index], self.bins[index+1]
        if (clb > coord):
            return 0.0
        elif (cub < coord):
            return 1.0
        else:
            return (coord-clb)/float(cub-clb)
        
    def get_coordinate_density_range(self, index):
        return self.bins[index], self.bins[index+1]
    
def map_indicies(input_coordinates, target_coordinates):
    """assign indexes to the input coordinates which place them on to the indexing of the target coordinates
    target_coordinates must be a monotonically increasing sequence
    interior to the target coordinate bounds interpolation is done and external to the target coordinate bounds
    linear extrapolation is done.
    
    examples:
    >>> input = np.linspace(-1, 1.5, 3)
    >>> input
    array([-1.  ,  0.25,  1.5 ])
    >>> target = np.arange(4)
    >>> target
    array([0, 1, 2, 3])
    >>> map_indicies(input, target)
    array([-1.  ,  0.25,  1.5 ])
    
    >>> target = np.arange(2, 6)
    >>> target
    array([2, 3, 4, 5])
    >>> map_indicies(input, target)
    array([-3.  , -1.75, -0.5 ])
    """
    target_index_interpolator = interp1d(target_coordinates, np.arange(len(target_coordinates)), bounds_error = False)
    output_indicies = target_index_interpolator(input_coordinates)
    target_min = target_coordinates[0]
    target_max = target_coordinates[-1]
    low_point_idxs = np.where(np.isnan(output_indicies)*(input_coordinates <= target_min))[0]
    upper_point_idxs = np.where(np.isnan(output_indicies)*(input_coordinates >= target_max))[0]
    start_slope = 1.0/(target_coordinates[1]-target_coordinates[0])
    end_slope = 1.0/(target_coordinates[-1]-target_coordinates[-2])
    for lpi in low_point_idxs:
        output_indicies[lpi] = start_slope*(input_coordinates[lpi]-target_min)
    for upi in upper_point_idxs:
        output_indicies[upi] = end_slope*(input_coordinates[upi]-target_max) + len(target_coordinates) -1
    return output_indicies
    
def get_resampling_matrix(input_centers, output_centers, pixel_density = None, preserve_normalization = False, upweight_ends = True):
    """takes a set of input and output coordinates and generates a matrix to transform from the input coords to the output coords
    both the input_centers and output_centers must be monotonically increasing sequences
    
    pixel_density: the density associated to each input pixel, if None a flat box density is assumed.
    preserve_normalization: If True the matrix rows are rescaled 
        to ensure that the matrix times an input vector of ones has as a result an output vector of ones.
    upweight_ends: if True the first few and last few rows of the resampling matrix with non-zero row sum
        are rescaled to match the row sum of the row immediately interior to them.
        The number of rows that are rescaled is determined by the input pixel_density coordinate ranges.
        this can adjust for edge effects caused by fewer input pixels per output pixel at the edges.
    """
    if pixel_density == None:
        pixel_density = Box_Density(input_centers) #default to a box density for input pixels
    output_bins = centers_to_bins(output_centers)
    input_bins = centers_to_bins(input_centers)
    n_in, n_out = len(input_centers), len(output_centers)
    #TODO: change the building matrix type to coo_matrix in order to make the matrix building more efficient
    #trans = lil_matrix((n_out, n_in))
    all_row_idxs = []
    all_col_idxs = []
    all_matrix_vals = []
    central_index_vals = np.array(np.around(map_indicies(output_centers, input_centers)), dtype = int)
    available_idxs = np.where(central_index_vals >= 0, central_index_vals, np.zeros(n_out))
    available_idxs = np.where(central_index_vals < n_in, available_idxs, np.ones(n_out)*(n_in-1))
    for out_idx in xrange(n_out):
        central_in_idx = central_index_vals[out_idx]
        c_output_lb, c_output_ub = output_bins[out_idx], output_bins[out_idx+1]
        c_input_lb, temp = pixel_density.get_coordinate_density_range(available_idxs[max(0, out_idx-1)])
        temp, c_input_ub = pixel_density.get_coordinate_density_range(available_idxs[min(n_out-1, out_idx+1)])
        #fill in the diagonal
        if 0 <= central_in_idx < n_in:
            c_in_idx = central_in_idx
            all_row_idxs.append(out_idx)
            all_col_idxs.append(c_in_idx)
            all_matrix_vals.append(pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub))
            #trans[out_idx, c_in_idx] = pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub)
        else:
            sharp_ub = input_bins[available_idxs[min(n_out-1, out_idx+1)]+1]
            sharp_lb = input_bins[available_idxs[max(0, out_idx-1)]]
            if (c_output_lb > sharp_ub) or (c_output_ub < sharp_lb):
                #if there is no overlap of the last input pixel and the 
                continue
        #fill in the above diagonal terms
        idx_delta = 1
        while True:
            c_in_idx = central_in_idx + idx_delta
            if 0 <= c_in_idx:
                if c_in_idx < n_in:
                    all_row_idxs.append(out_idx)
                    all_col_idxs.append(c_in_idx)
                    all_matrix_vals.append(pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub))
                    #trans[out_idx, c_in_idx] = pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub)
                    if input_centers[c_in_idx] > c_input_ub:
                        break
                else: 
                    break
            idx_delta += 1
        #fill in the below diagonal terms
        idx_delta = 1
        while True:
            c_in_idx = central_in_idx - idx_delta
            if c_in_idx < n_in:
                if c_in_idx >= 0:
                    all_row_idxs.append(out_idx)
                    all_col_idxs.append(c_in_idx)
                    all_matrix_vals.append(pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub))
                    #trans[out_idx, c_in_idx] = pixel_density.integrate(c_in_idx, c_output_lb, c_output_ub)
                    if input_centers[c_in_idx] < c_input_lb:
                        break
                else: 
                    break
            idx_delta += 1
    row_col_idxs = np.array([all_row_idxs, all_col_idxs])
    all_matrix_vals = np.array(all_matrix_vals)
    trans = scipy.sparse.coo_matrix((all_matrix_vals, row_col_idxs), (n_out, n_in))
    #trans = trans.tocsr()
    row_sum = trans*np.ones(n_in)
    row_rescale = np.ones(n_out)
    if upweight_ends:
        row_rescale = np.ones(n_out)
        nz_pts = np.where(row_sum)[0]
        if len(nz_pts) > 3:
            first_nz, last_nz = nz_pts[0], nz_pts[-1] #indicies of the first and last non-zero row sums
            nz_delta = 1
            center_delta = np.abs(output_centers[first_nz + nz_delta] - output_centers[first_nz])
            cclb, ccub = pixel_density.get_coordinate_density_range(available_idxs[first_nz])
            width_delta = ccub-cclb
            while center_delta < width_delta:
                nz_delta += 1
                try:
                    center_delta = np.abs(output_centers[first_nz + nz_delta] - output_centers[first_nz])
                except:
                    nz_delta -= 1
                    break
            for nzd in xrange(nz_delta+1):
                row_rescale[first_nz+nzd] = row_sum[first_nz+nz_delta]/row_sum[first_nz+nzd]
                row_rescale[last_nz-nzd] = row_sum[last_nz-nz_delta]/row_sum[last_nz-nzd]
    if preserve_normalization:
        row_rescale = 1.0/np.where(row_sum > 0, row_sum, np.ones(n_out))
    dia_trans = scipy.sparse.dia_matrix((row_rescale, 0), (n_out, n_out))
    trans = dia_trans*trans
    return trans

def get_transformed_covariances(transform_matrix, input_covariance, fill_variance = 0):
    #import pdb; pdb.set_trace()
    if len(input_covariance.shape) == 2:
        out_var = transform_matrix*input_covariance*transform_matrix.transpose()
    elif len(input_covariance.shape) == 1:
        ndat = len(input_covariance)
        ccov = scipy.sparse.dia_matrix((input_covariance, 0), (ndat, ndat))
        out_var = transform_matrix*ccov*transform_matrix.transpose()
    out_var = out_var.tolil()
    if fill_variance != 0:
        for i in range(transform_matrix.shape[0]):
            if out_var[i, i] == 0:
                out_var[i, i] = fill_variance
    out_var = out_var.tocsr()
    return out_var

def Bolton_Schlegel_Diagonalization(flux, covariance):
    """covariance needs to be a dense matrix
"""
    invcov = np.linalg.pinv(covariance)
    evecs, evals = np.linalg(invcov)
    eval_sqrt = np.sqrt(eigen_vals)
    rotmat_no_norm = np.dot(eigen_vecs*eval_sqrt, eigen_vecs.transpose())
    rot_norm = np.dot(rotmat_no_norm, np.ones(len(rotmat_no_norm)))
    rotmat = rotmat_no_norm/rot_norm
    newflux = np.dot(rotmat, flux)
    return newflux, rot_norm, rotmat

def generate_wv_standard(min_wv, max_wv, npts, kind = "linear"):
    """if type == 'linear' wavelengths are equally spaced 
if type == 'log' the wavelengths will be equally spaced in log wavelengths which is equivalently a constant resolution """
    if kind == "log":
        log_wvs = np.linspace(np.log10(min_wv), np.log10(max_wv), npts)
        wvs = np.power(10.0, log_wvs)
    if kind == "linear":
        wvs = np.linspace(min_wv, max_wv, npts)
    return wvs

def simple_coadd(data_wvs, data_flux, data_covar, output_wvs):
    "provides a simple coadd of the data not taking into account the covariance matrix"
    #do the first order by hand
    trans = get_resampling_matrix(data_wvs[0], output_wvs)
    #import pdb; pdb.set_trace()
    output_flux = trans*data_flux[0]
    output_covar = get_transformed_covariances(trans, data_covar[0])
    one_trans = np.zeros(output_wvs.shape)
    one_trans += trans*np.ones(data_flux[0].shape)
    for order_idx in range(1, len(data_wvs)):
            trans = get_resampling_matrix(data_wvs[order_idx], output_wvs, preserve_normalization = True)
            output_flux += trans*data_flux[order_idx]
            one_trans += trans*np.ones(data_flux[order_idx].shape)
            output_covar = output_covar + get_transformed_covariances(trans, data_covar[order_idx])
    output_flux /= one_trans + (one_trans == 0)
    return output_flux, output_covar

def get_curvature_matrix(npts):
    diags = 0.5*np.ones((3, npts))
    diags[1] = -1*np.ones(npts)
    diags[0, -2] = 1.0
    diags[2, 1] = 1.0
    cmat = scipy.sparse.dia_matrix((diags, [-1, 0, 1]), (npts, npts))
    return cmat

def coadd_bolton_schlegel(input_wvs, input_fluxes, input_invvar, input_resolution, output_wvs, output_resolution, preserve_normalization = True, damp = 1e-4, eigen_k = 20):
    """
"""
    transforms = []
    for order_idx in range(len(input_wvs)):
        fill_resval = np.mean(input_resolution[order_idx])
        res_interp = interp1d(input_wvs[order_idx], input_resolution[order_idx], bounds_error = False, fill_value = fill_resval)
        interped_res = res_interp(output_wvs)
        diff_res = np.sqrt(interped_res**2 - output_resolution**2)
        gd = Gaussian_Density(output_wvs, diff_res)
        ctrans = get_resampling_matrix(output_wvs, input_wvs[order_idx], gd, preserve_normalization = preserve_normalization)
        transforms.append(ctrans)
    blocks = [[t] for t in transforms]
    back_trans = scipy.sparse.bmat(blocks)
    npts_in, npts_out = back_trans.shape
    flat_inputs = np.zeros(npts_in)
    flat_invvar = np.zeros(npts_in)
    ridx = 0
    for order_idx in range(len(input_wvs)):
        clen = len(input_wvs[order_idx])
        flat_inputs[ridx:ridx+clen] = input_fluxes[order_idx]
        flat_invvar[ridx:ridx+clen] = input_invvar[order_idx]
        ridx += clen
    full_invvar = scipy.sparse.dia_matrix((flat_invvar, 0), (npts_in, npts_in))
    back_trans_transpose = back_trans.transpose()
    lhs_mat = (back_trans_transpose*full_invvar*back_trans)
    #eigen_vals, eigen_vecs = scipy.sparse.linalg.eigsh(lhs_mat, eigen_k)
    eigen_vals, eigen_vecs = np.linalg.eigh(lhs_mat.todense())
    eval_sqrt = np.sqrt(eigen_vals)
    rotmat_no_norm = np.dot(eigen_vecs*eval_sqrt, eigen_vecs.transpose())
    rot_norm = np.dot(rotmat_no_norm, np.ones(len(rotmat_no_norm)))
    rotmat = rotmat_no_norm/rot_norm
    #import pdb; pdb.set_trace()
    
    rhs_dat = back_trans_transpose*full_invvar*flat_inputs
    if damp != 0:
        sec_der_mat = get_curvature_matrix(npts_out)
        full_lhs_mat = scipy.sparse.bmat([[lhs_mat], [damp*sec_der_mat]])
        full_rhs_dat = np.zeros(npts_out*2)
        full_rhs_dat[:npts_out] = rhs_dat
    else:
        full_lhs_mat = lhs_mat
        full_rhs_dat = rhs_dat
    out_data = scipy.sparse.linalg.lsqr(full_lhs_mat, full_rhs_dat, atol = 0, btol = 0, conlim = 0, iter_lim = 5e5)
    #out_data = scipy.sparse.linalg.lsqr(lhs_mat, rhs_dat, damp = damp, iter_lim = 5e5, show= True)
    #import pdb; pdb.set_trace()
    output_flux = out_data[0]
    reconv_flux = np.dot(rotmat, output_flux)
    return output_flux, rot_norm, rotmat, transforms


def coadd_data(input_wvs, input_fluxes, input_invvar, input_resolution, output_wvs, output_resolution, preserve_normalization = True, damp = 1e-4):
    transforms = []
    for order_idx in range(len(input_wvs)):
        fill_resval = np.mean(input_resolution[order_idx])
        res_interp = interp1d(input_wvs[order_idx], input_resolution[order_idx], bounds_error = False, fill_value = fill_resval)
        interped_res = res_interp(output_wvs)
        diff_res = np.sqrt(interped_res**2 - output_resolution**2)
        gd = Gaussian_Density(output_wvs, diff_res)
        ctrans = get_resampling_matrix(output_wvs, input_wvs[order_idx], gd, preserve_normalization = preserve_normalization)
        transforms.append(ctrans)
    blocks = [[t] for t in transforms]
    back_trans = scipy.sparse.bmat(blocks)
    npts_in, npts_out = back_trans.shape
    flat_inputs = np.zeros(npts_in)
    flat_invvar = np.zeros(npts_in)
    ridx = 0
    for order_idx in range(len(input_wvs)):
        clen = len(input_wvs[order_idx])
        flat_inputs[ridx:ridx+clen] = input_fluxes[order_idx]
        flat_invvar[ridx:ridx+clen] = input_invvar[order_idx]
        ridx += clen
    full_invvar = scipy.sparse.dia_matrix((flat_invvar, 0), (npts_in, npts_in))
    back_trans_transpose = back_trans.transpose()
    lhs_mat = (back_trans_transpose*full_invvar*back_trans)
    rhs_dat = back_trans_transpose*full_invvar*flat_inputs
    if damp != 0:
        sec_der_mat = get_curvature_matrix(npts_out)
        full_lhs_mat = scipy.sparse.bmat([[lhs_mat], [damp*sec_der_mat]])
        full_rhs_dat = np.zeros(npts_out*2)
        full_rhs_dat[:npts_out] = rhs_dat
    else:
        full_lhs_mat = lhs_mat
        full_rhs_dat = rhs_dat
    out_data = scipy.sparse.linalg.lsqr(full_lhs_mat, full_rhs_dat, atol = 0, btol = 0, conlim = 0, iter_lim = 5e5)
    #out_data = scipy.sparse.linalg.lsqr(lhs_mat, rhs_dat, damp = damp, iter_lim = 5e5, show= True)
    #import pdb; pdb.set_trace()
    output_flux, output_var = out_data[0], out_data[-1]
    return output_flux, output_var, transforms

def coadd_data_broken(input_wvs, input_fluxes, input_variances, output_wvs, block_size = 50, block_overlap = 10, wv_overlap = 0.1, preserve_normalization = True):
    """resamples a collection of input data onto a one dimensional output wavelength solution.
    in order to obtain good results it is necessary that the input data are all cross normalized with each other
    meaning that the best fit coefficient of a in a*X = Y for two data vectors X and Y should be a~1. 
    """
    if block_overlap >= block_size:
        print "coadd_error: overlap must be less than block size!"
        return None
    elif block_overlap <= 2:
        print "WARNING: block_overlap should be 3 or greater, smaller values give anomalous results"
    step_size = block_size - block_overlap
    n_out = len(output_wvs)
    n_data = len(input_wvs)
    output_data = np.zeros(output_wvs.shape)
    output_weight_sum = np.zeros(output_wvs.shape)
    n_blocks = int(n_out/step_size)+1
    output_blocks = [(step_size*bi, min(n_out, step_size*bi+block_size)) for bi in range(n_blocks-1)]
    output_blocks.append((n_out-block_size, n_out)) #make sure the last block is the same size as the others
    stime = time.time()
    for block_idx in range(n_blocks):
        cl_idx, cu_idx = output_blocks[block_idx]
        #import pdb; pdb.set_trace()
        min_wv, max_wv = output_wvs[cl_idx]-wv_overlap, output_wvs[cu_idx-1]+wv_overlap
        out_block_wvs = output_wvs[cl_idx:cu_idx]
        input_masks = [(cwvs > min_wv)*(cwvs < max_wv) for cwvs in input_wvs]
        msums = np.array([np.sum(im) for im in input_masks])
        ac_idxs = np.where(msums > 1)[0]
        if len(ac_idxs) == 0:
            continue
        model_matrix = lil_matrix((block_size, len(ac_idxs)*block_size))
        for bi in range(block_size):
            for di in range(len(ac_idxs)):
                model_matrix[bi, bi+di*block_size] = 1.0
        model_matrix = model_matrix.tocsr()
        transforms = [get_resampling_matrix(input_wvs[i][input_masks[i]], out_block_wvs, preserve_normalization = preserve_normalization) for i in ac_idxs]
        c_data = [transforms[i]*(input_fluxes[ac_idxs[i]][input_masks[ac_idxs[i]]]) for i in range(len(ac_idxs))]
        c_variance = [get_transformed_covariances(transforms[i], input_variances[ac_idxs[i]][input_masks[ac_idxs[i]]], fill_variance=0) for i in range(len(ac_idxs))]
        #import pdb; pdb.set_trace()
        c_invvar = []
        for var_idx in range(len(c_variance)):
            cvar_dense = np.array(c_variance[var_idx].todense())
            #invert only the non_zero part of the matrix
            non_zero_mask = np.sum(cvar_dense, axis = 0) > 0
            nzt = np.sum(non_zero_mask)
            if nzt > 0:
                sub_matrix = cvar_dense[non_zero_mask][:, non_zero_mask]
                cur_invvar_sub_matrix = np.linalg.pinv(sub_matrix)
                cur_invvar = np.zeros(cvar_dense.shape)
                cur_rect = np.zeros((np.sum(non_zero_mask), len(cvar_dense)))
                cur_rect[:, non_zero_mask] = cur_invvar_sub_matrix
                cur_invvar[non_zero_mask] = cur_rect
                c_invvar.append(cur_invvar)
            else:
                c_invvar.append(np.zeros(cvar_dense.shape))
        inv_rows = [[None for i in range(len(ac_idxs))] for j in range(len(ac_idxs))]
        for i in range(len(ac_idxs)):
            inv_rows[i][i] = scipy.sparse.csr_matrix(c_invvar[i])
        full_inverse = scipy.sparse.bmat(inv_rows)
        concat_data = np.zeros(block_size*len(ac_idxs))
        for data_idx in range(len(ac_idxs)):
            lb, ub = data_idx*block_size, (data_idx+1)*block_size
            concat_data[lb:ub] = c_data[data_idx]
        rhs = model_matrix*full_inverse*concat_data
        lhs_mat = model_matrix*full_inverse*model_matrix.transpose()
        if np.sum(rhs != 0) > 1:
            #TODO change this to handle the single element case by hand
            block_solution = lsqr(lhs_mat, rhs, damp = 1e-6)[0]
            output_alpha = np.ones(block_size, dtype = float)
            output_alpha[:block_overlap] = np.linspace(0, 1, block_overlap)
            output_alpha[-block_overlap:] = np.linspace(1, 0, block_overlap)
            output_alpha = output_alpha**2
            output_data[cl_idx:cu_idx] += output_alpha * block_solution
            output_weight_sum[cl_idx:cu_idx] += output_alpha
        if (block_idx + 1) % 100 == 0:
            print "processed", block_idx + 1, "of ", n_blocks, "in %d" % (time.time()-stime), "seconds %3.1f" % (float(block_idx+1)/n_blocks)
    output_data /= output_weight_sum + (output_weight_sum <= 0)
    return output_data
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_in = 500
    n_out = 169
    xes = np.linspace(-3*np.pi, 3*np.pi, n_in)
    #testpattern = np.sin(-xes)
    testpattern = np.ones(xes.shape)
    in_coords = np.linspace(1, 8, n_in)
    out_coords = np.linspace(0, 10, n_out)
    coord_spread = np.ones(n_in)*0.4
    gdi = Gaussian_Density(in_coords, coord_spread)
    #lid = Linear_Interpolated_Density(in_coords, testpattern)
    trans = get_resampling_matrix(in_coords, out_coords, gdi)
    #tdense = [lid.get_density_integral(20, oc) for oc in out_coords]
    cp = trans*testpattern
