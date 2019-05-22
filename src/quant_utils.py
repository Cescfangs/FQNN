import numpy as np


def quantize_mat(arr, percentile=0):
    low, high = np.percentile(
        arr, percentile), np.percentile(arr, 100 - percentile)
    mat_range = high - low
    uint8_mat = np.round((arr - low) * 255 / mat_range).astype(np.uint8)
    offset = -int(round(low * 255 / mat_range))
    return uint8_mat, mat_range, offset


def uint8_matmul(a, a_offset, a_range, b, b_offset, b_range):
    a_ = a.astype(np.int32)
    b_ = b.astype(np.int32)
    a_ += a_offset
    b_ += b_offset
    c = a_.dot(b_).astype(np.float32)
    c *= a_range * b_range / 255 / 255
    return c


def compute_error(ref_mat, inf_mat, verbose=0):
    diff = ref_mat - inf_mat
    if verbose > 0:
        print('=' * 10 + ' ref mat ' + '=' * 10)
        print(ref_mat)
        print('=' * 10 + ' inf mat ' + '=' * 10)
        print(inf_mat)
        print('=' * 10 + ' diff ' + '=' * 10)
        print(diff)
    error = np.abs(diff).sum() / np.abs(ref_mat).sum()
    return error


def add_noise(mat, noise_prob=0.1, noise_level=1):
    mask = np.random.rand(*mat.shape) < noise_prob
    mat_ = mat.copy()
    mat_[mask] *= noise_level
    return mat_
