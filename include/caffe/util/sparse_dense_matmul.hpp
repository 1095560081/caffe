/*
 * sparse_dense_matmul.hpp
 *
 *  Created on: Oct 3, 2017
 *      Author: cfeng
 */

#ifndef INCLUDE_CAFFE_UTIL_SPARSE_DENSE_MATMUL_HPP_
#define INCLUDE_CAFFE_UTIL_SPARSE_DENSE_MATMUL_HPP_

#include "caffe/util/math_functions.hpp"

namespace caffe {

//the following functions modified from:
//https://github.com/beniz/caffe/blob/master_dd_integ_sparse/src/caffe/util/math_functions.cpp

template<typename Dtype>
void caffe_axpy2(const int N, const Dtype alpha, const Dtype* X, Dtype* Y,
                       const int ldx, const int ldy);


template<typename Dtype>
void caffe_cpu_csr_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const Dtype alpha,
                        const Dtype* A, const Dtype* indices, const Dtype* ptr,
                        const Dtype* B, const Dtype beta, Dtype* C,
                        const CBLAS_ORDER orderC);


template<typename Dtype>
void caffe_gpu_csr_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                        const int M, const int N, const int K,
                        const Dtype alpha,
                        const Dtype* A, const Dtype* indices, const Dtype* ptr,
                        const Dtype* B,
                        const Dtype beta,
                        Dtype* C,
                        const CBLAS_ORDER orderC,
                        const int nnz=0);

}


#endif /* INCLUDE_CAFFE_UTIL_SPARSE_DENSE_MATMUL_HPP_ */
