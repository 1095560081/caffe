/*
 * sparse_dense_matmul.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: cfeng
 */

#include "caffe/util/math_functions.hpp"
#include "caffe/util/sparse_dense_matmul.hpp"

namespace caffe {
#define SDMM_USE_OPENMP 0

//the following functions modified from:
//https://github.com/beniz/caffe/blob/master_dd_integ_sparse/src/caffe/util/math_functions.cpp

template<>
void caffe_axpy2<float>(const int N, const float alpha, const float* X, float* Y,
                       const int ldx, const int ldy) {
  cblas_saxpy(N, alpha, X, ldx, Y, ldy);
}

template<>
void caffe_axpy2<double>(const int N, const double alpha, const double* X,
                        double* Y, const int ldx, const int ldy) {
  cblas_daxpy(N, alpha, X, ldx, Y, ldy);
}


template<typename Dtype>
void caffe_cpu_csr_gemm(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const Dtype alpha,
                        const Dtype* A, const Dtype* indices, const Dtype* ptr,
                        const Dtype* B, const Dtype beta, Dtype* C,
                        const CBLAS_ORDER orderC)
{
  if (TransA == CblasNoTrans) {  // CSR
    caffe_scal(M * N, beta, C);
    if (orderC == CblasRowMajor) {
      if (TransB == CblasNoTrans) {
#if SDMM_USE_OPENMP
#pragma omp parallel for
#endif
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + (N * rowA);
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolAN = B + ((int)indices[pos] * N);
            const Dtype AatPos = alpha * A[pos];
            caffe_axpy2(N, AatPos, BcolAN, CrowA, 1, 1);
          }
        }
      } else {
#if SDMM_USE_OPENMP
#pragma omp parallel for
#endif
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + (N * rowA);
          for (int pos = begin; pos < end; pos++) {
            const Dtype AatPos = alpha * A[pos];
            const Dtype* BcolA = B + (int)indices[pos];
            caffe_axpy2(N, AatPos, BcolA, CrowA, K, 1);
          }
        }
      }
    } else {
      if (TransB == CblasNoTrans) {
#if SDMM_USE_OPENMP
#pragma omp parallel for
#endif
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + rowA;
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolAN = B + ((int)indices[pos] * N);
            const Dtype AatPos = alpha * A[pos];
            caffe_axpy2(N, AatPos, BcolAN, CrowA, 1, M);
          }
        }
      } else {
#if SDMM_USE_OPENMP
#pragma omp parallel for
#endif
        for (int rowA = 0; rowA < M; rowA++) {
          const int begin = (int)ptr[rowA];
          const int end = (int)ptr[rowA + 1];
          Dtype* CrowA = C + rowA;
          for (int pos = begin; pos < end; pos++) {
            const Dtype* BcolA = B + (int)indices[pos];
            const Dtype AatPos = alpha * A[pos];
            caffe_axpy2(N, AatPos, BcolA, CrowA, K, M);
          }
        }
      }
    }
  } else {  // A is CSC
    caffe_scal(M * N, beta, C);
    if (orderC == CblasRowMajor) {
      if (TransB == CblasNoTrans) {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColAN = B + (colA * N);
          for (int pos = begin; pos < end; pos++) {
            caffe_axpy2(N, A[pos] * alpha, BColAN,
                            C + ((int)indices[pos] * N), 1, 1);
          }
        }
      } else {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColA = B + colA;
          for (int pos = begin; pos < end; pos++) {
            caffe_axpy2(N, A[pos] * alpha, BColA, C + ((int)indices[pos] * N),
                           K, 1);
          }
        }
      }
    } else {
      if (TransB == CblasNoTrans) {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColAN = B + (colA * N);
          for (int pos = begin; pos < end; pos++) {
            caffe_axpy2(N, A[pos] * alpha, BColAN, C + (int)indices[pos], 1, M);
          }
        }

      } else {
        for (int colA = 0; colA < K; colA++) {
          const int begin = (int)ptr[colA];
          const int end = (int)ptr[colA + 1];
          const Dtype* BColA = B + colA;
          for (int pos = begin; pos < end; pos++) {
            caffe_axpy2(N, A[pos] * alpha, BColA, C + (int)indices[pos], K,  M);
          }
        }
      }
    }
  }
}

// instantiation
template
void caffe_cpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const float alpha,
                        const float* A, const float* indices, const float* ptr,
                        const float* B, const float beta, float* C,
                        const CBLAS_ORDER orderC);

template
void caffe_cpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha,
                        const double* A, const double* indices, const double* ptr,
                        const double* B, const double beta, double* C,
                        const CBLAS_ORDER orderC);


#if 0
template<typename Dtype>
static void test_csr_gemm()
{
  Dtype A_data[4]    = {1,3,2,4};
  Dtype A_indices[4] = {0,2,1,2};
  Dtype A_indptr[3]  = {0,2,4};
  const int A_rows = 2;
  const int A_cols = 3;

  Dtype B[6] = {
      1, 4,
      2, 5,
      3, 6
  };
  //const int B_rows = 3;
  const int B_cols = 2;

  Dtype AB[4] = {
      10, 22,
      16, 34
  };

  Dtype C[4] = {
      1, 4,
      2, 5
  };
  //const int C_rows = 2;
  const int C_cols = 2;

  Dtype AtC[6] = {
      1, 4,
      4, 10,
      11, 32
  };

  Dtype AB_cmp[4] = {0,0,0,0};

  caffe_cpu_csr_gemm(CblasNoTrans, CblasNoTrans,
      A_rows, B_cols, A_cols,
      (Dtype)1.0,
      A_data, A_indices, A_indptr,
      B,
      (Dtype)0.0,
      AB_cmp,
      CblasRowMajor);
  for(int i=0; i<A_rows; ++i) {
    for(int j=0; j<B_cols; ++j) {
      CHECK_EQ(AB[i*B_cols+j], AB_cmp[i*B_cols+j]);
    }
  }

  Dtype AtC_cmp[6] = {0,0,0,0,0,0};
  caffe_cpu_csr_gemm(CblasTrans, CblasNoTrans,
      A_cols, C_cols, A_rows,
      (Dtype)1.0,
      A_data, A_indices, A_indptr,
      C,
      (Dtype)0.0,
      AtC_cmp,
      CblasRowMajor);
  for(int i=0; i<A_cols; ++i) {
    for(int j=0; j<C_cols; ++j) {
      CHECK_EQ(AtC[i*C_cols+j], AtC_cmp[i*C_cols+j]);
    }
  }

  std::cout << "test_csr_gemm passed OK!" <<std::endl <<std::flush;
}
#endif

}
