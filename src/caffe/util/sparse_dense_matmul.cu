/*
 * sparse_dense_matmul.cu
 *
 *  Created on: Oct 3, 2017
 *      Author: cfeng
 */

#include "caffe/util/sparse_dense_matmul.hpp"

namespace caffe {

//////////////////////////////////////////////////////////////////////////////////////////////////////
//following part modified from:
//https://github.com/beniz/caffe/blob/master_dd_integ_sparse/src/caffe/util/math_functions.cu
//////////////////////////////////////////////////////////////////////////////////////////////////////
#define THREADS_PER_BLOCK_CSR 32 //TODO: 32 or 64 or 128 or 256 or 512?

template<typename Dtype>
__device__ void caffe_gpu_csr_gemm_kernel_core(const int M, const int N, const int K,
                                               const Dtype alpha,
                                               const Dtype* A, const Dtype* indices, const Dtype* ptr,
                                               const Dtype* B,
                                               const int ldb1, const int ldb2,
                                               const Dtype beta,
                                               Dtype* C,
                                               const int ldc1, const int ldc2)
{
  __shared__ volatile Dtype sums[THREADS_PER_BLOCK_CSR * 2];

  for (int rowA = blockIdx.x; rowA < M; rowA += gridDim.x) {
    const int begin = (int)ptr[rowA];  //TODO: use reinterpret_cast<int>
    const int end = (int)ptr[rowA + 1];//TODO: use reinterpret_cast<int>
    const int offset_c_part = rowA * ldc1;
    for (int colC = blockIdx.y; colC < N; colC += gridDim.y) {
      Dtype sum = 0.0;
      const int offset_b_part = colC * ldb2;
      for (int pos = begin + threadIdx.x; pos < end; pos +=
          THREADS_PER_BLOCK_CSR) {
        const int colA = (int)indices[pos]; //TODO: use reinterpret_cast<int>
        sum += A[pos] * B[colA * ldb1 + offset_b_part];
      }
      sums[threadIdx.x] = sum;
      __syncthreads();

      ///* hardcoded reduction for THREADS_PER_BLOCK_CSR threads */
      //sums[threadIdx.x] += sums[threadIdx.x + 16];
      //sums[threadIdx.x] += sums[threadIdx.x + 8];
      //sums[threadIdx.x] += sums[threadIdx.x + 4];
      //sums[threadIdx.x] += sums[threadIdx.x + 2];
      //sums[threadIdx.x] += sums[threadIdx.x + 1];
      int offset=THREADS_PER_BLOCK_CSR/2;
      while(offset>0) {
        sums[threadIdx.x] += sums[threadIdx.x + offset];
        offset/=2;
        //__syncthreads(); //note: this seems not necessary
      }

      if (threadIdx.x == 0) {
        const int offsetC = offset_c_part + colC * ldc2;
        C[offsetC] = beta * C[offsetC] + alpha * sums[0];
      }
    }
  }
}

template<typename Dtype>
__global__ void caffe_gpu_csr_gemm_kernel(const CBLAS_TRANSPOSE TransB,
                                          const int M, const int N, const int K,
                                          const Dtype alpha,
                                          const Dtype* A, const Dtype* indices, const Dtype* ptr,
                                          const Dtype* B,
                                          const Dtype beta,
                                          Dtype* C,
                                          const CBLAS_ORDER orderC)
{
  if (orderC == CblasRowMajor) {
    if (TransB == CblasNoTrans) {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, N,
                                     1, beta, C, N, 1);
    } else {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, 1,
                                     K, beta, C, N, 1);
    }
  } else {
    if (TransB == CblasNoTrans) {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, N,
                                     1, beta, C, 1, M);
    } else {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, 1,
                                     K, beta, C, 1, M);
    }
  }
}

template<typename Dtype>
__device__ void caffe_gpu_csr_rank1_update_kernel_core(const int M, const int N,
                                                       const Dtype alpha,
                                                       const Dtype* A, const Dtype* indices, const Dtype* ptr,
                                                       const Dtype* B,
                                                       int ldb,
                                                       Dtype* C,
                                                       const int ldc1, const int ldc2)
{
  const int begin = (int)ptr[0]; //TODO: use reinterpret_cast<int>
  const int end = (int)ptr[1];   //TODO: use reinterpret_cast<int>
  for (int pos = blockIdx.x * blockDim.x + begin + threadIdx.x; pos < end;
      pos += blockDim.x * gridDim.x) {
    const Dtype valA = A[pos] * alpha;
    const int offset_part = (int)indices[pos] * ldc1; //TODO: use reinterpret_cast<int>
    for (int colC = blockIdx.y * blockDim.y + threadIdx.y; colC < N;
        colC += blockDim.y * gridDim.y) {
      const int C_offset = offset_part + colC * ldc2;
      C[C_offset] = C[C_offset] + B[colC * ldb] * valA;
    }
  }
}

// C = alpha A * B^T +  C where A and B are vectors.
// A is a sprase vector and B is a dense vector
template<typename Dtype>
__device__ void caffe_gpu_csr_rank1_update_kernel(const int M, const int N,
                                                  const Dtype alpha,
                                                  const Dtype* A, const Dtype* indices, const Dtype* ptr,
                                                  const Dtype* B,
                                                  int ldb,
                                                  Dtype* C,
                                                  const CBLAS_ORDER orderC)
{
  if (orderC == CblasRowMajor) {
    caffe_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, N, 1);
  } else {
    caffe_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, 1, M);
  }
}

template<typename Dtype>
__global__ void caffe_gpu_csr_rank1_update_kernel_multi(
    const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const Dtype alpha,
    const Dtype* A, const Dtype* indices, const Dtype* ptr,
    const Dtype* B,
    int ldb,
    Dtype* C,
    const CBLAS_ORDER orderC)
{
  if (TransB == CblasNoTrans) {
    for (int i = 0; i < K; i++) {
      caffe_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i,
                                        B + (N * i), 1, C, orderC);
    }
  } else {
    for (int i = 0; i < K; i++) {
      caffe_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i, B + i,
                                        K, C, orderC);
    }
  }
}

const int MAX_NR_OF_A_ELEMENTS = 512;
// Declare as a static variable to
// limit the scopt within this file.
static
__constant__ double4 pd4ABuffer_const[MAX_NR_OF_A_ELEMENTS];

template<typename Dtype>
__global__ void caffe_gpu_csr_gemm_single_row_kernel(
		const CBLAS_TRANSPOSE TransB,
		const unsigned int M,
		const unsigned int N,
		const unsigned int K,
		const unsigned int uARow,
		const Dtype fAlpha,
		const unsigned int uNrOfAElements,
		const Dtype* pfB,
		Dtype* pfC,
		const CBLAS_ORDER orderC)
{
  if ( CblasRowMajor == orderC ) {
    if ( CblasNoTrans == TransB ) {

    	const unsigned int uCCol = blockIdx.x * blockDim.x + threadIdx.x;

		if( N > uCCol )
		{
			const unsigned int uBCol = uCCol;
			Dtype fSum = Dtype(0);
			for(unsigned int a = 0; a < uNrOfAElements; ++a)
			{
				const double4 d4AElement = pd4ABuffer_const[a];
				const unsigned int uACol = (unsigned int)d4AElement.x;
				const unsigned int uBRow = uACol;

				fSum += pfB[uBRow * N + uBCol] * Dtype(d4AElement.z);
			}
			const unsigned int uCRow = uARow;
			const unsigned int uCIndex = uCRow * N + uCCol;
			pfC[uCIndex] += fAlpha * fSum;
		}

    } else {
    	// NOT IMPLEMENTED YET.
    	/*
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, 1,
                                     K, beta, C, N, 1);
                                     */
    }
  } else {
  	// NOT IMPLEMENTED YET.
	/*
    if (TransB == CblasNoTrans) {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, N,
                                     1, beta, C, 1, M);
    } else {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, A, indices, ptr, B, 1,
                                     K, beta, C, 1, M);
    }
	*/
  }
}

template<typename Dtype>
void caffe_gpu_csr_gemm_row_based(
		const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB,
		const int M,
		const int N,
		const int K,
		const Dtype alpha,
		const Dtype* A,
		const Dtype* indices,
		const Dtype* ptr,
		const Dtype* B,
		const Dtype beta,
		Dtype* C,
		const CBLAS_ORDER orderC,
		const int nnz)
{
	  /*
	   * An implementation w/ coalesced memory access.
	   * For each row i of the sparse A,
	   * 	Unpack the values of A[i ...] to GPU constant memory.
	   * 	Assign each row of B to a kernel.
	   * 	Inside the kernel for row j,
	   * 		Iterate through the constant memory:
	   * 		for k = ...
	   * 			sum += A[i, k] * B[k, j].
	   *		C[i, j] = sum.
	   *
	   */
	  /*
	   * Nevertheless because constant memory in GPU is small,
	   * each row of A should be divided into segments
	   * s.t. each segment can be fitted into the constant memory.
	   *
	   * Thus the revised algorithm is:
	   *
	   * For each row i of the sparse A,
	   * 	for each segment along the the row,
	   * 		Unpack the values of A[i ...] to GPU constant memory.
	   * 		Assign each row of B to a kernel.
	   * 		Inside the kernel for row j,
	   * 			Iterate through the constant memory:
	   * 			for k = ...
	   * 				sum += A[i, k] * B[k, j].
	   * 			C[i, j] = sum.
	   *
	   */
	  // TODO: Replace the iteration in the loop by reduction operators.

	  caffe_gpu_scal(M * N, beta, C);

	  // TODO: Support double.
	  const unsigned int uNrOfAElementsPerKernelCall = sizeof(pd4ABuffer_const) / sizeof(pd4ABuffer_const[0]);
	  double4 pd4ABuffer_host[uNrOfAElementsPerKernelCall];

	  for(unsigned int uARow = 0; uARow < (unsigned int)M; ++uARow)
	  {
          const unsigned int begin = (unsigned int)ptr[uARow];
          const unsigned int end = (unsigned int)ptr[uARow + 1];

          for(unsigned int uNrOfBufferedAElements = 0, pos = begin; pos < end; ++pos)
          {
        	  pd4ABuffer_host[uNrOfBufferedAElements] = make_double4(
        			  double(indices[pos]),
					  double(uARow),
					  double(A[pos]),
					  double(0) // Reserved.
					  );
        	  ++uNrOfBufferedAElements;

        	  if( uNrOfAElementsPerKernelCall <= uNrOfBufferedAElements || (end - 1 <= pos && 0 < uNrOfBufferedAElements ) )
        	  {
				  CUDA_CHECK(
					  cudaMemcpyToSymbol(
							pd4ABuffer_const,
							&pd4ABuffer_host[0],
							uNrOfBufferedAElements * sizeof(pd4ABuffer_host[0]),
							0,
							cudaMemcpyHostToDevice)
				  );

				  dim3 v3Block(128);
				  dim3 v3Grid((N + v3Block.x - 1) / v3Block.x);

				  caffe_gpu_csr_gemm_single_row_kernel<<<v3Grid, v3Block, 0>>>
				  (
						  TransB,
						  (unsigned int)M,
						  (unsigned int)N,
						  (unsigned int)K,
						  uARow,
						  alpha,
						  uNrOfBufferedAElements,
						  B,
						  C,
						  orderC
				  );
				  CUDA_POST_KERNEL_CHECK;

				  uNrOfBufferedAElements = 0;
        	  }
          }
	  }
}

template<typename Dtype>
void caffe_gpu_csr_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const Dtype alpha,
        const Dtype* A,
		const Dtype* indices,
		const Dtype* ptr,
        const Dtype* B,
        const Dtype beta,
		Dtype* C,
        const CBLAS_ORDER orderC,
        const int nnz)
{
  // Current version can only work if
  // the data/pointers/indices of A are on CPU,
  // which will not work w/ the old implementation.
  // Thus it should immediately fail
  // for non-supported cases.
  LOG_ASSERT( CblasRowMajor == orderC );
  LOG_ASSERT( CblasNoTrans == TransB );

  unsigned int uNrOfARows = M;
  unsigned int uNrOfACols = K;
  // Here we need to undo the transform,
  // which is confusing.
  if ( CblasTrans == TransA )
  {
	  uNrOfARows = K;
	  uNrOfACols = M;
  }
  const Dtype* AData_host = A;
  const Dtype* AIndices_host = indices;
  const Dtype* APtr_host = ptr;

  vector<Dtype> vATData;
  vector<Dtype> vATIndices;
  vector<Dtype> vATPtr;

  if ( CblasTrans == TransA )
  {
	const unsigned int uNrOfATRows = uNrOfACols;
	const unsigned int uNrOfATCols = uNrOfARows;

	// Generate new data/indices/ptrs to mimic the transpose of A.
	vector<unsigned int> vuATRowCounts(uNrOfATRows);
	std::fill(vuATRowCounts.begin(), vuATRowCounts.end(), (unsigned int)0);

	for(unsigned int uARow = 0; uARow < uNrOfARows; ++uARow)
	{
		const unsigned int begin = (unsigned int)ptr[uARow];
		const unsigned int end = (unsigned int)ptr[uARow + 1];
		for(int e = begin; e < end; ++e)
		{
			const unsigned int uACol = indices[e];
			const unsigned int uATRow = uACol;
			vuATRowCounts[uATRow]++;
		}
	}

	vector<unsigned int> vuATRowIndices(vuATRowCounts.size() + 1);
	vuATRowIndices[0] = 0;

	vATPtr.resize(vuATRowIndices.size());
	vATPtr[0] = Dtype(0);
	unsigned int uNrOfAElements = 0;
	for(unsigned int uATRow = 0; uATRow < uNrOfATRows; ++uATRow)
	{
		uNrOfAElements += vuATRowCounts[uATRow];
		vuATRowIndices[uATRow + 1] = uNrOfAElements;
		vATPtr[uATRow + 1] = Dtype(uNrOfAElements);
	}

	vATIndices.resize(uNrOfAElements);
	vATData.resize(uNrOfAElements);
	for(unsigned int uARow = 0; uARow < uNrOfARows; ++uARow)
	{
		const unsigned int begin = (unsigned int)ptr[uARow];
		const unsigned int end = (unsigned int)ptr[uARow + 1];
		for(int e = begin; e < end; ++e)
		{
			const unsigned int uACol = indices[e];
			const unsigned int uATRow = uACol;
			const unsigned int uATCol = uARow;

			const unsigned int uATIndex = vuATRowIndices[uATRow];

			vATIndices[uATIndex] = uATCol;
			vATData[uATIndex] = A[e];

			vuATRowIndices[uATRow]++;
		}
	}

    uNrOfARows = uNrOfATRows;
    uNrOfACols = uNrOfATCols;
	AData_host = vATData.data();
	AIndices_host = vATIndices.data();
	APtr_host = vATPtr.data();
  }

  caffe_gpu_csr_gemm_row_based<Dtype>(
			TransA,
			TransB,
			uNrOfARows,
			N,
			uNrOfACols,
			alpha,
			AData_host,
			AIndices_host,
			APtr_host,
			B,
			beta,
			C,
			orderC,
			nnz);
}

// instantiation
template
void caffe_gpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const float alpha,
        const float* A,
		const float* indices,
		const float* ptr,
        const float* B,
        const float beta,
		float* C,
        const CBLAS_ORDER orderC,
        const int nnz);

template
void caffe_gpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const double alpha,
        const double* A,
		const double* indices,
		const double* ptr,
        const double* B,
        const double beta,
		double* C,
        const CBLAS_ORDER orderC,
        const int nnz);

}
