import numpy as np
import sys
import caffe

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver("solve_matmul.ptt")
solver.net.blobs['W'].data[...]=np.random.randn(1,2)
print("W_0=\n")
print(np.array(solver.net.blobs['W'].data))
solver.solve()
print("W=\n")
print(np.array(solver.net.blobs['W'].data))
print("b=\n")
print(solver.net.params["add_WX_b"][0].data)
