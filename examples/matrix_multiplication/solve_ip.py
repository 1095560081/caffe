import numpy as np
import sys
import caffe

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver("solve_ip.ptt")
print("W_0=\n")
print(np.array(solver.net.params['ip_WXpb'][0].data))
solver.solve()
print("W=\n")
print(np.array(solver.net.params['ip_WXpb'][0].data))
print("b=\n")
print(np.array(solver.net.params['ip_WXpb'][1].data))
