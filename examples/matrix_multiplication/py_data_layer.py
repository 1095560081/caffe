import caffe
import numpy as np

class LineDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 10;
        top[0].reshape(self.batch_size, 2, 1)
        top[1].reshape(self.batch_size)
        
        self.W = np.array([2,5], dtype=np.float32)
        self.b = -7.0
        print("W=\n"+str(self.W))
        print("b="+str(self.b))
    
    def forward(self, bottom, top):
        for b in xrange(self.batch_size):
            x = np.random.rand(2).astype(np.float32)
            y = self.W.dot(x)+self.b
            top[0].data[b,...]=x.reshape(2,1)
            top[1].data[b,...]=y
    
    def reshape(self, bottom, top):
        pass
    
    def backward(self, top, propagate_down, bottom):
        pass