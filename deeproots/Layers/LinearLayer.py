import Mat from deeproots.DataTypes.Mat import Mat

class LinearLayer:

    def __init__(self, nInputs, nNeurons, tag=''):
        self.nInputs  = nInputs
        self.nNeurons = nNeurons
        self.tag      = tag

        self.w = Mat(nNeurons, nInputs, tag='w')
        self.b = Mat(nNeurons, 1, tag='b')
        self.x = None


    def forward(self,x):
        self.x = x
        self.a = Mat.dot(self.w, self.x) + self.b
        return self.a


    def backward(self, dL_da):

        self.dw    = Mat.dot(dL_da, self.x)
        self.db    = Mat.sum(dL_da, axis=0)
        self.dl_dx = Mat.dot(self.w.T, dL_da)

        


#----------------------------------------------------------------------


def main():

    x = Mat(2, 3, tag='test')
    print(x)

    layer = LinearLayer(2, 3, tag='test')
    layer.forward(x)



#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------
