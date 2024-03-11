import torch
import numpy as np

class Quantizer():

    def __init__(self):
        self.qp = 22
        self.qpStep = pow(pow(2,(1/6)), self.qp-4)

        self.quant_matrix = torch.tensor([
            [16, 16, 16, 16, 17, 18, 21, 24],
            [16, 16, 16, 16, 17, 19, 22, 25],
            [16, 16, 17, 18, 20, 22, 25, 29],
            [16, 16, 18, 21, 24, 27, 31, 36],
            [17, 17, 20, 24, 30, 35, 41, 47],
            [18, 19, 22, 27, 35, 44, 54, 65],
            [21, 22, 25, 31, 41, 54, 70, 88],
            [24, 25, 29, 36, 47, 65, 88, 115]
        ])
        self.bitDepth = 8
        self.size = self.quant_matrix.shape[0]
        
        

        self.Fqp = pow(2,14)/pow(2,((self.qp-4)/6))

        #infering that is the oposi of the ofssetIQ
        self.offsetQ = 1 >> (self.size-6+self.bitDepth)
        self.offsetIQ = 1 << (self.size-6+self.bitDepth)
        self.shift1 = (self.size-5+self.bitDepth)
        self.shift2 = 29-self.size-self.bitDepth

    def dequantize(self, image: torch.tensor):
        #the pows (shift,2) are substituting the shifts
        coeff =  (image * self.quant_matrix * self.qpStep + self.offsetIQ) / pow(2,self.shift1)

        return coeff

    def quantize(self, image: torch.tensor):

        level = torch.sign(image) * (((torch.abs(image) * self.Fqp * (16/self.quant_matrix) + self.offsetQ)
                                / (pow(2,self.qp/6))) / pow(2, self.shift2))
        return level

#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#quantizer = Quantizer()

class lowPass():

    def __init__(self, block_size = 32):

        x, y = np.meshgrid(np.arange(block_size), np.arange(block_size))
        distances = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)
        scaled_distances = distances / distances.max()
        self.matrix_values = 1 - 0.75 * scaled_distances
        self.matrix_values = np.clip(self.matrix_values, 0.25, 1)
        self.matrix_values = torch.from_numpy(self.matrix_values)

    def quantize(self, image):
        return image*self.matrix_values
    def dequantize(self, image):
        return image/self.matrix_values
    def quantDequant(self, image):
        return self.dequantize(self.quantize(image))

#np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.inf)
#       
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#quantizer = lowPass(8)
#image = torch.rand(8,8)
#print("orig:", image)
#imageq = quantizer.quantize(image)
#print("quant:", imageq)
#imaged = quantizer.dequantize(imageq)
#print("deq:", imaged)
#print("MSE: ", torch.mean((imaged-image)**2))