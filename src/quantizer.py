import torch


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
        
        self.shift1 = (self.size-5+self.bitDepth)
        self.shift2 = 29-self.size-self.bitDepth

        self.Fqp = pow(2,14)/pow(2,((self.qp-4)/6))

        #infering that is the oposi of the ofssetIQ
        self.offsetQ = 1 >> (self.bitDepth-6+self.size)
        self.offsetIQ = 1 << (self.bitDepth-6+self.size)
        

    def dequantize(self, image: torch.tensor):
        #the pows (shift,2) are substituting the shifts
        coeff =  (image * self.quant_matrix * self.qpStep + self.offsetIQ) / pow(self.shift1,2)

        return coeff

    def quantize(self, image: torch.tensor):
        level = torch.sign(image) * (((torch.abs(image)*self.Fqp * (16/self.quant_matrix)+self.offsetQ)
                                        / (pow(self.qp/6,2))) / pow(self.shift2,2))
        return level

quantizer = Quantizer()
image = torch.rand(8,8)
print("orig:", image)
image = quantizer.quantize(image)
print("orig:", image)
image = quantizer.dequantize(image)
print("orig:", image)