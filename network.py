import torch.nn as nn
import torch


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = None
        # xf: the fourier transformation of target patch x.
        self.xf = None
        self.config = config

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (num_scale, 3, crop_sz, crop_sz)
        :return response: the response of cross correlation. Shape (num_scale, 1, crop_sz, crop_sz)

        You are required to calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window
        z = self.feature(z) * self.config.cos_window
        # TODO: You are required to calculate response using self.wf to do cross correlation on the searching patch z
        num_scale, channels, crop_sz, crop_sz = z.shape
        # take fourier transform of z
        zf = torch.rfft(z, 2, normalized=False)
        # w_star = the complex conjugate of wf 
        w_star = self.wf.clone().detach()
        w_star[:,:,:,:,1] = w_star[:,:,:,:,1] * -1
        # ftrans_out is the output matrix before you do the inverse fourier transform
        
        ftrans_out = torch.cuda.FloatTensor(num_scale, 1, crop_sz, crop_sz//2+1, 2).fill_(0)
        # the first dimension of the output is num_scale, but the first dimension of w is 1, so Calculate for each numscale?  
        for c in range(num_scale):
            # we want to sum over channels, so cycle through each channel
            for l in range(channels):
                # temp is W*z like in in equation 4
                temp = torch.mul(w_star[0,1,:,:,:],zf[c,l,:,:,:])
                out_real, out_imag = self.imag_mult(w_star[0,1,:,:,0],w_star[0,1,:,:,1],zf[c,l,:,:,0],zf[c,l,:,:,1]) 
                temp[:,:,0] = out_real
                temp[:,:,1] = out_imag
                ftrans_out[c,0,:,:,:] += temp
        # the response is the inverse Fourier transform of this
        response = torch.irfft(ftrans_out, 2)

        return response


    def imag_mult(self, matrix_a_real, matrix_a_imag, matrix_b_real, matrix_b_imag):
        out_real = matrix_a_real*matrix_b_real - matrix_a_imag*matrix_b_imag
        out_imag = matrix_a_real*matrix_b_imag + matrix_a_imag*matrix_b_real
        return out_real, out_imag

    def imag_div(self, matrix_a_real, matrix_a_imag, matrix_b_real, matrix_b_imag):
         out_real = (matrix_a_real*matrix_b_real - matrix_a_imag*matrix_b_imag) / (torch.mul(matrix_b_real,matrix_b_real) + torch.mul(matrix_b_imag, matrix_b_imag))
         out_imag =  (matrix_a_real*matrix_b_imag + matrix_a_imag*matrix_b_real) / (torch.mul(matrix_b_real,matrix_b_real) + torch.mul(matrix_b_imag, matrix_b_imag))
         return out_real, out_imag

    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula,
                    Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
        x = self.feature(x) * self.config.cos_window
        # TODO: calculate self.xf and self.wf
        # put your code here
        scale_size, channels, crop_sz, crop_sz = x.shape
 
        # xf is the fourier transform of x
        xf = torch.rfft(x,2)
        # if self.xf is none (hasn't been defined yet) update it to lr * xf, otherwise, use the accumulation rule from github
        if type(self.xf) == type(None):
            self.xf = lr*xf
        else:
            self.xf = (1-lr)*self.xf.data + lr*xf
        # the numerator of equation 3 is xf for each l multiplied by yf
        y_star = self.config.yf.clone().detach()
        y_star[:,:,:,:,1] = y_star[:,:,:,:,1] * -1
        numerator = torch.mul(self.xf, y_star)
        out_real, out_imag = self.imag_mult(self.xf[:,:,:,:,0],self.xf[:,:,:,:,1],self.config.yf[:,:,:,:,0],self.config.yf[:,:,:,:,1])
        numerator[:,:,:,:,0] = out_real 
        numerator[:,:,:,:,1] = out_imag
        # to get the denominator, we sum over all channels the product of the complex conjugate of self.xf * lambda
        denominator = torch.cuda.FloatTensor(1, channels, crop_sz, crop_sz//2+1, 2).fill_(0)
        phi_K = torch.cuda.FloatTensor(1,crop_sz, crop_sz//2+1, 2).fill_(0)
        for k in range(channels):
            # since we're summing for each channel, phi_K becomes a 4 dimensional tensor where the last two columns are the real and imaginary componenets
            phi_K[:,:,:,:] = self.xf[:,k,:,:,:]
            conj_phi_K = phi_K.clone().detach()
            conj_phi_K[:,:,:,1] = conj_phi_K[:,:,:,1] * -1
            final_prod = phi_K*conj_phi_K
            # the product of complex conjugates a+bi * a-bi is a^2 + b^2, so the real component of the final product is the data in column 0 squared times plus the data in column 1 squared.  There is no imaginary component after this transformation, so the imaginary components can all be set to zero
            out_real, out_imag = self.imag_mult(phi_K[:,:,:,0], phi_K[:,:,:,1], conj_phi_K[:,:,:,0], conj_phi_K[:,:,:,1])
            final_prod[:,:,:,0] = out_real
            final_prod[:,:,:,1] = out_imag 
            denominator[:,0,:,:,:] += final_prod + self.config.lambda0
        for k in range(channels):
            denominator[:,k,:,:,:] = denominator[:,0,:,:,:]
        wf = numerator / denominator
        out_real, out_imag = self.imag_div(numerator[:,:,:,:,0], numerator[:,:,:,:,1], denominator[:,:,:,:,0], denominator[:,:,:,:,1])
        wf[:,:,:,:,0] = out_real
        wf[:,:,:,:,1] = out_imag
        # if self.wf hasn't been initialized, set it to lr * wf, otherwise use the update rule in github
        if type(self.wf) == type(None):
            self.wf =lr*wf
        else:
            self.wf = (1-lr)*self.wf.data + lr * wf


            

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

