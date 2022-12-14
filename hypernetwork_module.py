

# @article{gaurav2018hypernetsgithub,
#   title={HyperNetworks(Github)},
#   author={{Mittal}, G.},
#   howpublished = {\url{https://github.com/g1910/HyperNetworks}},
#   year={2018}
# }


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


#aim of this HyperNetwork

#take the embedding of u,i,j
#objective function = same as the adversarial training, to miminize the worst case cross entropy loss

# how should we proceed with model architecture


class HyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim # size of hidden network
        self.f_size = f_size #f x f = size of filter
        self.out_size = out_size #n out
        self.in_size = in_size # n in

        # this is equivalent to the final layer of network proposed in page 3, linear operation of input vector ai
        # Wout = fsize * Nout fsize * d, now no = dim, col no = f.Nout.f

        #Parameter  The need to cache a Variable instead of having it automatically register as a parameter to the model is why we have an explicit way of registering parameters to our model i.e. nn.Parameter class.
        # with this, we can feed param to the optimizer easily
        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(),2))


        # this is input matrix Wi, with Wi has dimension d x in_size

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))

    def forward(self, z):

        #forward pass weight * zj (where z is layer embedding) plus bias (this is equivalent to aij in page 3 of hypernet paper)
        h_in = torch.matmul(z, self.w2) + self.b2
        #shape h_in to insize * z_dim dimension
        h_in = h_in.view(self.in_size, self.z_dim)


        #h_final take the input of h_in and weight and bias finale generated above
        h_final = torch.matmul(h_in, self.w1) + self.b1

        #kernal is a reshaped version of h_final computed above (baiscally it's a filter or feature detector)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
