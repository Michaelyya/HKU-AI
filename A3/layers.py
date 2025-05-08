import torch
import torch.nn as nn
import math

"""
Allowed functions/operation:

Basic arithmetic operations: 
    - Creation Ops: https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
    - Indexing, Slicing, Joining, Mutating Ops: https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops

Some advanced functions:
    - nn.functional.unfold: https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html
    - torch.einsum: https://pytorch.org/docs/stable/generated/torch.einsum.html
"""

########################################## DECLARE #####################################################
# You must declare with ONE of the following statements if you have used any GenAI tools:              #
#   - I did not use any AI technologies in preparing or writing up this assignment.                    #
#   - I acknowledge the use of <insert AI system(s) and link> to generate initial ideas for            #
#       background research in the drafting of this assignment.                                        #
#   - I acknowledge the use of <insert AI system(s) and link> to generate materials that were          #
#       included within my final assignment in its modified form.                                      #
# e.g.                                                                                                 #
#   I acknowledge the use of ChatGPT <https://chatgpt.hku.hk/> to generate initial math formula        #
#   for convolution. I then use the                                                                    #
#                                                                                                      #
# If you have used GenAI tool(s), you must (i) name the tool(s), (ii) describe how it/they were used,  #
# AND (iii) reference the tool(s):                                                                     #
#                                                                                                      #
########################################################################################################

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Initialization code remains unchanged
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization remains unchanged
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Implement linear transformation y = xW^T + b
        Allowed functions/operations:
        - torch.einsum()
        - Tensor.view()
        - Basic arithmetic operations
        """
        ###########################################################################
        # TODO: Process input to produce output with shape:                       #
        # (..., out_features) where ... preserves input dimensions                #
        #                                                                         #
        # HINT: Consider efficient tensor operations for matrix multiplication    #
        # Student's implementation here                                           #
        ###########################################################################
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Initialization code remains unchanged
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization remains unchanged
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Implement 2D convolution using tensor operations
        Allowed functions/operations:
        - torch.Tensor.shape
        - Tensor.view() 
        - torch.einsum()
        - nn.functional.unfold()
        - Basic arithmetic operations
        """
        ################################################################################
        # TODO: Transform input using allowed operations to produce output with shape: #
        # (N, out_channels, H_out, W_out)                                              #
        #                                                                              #
        # HINT: Consider how to reshape the weight matrix and process unfolded patches #
        #                                                                              #
        # Student's implementation here                                                #
        ################################################################################
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x):
        """
        Implement 2D max pooling using tensor operations
        Allowed functions/operations:
        - torch.Tensor.shape
        - Tensor.view() 
        - Tensor.max()
        - nn.functional.unfold()
        """
        ###########################################################################
        # TODO: Process input to produce output with shape:                       #
        # (N, C, H_out, W_out)                                                    #
        #                                                                         #
        # HINT: Consider how to extract and process local windows                 #
        #                                                                         #
        # Student's implementation here                                           #
        ############################################################################
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Implement cross entropy loss with numerical stability
        Allowed functions/operations:
        - Tensor.max(), .exp(), .log(), .sum()
        - torch.gather()
        - Basic arithmetic operations
        - Reduction methods (mean(), sum())
        """
        ###########################################################################
        # TODO: Compute loss without using nn.CrossEntropyLoss                    #
        #                                                                         #
        # HINT: Consider numerical stability when working with exponents          #
        #                                                                         #
        # Student's implementation here                                           #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################