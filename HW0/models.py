"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
        

class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
        self.embedding_dim = embedding_dim
        self.userEmbedder = ScaledEmbedding(num_users,self.embedding_dim)
        self.itemEmbedder = ScaledEmbedding(num_items,self.embedding_dim)
        self.userbias = ZeroEmbedding(num_users,1)
        self.itembias = ZeroEmbedding(num_items,1)
        self.numUsers = num_users
        self.numItems = num_items
        self.hidden = nn.Linear(layer_sizes[0],layer_sizes[1])
        self.out = nn.Linear(layer_sizes[1],1)


    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        item_ids_ = torch.randint(1,self.numItems+1, (user_ids.size(0),))
        userEmbedded = self.userEmbedder(user_ids)
        itemEmbedded = self.itemEmbedder(item_ids)
        user_bias = self.userbias(user_ids).reshape(user_ids.size(0))
        item_bias = self.itembias(item_ids).reshape(user_ids.size(0))
        elemwiseProd = userEmbedded*itemEmbedded
        pij = torch.sum(elemwiseProd,dim=1) + user_bias + item_bias
        inp = torch.cat((userEmbedded,itemEmbedded,elemwiseProd),dim=1)
        out = self.hidden(inp)
        out = self.out(out)
        out = out.reshape(user_ids.size(0))
        ## Make sure you return predictions and scores of shape (batch,)
        # if (len(predictions.shape) > 1) or (len(score.shape) > 1):
        #     raise ValueError("Check your shapes!")
        
        return pij, out