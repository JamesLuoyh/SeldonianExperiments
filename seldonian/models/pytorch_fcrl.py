import torch
from torch.nn import Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid, Sequential, Parameter
from .pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli
from torch.nn.functional import softplus
import torch.nn.functional as F


class PytorchFCRL(SupervisedPytorchBaseModel):
    """
    Implementation of the Variational Fair AutoEncoder. Note that the loss has to be computed separately.
    """
    def __init__(self,device, **kwargs):
      """ Implements an example CNN with PyTorch. 
      CNN consists of two hidden layers followed 
      by a linear + softmax output layer 

      :param device: The torch device, e.g., 
        "cuda" (NVIDIA GPU), "cpu" for CPU only,
        "mps" (Mac M1 GPU)
      """
      super().__init__(device, **kwargs)

    # def create_model(self,**kwargs):
    def create_model(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 alpha_adv,
                 activation=ReLU(),
                 s_num=2,
                 nce_size=50,
                 ):
    #     super().__init__()
        # y_dim = kwargs[y_dim]
        # s_dim = kwargs[s_dim]
        # x_dim = kwargs[x_dim]
        # z_dim = kwargs[z_dim]
        # z1_enc_dim = kwargs[z1_enc_dim]
        # activation = kwargs[activation]
        # z2_enc_dim = kwargs[z2_enc_dim]
        # z1_dec_dim = kwargs[z1_dec_dim]
        # x_dec_dim = kwargs[x_dec_dim]
        # dropout_rate = kwargs[dropout_rate]
        self.fcrl = ContrastiveVariationalAutoEncoder(x_dim,
                s_dim,
                y_dim,
                z1_enc_dim,
                z2_enc_dim,
                z1_dec_dim,
                x_dec_dim,
                z_dim,
                dropout_rate,
                activation=ReLU(),
                s_num=s_num,
                nce_size=nce_size,
                device=self.device)
        self.s_dim = s_dim
        return self.fcrl

class ContrastiveVariationalAutoEncoder(Module):
    """
    Implementation of the Variational Fair AutoEncoder
    """

    def __init__(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 activation=ReLU(),
                 s_num=2,
                 nce_size=50,
                 device='cpu'):
        super().__init__()
        self.y_out_dim = y_dim #2 if y_dim == 1 else y_dim
        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, self.y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim + s_dim, activation)

        self.dropout = Dropout(dropout_rate)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_num = s_num
        self.loss = VFAELoss()
        self._nce_estimator = CPC([x_dim], z_dim, s_num, nce_size, device)
        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device=device)
    
    @staticmethod
    def kl_gaussian(logvar_a, mu_a):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples. https://arxiv.org/abs/1405.2664

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :return: kl divergence
        """
        per_example_kl = - logvar_a - 1 + (logvar_a.exp() + (mu_a).square())
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl

    def forward(self, inputs):
        """
        :param inputs: dict containing inputs: {'x': x, 's': s, 'y': y} where x is the input feature vector, s the
        sensitive variable and y the target label.
        :return: dict containing all 8 VFAE outputs that are needed for computing the loss term, i.e. :
            - x_decoded: the reconstructed input with shape(x_decoded) = shape(concat(x, s))
            - y_decoded: the predictive posterior output for target label y
            - z1_encoded: the sample from latent variable z1
            - z1_enc_logvar: variance of the z1 encoder distribution
            - z1_enc_mu: mean of the z1 encoder distribution
            - z2_enc_logvar: variance of the z2 encoder distribution
            - z2_enc_mu: mean of the z2 encoder distribution
            - z1_dec_logvar: variance of the z1 decoder distribution
            - z1_dec_mu: mean of the z1 decoder distribution
        """
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(self.dropout(x_s))

        # z1_y = torch.cat([z1_encoded, y], dim=1)
        # z2_encoded, z2_enc_logvar, z2_enc_mu = self.encoder_z2(z1_y)

        # # decode
        # z2_y = torch.cat([z2_encoded, y], dim=1)
        # z1_decoded, z1_dec_logvar, z1_dec_mu = self.decoder_z1(z2_y)

        z1_s = torch.cat([z1_encoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        y_decoded = self.decoder_y(z1_encoded)        
   
        
        # print(self.mi_sz)
        outputs = {
            # predictive outputs
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,
            'z1_encoded': z1_encoded,

            # outputs for regularization loss terms
            'z1_enc_logvar': z1_enc_logvar,
            'z1_enc_mu': z1_enc_mu,

            # 'z2_enc_logvar': z2_enc_logvar,
            # 'z2_enc_mu': z2_enc_mu,

            # 'z1_dec_logvar': z1_dec_logvar,
            # 'z1_dec_mu': z1_dec_mu
        }
        # will return the constraint C2 term. log(qu) - log(pu) instead of y_decoded
        self.vae_loss = self.loss(outputs, {'x': x, 's': s, 'y': y})
        
        kl_gaussian = self.kl_gaussian(z1_enc_logvar, z1_enc_mu)
        nce_estimate = self._nce_estimator(x, s, z1_encoded)
        self.mi_sz = kl_gaussian - nce_estimate
        self.mi_sz = self.mi_sz.unsqueeze(1)
        print(self.mi_sz.mean())
        # print(torch.softmax(y_decoded, dim=-1))
        self.pred = y_decoded # torch.softmax(y_decoded, dim=-1)[:, 1]
        self.s = s
        self.z = z1_encoded
        self.y_prob = y_decoded.squeeze()
        return self.vae_loss, self.mi_sz, self.y_prob
    
class VariationalMLP(Module):
    """
    Single hidden layer MLP using the reparameterization trick for sampling a latent z.
    """

    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.encoder = Linear(in_features, hidden_dim)
        self.activation = activation

        self.logvar_encoder = Linear(hidden_dim, z_dim)
        self.mu_encoder = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        """
        :param inputs:
        :return:
            - z - the latent sample
            - logvar - variance of the distribution over z
            - mu - mean of the distribution over z
        """
        x = self.encoder(inputs)
        logvar = (0.5 * self.logvar_encoder(x)).exp()
        mu = self.mu_encoder(x)

        # reparameterization trick: we draw a random z
        epsilon = torch.randn_like(mu)
        z = epsilon * mu + logvar
        return z, logvar, mu


class DecoderMLP(Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = Linear(hidden_dim, latent_dim)
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.sigmoid(self.lin_out(x))


class VFAELoss(Module):
    """
    Loss function for training the Variational Fair Auto Encoder.
    """

    def __init__(self, alpha=1.0, beta=0.0, mmd_dim=0, mmd_gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = BCELoss()
        self.ce = CrossEntropyLoss()
        # self.mmd = FastMMD(mmd_dim, mmd_gamma)

    def forward(self, y_pred, y_true):
        """

        :param y_pred: dict containing the vfae outputs
        :param y_true: dict of ground truth labels for x, s and y
        :return: the loss value as Tensor
        """
        x, s, y = y_true['x'], y_true['s'], y_true['y']
        x_s = torch.cat([x, s], dim=-1)
        device = y.device
        supervised_loss = self.bce(y_pred['y_decoded'], y.to(device))
        reconstruction_loss = F.binary_cross_entropy(y_pred['x_decoded'], x_s, reduction='sum')
        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss_z1 = self._kl_gaussian(y_pred['z1_enc_logvar'],
                                       y_pred['z1_enc_mu'],
                                       zeros,
                                       zeros)

        # # becomes kl between z2 and a standard normal when passing zeros

        # kl_loss_z2 = self._kl_gaussian(y_pred['z2_enc_logvar'],
        #                                y_pred['z2_enc_mu'],
        #                                zeros,
        #                                zeros)

        loss = reconstruction_loss + kl_loss_z1
        loss /= len(y)
        # loss = self.alpha * supervised_loss

        # compute mmd only if protected and non protected in batch
        # z1_enc = y_pred['z1_encoded']
        # z1_protected, z1_non_protected = self._separate_protected(z1_enc, s)
        # if len(z1_protected) > 0:
        #     loss += self.beta * self.mmd(z1_protected, z1_non_protected)
        return loss

    @staticmethod
    def _kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples. https://arxiv.org/abs/1405.2664

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :param logvar_b: standard deviation b
        :param mu_b: mean b
        :return: kl divergence, mean averaged over batch dimension.
        """
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.sum()

    @staticmethod
    def _separate_protected(batch, s):
        """separate batch based on labels indicating protected and non protected .

        :param batch: values to select from based on s.
        :param s: tensor of labels with s=1 meaning protected and s=0 non protected.
        :return:
            - protected - items from batch with protected label
            - non_protected - items from batch with non protected label
        """
        idx_protected = (s == 1).nonzero()[:, 0]
        idx_non_protected = (s == 0).nonzero()[:, 0]
        protected = batch[idx_protected]
        non_protected = batch[idx_non_protected]

        return protected, non_protected

class CPC(Module):
    def __init__(self, input_size, z_size, c_size, hidden_size, device):
        super().__init__()

        self.f_x = Sequential(
            Linear(input_size[0], hidden_size),
            ReLU(),
            Linear(hidden_size, z_size),
        )

        # just make one transform
        self.f_z = Sequential(Linear(z_size, z_size))
        self.w_s = Parameter(data=torch.randn(c_size, z_size, z_size))
        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device=device)

    def forward(self, x, c, z):
        N = x.shape[0]
        c = c.long()
        f_x = self.f_x(x)
        f_z = self.f_z(z)
        temp = torch.bmm(
            torch.bmm(
                f_x.unsqueeze(2).transpose(1, 2), self.w_s[c.reshape(-1)]
            ),
            f_z.unsqueeze(2),
        )
        T = softplus(temp.view(-1))

        neg_T = torch.zeros(N, device=self.device)

        for cat in set(c.reshape(-1).tolist()):
            f_z_given_c = f_z[(c == cat).reshape(-1)]
            f_x_given_c = f_x[(c == cat).reshape(-1)]

            # (N,Z) X (Z,Z)
            temp = softplus(
                f_x_given_c @ self.w_s[cat] @ f_z_given_c.transpose(0, 1)
            )
            # columns are different Z's, rows are different x's.
            # mean along dim 0, is the mean over the same Z different X
            # mean along dim 1, is the mean over the same X different Z
            # Change to sum because the contrastive estimation model overfit
            # for the Z corresponding to the X easily. When Z and X does not match
            # the T evaluated is almost 0 and the average is also 0.
            # This will make MI(Z;X|S) very big and the MI(Z;S) becomes negative.
            neg_T[(c == cat).reshape(-1)] = temp.mean(dim=1).view(-1)

        return torch.log(T + 1e-16) - torch.log(neg_T + 1e-16)
