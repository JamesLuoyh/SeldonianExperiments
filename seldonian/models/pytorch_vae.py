import torch
from torch.nn import Sequential, Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid, Tanh
from .pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli, Categorical

import torch.nn.functional as F
class PytorchVFAE(SupervisedPytorchBaseModel):
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
                 mi_version,
                 activation=ReLU(),
                 ):
        self.vfae = VariationalFairAutoEncoder(x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 mi_version,
                 activation=ReLU())
        self.discriminator = DecoderMLP(z_dim, z_dim, s_dim, activation).to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=alpha_adv)
        self.s_dim = s_dim
        self.mi_version = mi_version
        return self.vfae

    # set a prior distribution for the sensitive attribute for VAE case
    def set_pu(self, pu):
        if len(pu) == 1:
            pu_dist = Bernoulli(probs=torch.tensor(pu).to(self.device))
        else:
            pu_dist = Categorical(probs=torch.tensor(pu).to(self.device))
        self.vfae.set_pu(pu_dist)
        return

    def get_representations(self, X):
        return self.vfae.get_representations(X)

class VariationalFairAutoEncoder(Module):
    """
    Implementation of the Variational Fair AutoEncoder. Note that the loss has to be computed separately.
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
                 mi_version,
                 activation=ReLU()):
        super().__init__()
        # self.y_out_dim = 2 if y_dim == 1 else y_dim
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
        self.loss = VFAELoss()
        self.mi_version = mi_version



    def set_pu(self, pu):
        self.pu = pu
        return

    def get_representations(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)
        return z1_encoded

    def forward(self, inputs, discriminator):
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
        x_s = self.dropout(x_s)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)

        # z1_y = torch.cat([z1_encoded, y], dim=1)
        # z2_encoded, z2_enc_logvar, z2_enc_mu = self.encoder_z2(z1_y)

        # # decode
        # z2_y = torch.cat([z2_encoded, y], dim=1)
        # z1_decoded, z1_dec_logvar, z1_dec_mu = self.decoder_z1(z2_y)

        z1_s = torch.cat([z1_encoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        y_decoded = self.decoder_y(z1_encoded)
        s_decoded = discriminator(z1_encoded)
        if self.s_dim == 1:
            p_adversarial = Bernoulli(probs=s_decoded)
        else:
            p_adversarial = Categorical(probs=s_decoded)
            s = torch.argmax(s, dim=1)
        log_p_adv = p_adversarial.log_prob(s)
        log_p_u = self.pu.log_prob(s)
        if self.mi_version == 2:
            self.mi_sz = log_p_adv - log_p_u
        # u_ = self.pu.sample()
        # zs = u_.mean()
        # log_p_adv - (u_ * torch.log(zs) + (1.0 - u_) * torch.log(1.0 - zs))
        elif self.mi_version == 1:
            self.mi_sz = -0.5 * torch.sum(1 + z1_enc_logvar - z1_enc_mu ** 2 - z1_enc_logvar.exp(), dim = 1)
        # print(self.mi_version, self.mi_sz.mean())
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
        # self.encoder = 
        # self.encoder_2 = Linear(hidden_dim, hidden_dim)
        self.activation = activation
        self.encoder = Sequential(
          Linear(in_features, hidden_dim),
          self.activation,
        #   Linear(hidden_dim, hidden_dim),
        #   self.activation,
        #   nn.Linear(self.hidden_dim, self.hidden_dim),
        #   nn.ReLU()
        )

        self.logvar_encoder = Linear(hidden_dim, z_dim)
        # self.mu_encoder = Linear(hidden_dim, z_dim)

        self.mu_encoder = Linear(hidden_dim, z_dim)
        # self.tanh = Tanh()

    def forward(self, inputs):
        """
        :param inputs:
        :return:
            - z - the latent sample
            - logvar - variance of the distribution over z
            - mu - mean of the distribution over z
        """
        x = self.encoder(inputs)
        logvar = self.logvar_encoder(x)
        sigma = torch.exp(0.5 * logvar) #torch.sqrt(torch.exp(logvar))
        mu = self.mu_encoder(x)

        # reparameterization trick: we draw a random z
        epsilon = torch.randn_like(mu)
        # epsilon = torch.normal(mean=0, std=1, size=sigma.shape).to(mu.get_device())
        z = sigma * epsilon + mu # torch.randn_like(mu) epsilon * mu + logvar #
        return z, logvar, mu

# class VariationalMLP(Module):
#     """
#     Single hidden layer MLP using the reparameterization trick for sampling a latent z.
#     """

#     def __init__(self, in_features, hidden_dim, z_dim, activation):
#         super().__init__()
#         self.encoder = Linear(in_features, hidden_dim)
#         self.activation = activation

#         self.logvar_encoder = Linear(hidden_dim, z_dim)
#         self.mu_encoder = Linear(hidden_dim, z_dim)

#     def forward(self, inputs):
#         """
#         :param inputs:
#         :return:
#             - z - the latent sample
#             - logvar - variance of the distribution over z
#             - mu - mean of the distribution over z
#         """
#         x = self.encoder(inputs)
#         logvar = (0.5 * self.logvar_encoder(x)).exp()
#         mu = self.mu_encoder(x)

#         # reparameterization trick: we draw a random z
#         epsilon = torch.randn_like(mu)
#         z = epsilon * mu + logvar
#         return z, logvar, mu

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

        self.bce = BCELoss(reduce='mean')
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
        loss *= 0.1
        # print("loss" Zh,loss)
        # print(reconstruction_loss, kl_loss_z1)
        loss += self.alpha * supervised_loss
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


class FastMMD(Module):
    """ Fast Maximum Mean Discrepancy approximated using the random kitchen sinks method.
    """

    def __init__(self, out_features, gamma):
        super().__init__()
        self.gamma = gamma
        self.out_features = out_features

    def forward(self, a, b):
        in_features = a.shape[-1]

        # W sampled from normal
        w_rand = torch.randn((in_features, self.out_features), device=a.device)
        # b sampled from uniform
        b_rand = torch.zeros((self.out_features,), device=a.device).uniform_(0, 2 * pi)

        phi_a = self._phi(a, w_rand, b_rand).mean(dim=0)
        phi_b = self._phi(b, w_rand, b_rand).mean(dim=0)
        mmd = torch.norm(phi_a - phi_b, 2)

        return mmd

    def _phi(self, x, w, b):
        scale_a = sqrt(2 / self.out_features)
        scale_b = sqrt(2 / self.gamma)
        out = scale_a * (scale_b * (x @ w + b)).cos()
        return out
