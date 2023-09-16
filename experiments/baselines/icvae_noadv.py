import torch
from torch.nn import Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid, Sequential
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import experiments.utils as utils
    
class PytorchICVAEBaseline(SupervisedPytorchBaseModel):
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
                 lr,
                 use_validation=True,
                 activation=ReLU(),
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
        self.vfae = InvariantConditionalVariationalAutoEncoder(x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 activation=ReLU())
        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=1e-3)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.use_validation = use_validation
        return self.vfae
    
    def train(self, X_train, Y_train, batch_size, num_epochs,data_frac):
        print("Training model...")
        loss_list = []
        accuracy_list = []
        iter_list = []
        x_train_tensor = torch.from_numpy(X_train)
        y_train_label = torch.from_numpy(Y_train)
        train = torch.utils.data.TensorDataset(x_train_tensor, y_train_label)
        if self.use_validation:
            X_train_size = int(len(X_train) * 0.8)
            X_valid = X_train[X_train_size:]
            Y_valid = Y_train[X_train_size:]
            x_train_tensor = torch.from_numpy(X_train[:X_train_size])
            y_train_label = torch.from_numpy(Y_train[:X_train_size])
            x_valid_tensor = torch.from_numpy(X_valid)
            y_valid_label = torch.from_numpy(Y_valid)
        else:
            x_train_tensor = torch.from_numpy(X_train)
            y_train_label = torch.from_numpy(Y_train)
            x_valid_tensor = x_train_tensor
            y_valid_label = y_train_label
        train = torch.utils.data.TensorDataset(x_train_tensor, y_train_label)

        trainloader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True
        )
        print(
            f"Running gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
        )
        num_epochs_l = [int(60/data_frac)]#,30,60,90]#10,20]#30,60,90]
        lr_l = [1e-5]#, 1e-4, 1e-3]
        lam_l = [0.1]#, 1, 10]#np.logspace(-1,0,3)
        for lr in lr_l:
            for lam in lam_l:
                for num_epochs in num_epochs_l:
                    self.vfae.lam = lam
                    self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=lr)
                    itot = 0
                    for epoch in range(num_epochs):
                        for i, (features, labels) in enumerate(trainloader):
                            # Load images
                            features = features.float().to(self.device)
                            labels = labels.to(self.device)

                            # Clear gradients w.r.t. parameters
                            self.optimizer.zero_grad()
                            # Forward pass to get output/logits
                            vae_loss, mi_sz, y_prob = self.pytorch_model(features)

                            # Getting gradients w.r.t. parameters
                        
                            vae_loss.backward()

                            # Updating parameters
                            self.optimizer.step()
                            if i % 100 == 0:
                                it = f"{i+1}/{len(trainloader)}"
                                print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{vae_loss}")
                            itot += 1
                    # evaluate training data
                    if self.use_validation:
                        self.vfae.eval()
                        self.pytorch_model.eval()
                        kwargs = {
                            'downstream_lr'     : 1e-4,
                            'downstream_bs'     : 500,
                            'downstream_epochs' : 5,
                            'y_dim'             : 1,
                            's_dim'             : self.s_dim,
                            'z_dim'             : self.z_dim,
                            'device'            : self.device,
                            'X'                 : x_valid_tensor.numpy(),
                        }
                        y_pred = utils.unsupervised_downstream_predictions(self, self.get_model_params(), x_train_tensor.numpy(), y_train_label.numpy(), x_valid_tensor.numpy(), **kwargs)
                        x_valid_tensor = x_valid_tensor.float().to(self.device)

                        vae_loss, mi_sz, y_prob = self.pytorch_model(x_valid_tensor)
                        
                        y_pred_all = vae_loss, mi_sz, y_pred
                        delta_DP = utils.demographic_parity(y_pred_all, None, **kwargs)
                        # delta_DP = self.demographic_parity(self.vfae.y_prob, x_valid_tensor[:, self.x_dim:self.x_dim+self.s_dim])
                        auc = roc_auc_score(y_valid_label.numpy(), y_pred)
                        
                        df = pd.read_csv('./SeldonianExperimentResults/icvae.csv')
                        row = {'data_frac':data_frac, 'delta_dp': delta_DP, 'auc': auc, 'lr': lr, 'lam': lam, 'epoch': num_epochs}
                        print(row)
                        df = df.append(row, ignore_index=True)
                        df.to_csv('./SeldonianExperimentResults/icvae.csv', index=False)


    @staticmethod
    def demographic_parity(y_prob, s):
        g, uc = np.zeros([2]), np.zeros([2])
        y_ = (y_prob > 0.5).float()
        for i in range(s.shape[0]):
            if s[i] > 0:
                g[1] += y_[i].item()
                uc[1] += 1
            else:
                g[0] += y_[i].item()
                uc[0] += 1
        g = g / uc
        return np.abs(g[0] - g[1])


    def get_representations(self, X):
        return self.vfae.get_representations(X)

class InvariantConditionalVariationalAutoEncoder(Module):
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
                 activation=ReLU()):
        super().__init__()
        self.y_out_dim = y_dim #2 if y_dim == 1 else y_dim
        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, self.y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim, activation)

        self.dropout = Dropout(dropout_rate)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.loss = VFAELoss()
        self.reconstruct_loss = BCELoss(reduce=False)
        self.lam = 0

    #KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) + 
    #  (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
    #  \log( \frac{\det \sigma_1}{\det \sigma_0} )
    @staticmethod
    def all_pairs_gaussian_kl(mu, sigma, add_third_term=False):
        sigma_sq = sigma.square() + 1e-8
        sigma_sq_inv = torch.reciprocal(sigma_sq)

        #dot product of all sigma_inv vectors with sigma is the same as a matrix mult of diag
        first_term = torch.matmul(sigma_sq, sigma_sq_inv.T)
        r = torch.matmul(mu * mu,sigma_sq_inv.T)
        r2 = mu * mu * sigma_sq_inv 
        r2 = torch.sum(r2,1)
        #squared distance
        #(mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
        #uses broadcasting
        second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv).T)
        second_term = r - second_term + (r2.unsqueeze(1)).T
        # log det A = tr log A
        # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
        #   \tr\log \Sigma_1 - \tr\log \Sigma_0 
        # for each sample, we have B comparisons to B other samples...
        #   so this cancels out

        if(add_third_term):
            r = torch.sum(torch.log(sigma_sq),1)
            r = torch.reshape(r,[-1,1])
            third_term = r - r.T
        else:
            third_term = 0

        #- tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)))\
        # the dim_z ** 3 term comes fro
        #   -the k in the original expression
        #   -this happening k times in for each sample
        #   -this happening for k samples
        #return 0.5 * ( first_term + second_term + third_term - dim_z )
        return 0.5 * ( first_term + second_term + third_term )

    #
    # kl_conditional_and_marg
    #   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
    #

    #def kl_conditional_and_marg(args):
    def kl_conditional_and_marg(self, z_mean, z_log_sigma_sq, dim_z):
        z_sigma = ( 0.5 * z_log_sigma_sq ).exp()
        all_pairs_GKL = self.all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5*dim_z
        return torch.mean(all_pairs_GKL, dim=1)

    def get_representations(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)
        return z1_encoded

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

        self.mi_sz = self.kl_conditional_and_marg(z1_enc_mu, z1_enc_logvar, self.z_dim)
        # reconstruct_loss = torch.mean(self.reconstruct_loss(x_decoded, x_s), dim=1)
        reconstruct_loss = F.binary_cross_entropy(x_decoded, x, reduction='sum') / len(x)
        self.mi_sz += 0.1 * reconstruct_loss
        self.vae_loss += self.lam * self.mi_sz.mean()
        # print(torch.softmax(y_decoded, dim=-1))
        self.pred = y_decoded # torch.softmax(y_decoded, dim=-1)[:, 1]
        self.s = s
        self.z = z1_encoded
        self.y_prob = y_decoded.squeeze()
        return self.vae_loss, self.mi_sz, self.y_prob

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
        logvar = self.logvar_encoder(x)
        sigma = torch.sqrt(torch.exp(logvar))
        mu = self.mu_encoder(x)

        # reparameterization trick: we draw a random z
        # epsilon = torch.randn_like(mu)
        z = sigma * torch.randn_like(mu) + mu
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
        # reconstruction_loss = self.bce(y_pred['x_decoded'], x_s)
        recons_loss = F.binary_cross_entropy(y_pred['x_decoded'], x, reduction='sum')

        # supervised_loss = self.bce(prediction['y_decoded'], y.unsqueeze(1))
        # print(supervised_loss)
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

        loss = 0.1 * (recons_loss + kl_loss_z1) / len(y)

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
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b)**2) / logvar_b.exp()
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
