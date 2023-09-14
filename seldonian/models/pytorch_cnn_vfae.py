from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn
import torch
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F


class PytorchFacialVAE(SupervisedPytorchBaseModel):
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
                 activation=nn.ReLU(),
                 ):
        self.vfae = FacialVAE(x_dim,
                s_dim,
                y_dim,
                z1_enc_dim,
                z2_enc_dim,
                z1_dec_dim,
                x_dec_dim,
                z_dim,
                dropout_rate,
                mi_version,
                activation=nn.ReLU())
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

class FacialVAE(nn.Module):
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
                activation=nn.ReLU()):
        super(FacialVAE, self).__init__()
        self.latent_dim = z_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        modules = []
        in_channels = 1
        # if hidden_dims is None:
        self.hidden_dims = [16, 32, 64, 128]
        self.mi_version = mi_version
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*9, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*9, self.latent_dim)

        # Build Decoder
        modules = []

        # latent_dim + 1 for adding the sensitve attribute
        self.decoder_input = nn.Linear(self.latent_dim + self.s_dim, self.hidden_dims[-1] * 9)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.decoder_y = DecoderMLPBinary(z_dim, x_dec_dim, 1, activation)
        self.bce = nn.BCELoss()

    def set_pu(self, pu):
        self.pu = pu
        return

    def get_representations(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 3, 3)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, sensitive_attributes, labels, discriminator):
        x = input
        s = sensitive_attributes
        y = labels
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z_s = torch.cat([z, s], dim=1)

        s_decoded = discriminator(z)
        p_adversarial = Categorical(probs=s_decoded)
        s = torch.argmax(s, dim=1)
        log_p_adv = p_adversarial.log_prob(s)
        log_p_u = self.pu.log_prob(s)
        # print("log_p_adv", log_p_adv)
        # print("log_p_u", log_p_u)
        if self.mi_version == 2:
            self.mi_sz = log_p_adv - log_p_u
        elif self.mi_version == 1:
            self.mi_sz = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        x_decoded = self.decode(z_s)
        y_decoded = self.decoder_y(z)
        # print(self.mi_sz)
        outputs = {
            # predictive outputs
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,
            'z1_encoded': z,

            # outputs for regularization loss terms
            'z1_enc_logvar': log_var,
            'z1_enc_mu': mu,

            # 'z2_enc_logvar': z2_enc_logvar,
            # 'z2_enc_mu': z2_enc_mu,

            # 'z1_dec_logvar': z1_dec_logvar,
            # 'z1_dec_mu': z1_dec_mu
        }
        # will return the constraint C2 term. log(qu) - log(pu) instead of y_decoded
        self.vae_loss = self.loss_function(outputs, {'x': x, 's': s, 'y': y})
        # print(torch.softmax(y_decoded, dim=-1))
        self.mi_sz = self.mi_sz.flatten()
        self.pred = y_decoded # torch.softmax(y_decoded, dim=-1)[:, 1]
        self.s = s
        self.z = z
        self.y_prob = y_decoded.squeeze()
        return self.vae_loss, self.mi_sz, self.y_prob

    def loss_function(self, prediction, actual):
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log \frac{1}{sigma} + \frac{sigma^2 + mu^2}{2} - \frac{1}{2}
        :return:
        """
        recons = prediction['x_decoded']
        input = actual['x']
        y = actual['y']
        mu = prediction['z1_enc_mu']
        log_var = prediction['z1_enc_logvar']

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input, reduction='sum')


        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        supervised_loss = self.bce(prediction['y_decoded'], y.unsqueeze(1))
        # print(supervised_loss)
        loss = 0.1 * (recons_loss + kld_loss) / len(y)
        # loss = 0.1 * loss + 10*supervised_loss
        return loss #{'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class DecoderMLP(nn.Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = nn.Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = nn.Linear(hidden_dim, latent_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.softmax(self.lin_out(x))

class DecoderMLPBinary(nn.Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = nn.Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_encoder_2 = nn.Linear(in_features, hidden_dim//2)
        self.lin_out = nn.Linear(hidden_dim//2, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        x = self.activation(self.lin_encoder_2(x))
        return self.sigmoid(self.lin_out(x))