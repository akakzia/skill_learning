import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpi_utils.mpi_utils import sync_grads
from vae_models.data_buffer import DataBuffer

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ContextVAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.inner_sizes = args.layer_sizes
        self.state_size = args.env_params['goal']
        self.embedding_size = args.embedding_size
        self.latent_size = args.latent_size
        self.scaling_factor = args.scaling_factor


        encoder_layer_sizes = [self.state_size + self.embedding_size] + self.inner_sizes
        decoder_layer_sizes = [self.latent_size + self.embedding_size] + self.inner_sizes + [self.state_size]
        self.encoder = Encoder(encoder_layer_sizes, self.latent_size)
        self.decoder = Decoder(decoder_layer_sizes)

        self.buffer = DataBuffer(args.env_params, args.vae_buffer_size)
        self.batch_size = args.vae_batch_size
        self.learning_rate = args.learning_rate

        self.k_param = args.k_param

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Bounds of discovered goal space
        self.upper_bounds = np.zeros(self.args.env_params['goal'])
        self.lower_bounds = np.ones(self.args.env_params['goal'])


    def forward(self, states, embeddings):

        batch_size = states.size(0)
        assert states.size(0) == embeddings.size(0)

        # means, log_var = self.encoder(torch.cat((initial_s, embeddings, current_s), dim=1))
        means, log_var = self.encoder(torch.cat([states, embeddings], dim=-1))

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        if self.args.cuda:
            eps = eps.cuda()
        z = eps * std + means

        recon_x = self.decoder(torch.cat((z, embeddings), dim=1))

        return recon_x, means, log_var, z

    def inference(self,embeddings, n=1):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        embeddings = torch.Tensor(embeddings)

        if self.args.cuda:
            z = z.cuda()
            embeddings = embeddings.cuda()

        recon_state = self.decoder(torch.cat((z, embeddings), dim=1))

        # Scale down reconstructions
        recon_state = recon_state / self.scaling_factor

        return recon_state
    
    def store(self, goal_batch):
        # Store bounds
        batch_upper_bounds = np.max(goal_batch, axis=0)
        batch_lower_bounds = np.min(goal_batch, axis=0)
        self.upper_bounds = np.maximum(batch_upper_bounds, self.upper_bounds)
        self.lower_bounds = np.minimum(batch_lower_bounds, self.lower_bounds)
        # Clip to avoid falling objects 
        self.upper_bounds = np.clip(self.upper_bounds, a_min=0, a_max=np.array([0.25, 0.25, 0.25]))
        # Store episodes in buffer
        self.buffer.store_data(data_batch=goal_batch)
    
    def train(self, states):
        def loss_fn(recon_x, x, mean, log_var):
            MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (MSE + self.k_param * KLD) / x.size(0), MSE / x.size(0), KLD / x.size(0)
        states_tensor = torch.Tensor(states * self.scaling_factor)
        embeddings = torch.zeros([self.batch_size, 3])
        if self.args.cuda:
            states_tensor = states_tensor.cuda()
            embeddings = embeddings.cuda()

        recon_x, means, log_var, z = self.forward(states_tensor, embeddings)
        # Compute loss 
        loss, loss_mse, loss_kld = loss_fn(recon_x, states_tensor, means, log_var)
        self.optimizer.zero_grad()
        loss.backward()
        sync_grads(self)
        self.optimizer.step()

        return loss.item(), loss_mse.item(), loss_kld.item()
    
    def save(self, model_path, epoch):
        # Save buffer and parameters
        torch.save([self.buffer.buffer, self.buffer.current_size, self.encoder.state_dict(), self.decoder.state_dict()],
                    model_path + '/vae_model_{}.pt'.format(epoch))

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()
        input_size = layer_sizes[0]
        hidden_size = layer_sizes[1]
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.MLP = nn.Sequential()
        # for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(hidden_size, latent_size)
        self.linear_log_var = nn.Linear(hidden_size, latent_size)

        self.apply(weights_init_)
    def forward(self, x):

        # x = self.MLP(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes):

        super().__init__()
        input_size = layer_sizes[0]
        hidden_size = layer_sizes[1]
        output_size = layer_sizes[-1]
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.apply(weights_init_)
        # self.MLP = nn.Sequential()

        # for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     if i + 2 < len(layer_sizes):
        #         self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())


    def forward(self, z):

        # x = self.MLP(z)
        x = F.relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x