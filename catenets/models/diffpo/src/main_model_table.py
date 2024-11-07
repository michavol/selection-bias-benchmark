import numpy as np
import torch
import torch.nn as nn
from .diff_models_table import diff_CSDI
import yaml


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        # self.target_dim = target_dim #1
        self.target_dim = config["train"]["batch_size"] #8
        

        self.emb_time_dim = config["model"]["timeemb"] # 32
        self.emb_feature_dim = config["model"]["featureemb"] #32 # 16 

        self.is_unconditional = config["model"]["is_unconditional"] #0
        self.target_strategy = config["model"]["target_strategy"] #'random'

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        # self.emb_total_dim = self.emb_feature_dim

        self.cond_dim = config["diffusion"]["cond_dim"] #178
        self.mapping_noise = nn.Linear(2, self.cond_dim)

        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        # self.embed_layer = nn.Embedding(
        #     num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        # )
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask # observed_mask.shape (8,1,30)
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) # rand_for_mask.shape (8, 30)

        for i in range(len(observed_mask)): # len(observed_mask):8, for each batch / each row.
            sample_ratio = 0.5  # np.random.rand()  # missing ratio # original 0.8
            num_observed = observed_mask[i].sum().item() # 30
            num_masked = round(num_observed * sample_ratio) # 24
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape 
        side_info = cond_mask
        
        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        # In validation, perform T steps forward and backward.
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    # def calc_loss(
    #     self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    # ): # original transfer observed_mask, change to gt_mask
    def calc_loss(
        self, observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, propnet=None
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else: # for training
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        ## # print('observed_data.shape', observed_data.shape) #observed_data.shape torch.Size([8, 1, 182])
        noise = torch.randn_like(observed_data[:, :, 1:3]) # only want column 1,2
        ## # print('check noise.shape', noise.shape) #noise.shape torch.Size([8, 1, 2])
        noisy_data = (current_alpha**0.5) * observed_data[:, :, 1:3] + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        ## # print('noisy_data.shape', noisy_data.shape) #([8, 1, 2])
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        # # print('forward diffmodel')
        # predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        # %%%%%%%%%%%%% change side info into cond_obs %%%%%%%%%%%%%%%%%%
        a = observed_data[:,:,0].unsqueeze(2)
        ## print('a.shape', a.shape) # t.shape torch.Size([8, 1, 1])
        x = observed_data[:,:,5:]
        ## print('x.shape', x.shape) # x.shape torch.Size([8, 1, 177])
        cond_obs = torch.cat([a,x], dim=2)
        ## print('cond_obs.shape', cond_obs.shape) # cond_obs.shape torch.Size([8, 1, 178])

        noisy_target = self.mapping_noise(noisy_data) 
        diff_input = cond_obs + noisy_target

        # predicted = self.diffmodel(diff_input, side_info, t).to(self.device)
        predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)

         # %%%%%%%%%%%%% change side info into cond_obs %%%%%%%%%%%%%%%%%%

        # # print('predicted.shape', predicted.shape) # predicted.shape torch.Size([8, 2])
  
        target_mask = gt_mask - cond_mask  # compute loss only on factual y.
        target_mask = target_mask.squeeze(1)[:,1:3]
        # # print('target_mask.shape', target_mask.shape)

        # # print('noise.shape', noise.shape) # noise.shape torch.Size([8, 1, 2])
        noise = noise.squeeze(1)
        residual = (noise - predicted) * target_mask
        # # print('residual.shape', residual.shape)
        # residual = (noise - predicted) * gt_mask
        # # # print('residual.shape', residual.shape) # residual.shape torch.Size([8, 1, 182])
        num_eval = target_mask.sum()
        # num_eval = gt_mask.sum()


        #====================== Modify the loss ===============================
        # Compute the weights

        ## # print('observed_data.shape', observed_data.shape) # observed_data.shape torch.Size([8, 1, 182])

        x_batch = observed_data[:, :, 5:].squeeze() 
        ## # print('x_batch.shape', x_batch.shape) # x_batch.shape torch.Size([8, 177])
        t_batch = observed_data[:, :, 0].squeeze()
        ## # print('t_batch.shape', t_batch.shape) # t_batch.shape torch.Size([8])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        propnet = propnet.to(device)

        ## # print('propnet in calc_loss function', propnet)
        pi_hat = propnet.forward(x_batch.float())
        ## # print('pi_hat.shape', pi_hat.shape) # pi_hat.shape torch.Size([8, 2])

        # # # print('pi_hat', pi_hat)
        # # # print('t_batch', t_batch)
        # # # print('(t_batch / pi_hat[:, 1])', (t_batch / pi_hat[:, 1]))
        # # # print('((1 - t_batch) / pi_hat[:, 0])', (1 - t_batch) / pi_hat[:, 0])

        weights = (t_batch / pi_hat[:, 1]) + ((1 - t_batch) / pi_hat[:, 0])
        # # # print('weights.shape', weights.shape) # weights.shape torch.Size([8])
        # # # print('weight', weights)
        weights = weights.reshape(-1, 1, 1) 

        ## # print('residual.shape', residual.shape) # residual.shape torch.Size([8, 1, 182])

        loss = (weights * (residual ** 2)).sum() / (num_eval if num_eval > 0 else 1)
        ## # print('weighted loss', loss)

        #======================#======================#======================#======================
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            #cond_obs = (cond_mask * observed_data).unsqueeze(1) # t, x
            a = observed_data[:,:,0].unsqueeze(2)
            # # print('a.shape', a.shape) # t.shape torch.Size([8, 1, 1])
            x = observed_data[:,:,5:]
            # # print('x.shape', x.shape) # x.shape torch.Size([8, 1, 177])
            cond_obs = torch.cat([a,x], dim=2)
            # # print('cond_obs.shape', cond_obs.shape) # cond_obs.shape torch.Size([8, 1, 178])
            noisy_target = self.mapping_noise(noisy_data) 
            # print('noisy_target.shape', noisy_target.shape) # noisy_target.shape torch.Size([8, 1, 178])
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            total_input = cond_obs + noisy_target
            # # print('total_input', total_input.shape) # total_input torch.Size([8, 1, 178])
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        ## print('n_samples', n_samples) # n_samples 15??
        B, K, L = observed_data.shape 

        imputed_samples = torch.zeros(B, n_samples, 2).to(self.device) 

        for i in range(n_samples):
            generated_target = observed_data[:,:,1:3]
            # # print(generated_target.shape)

            current_sample = torch.randn_like(generated_target)
            # # print('current_sample.shape', current_sample.shape) 

            # perform T steps backward
            for t in range(self.num_steps - 1, -1, -1):
                ## print('Inside impute')
                # cond_obs = (cond_mask * observed_data).unsqueeze(1)
                # # print('cond_obs.shape', cond_obs.shape) #cond_obs.shape torch.Size([8, 1, 1, 182])
                a = observed_data[:,:,0].unsqueeze(2)
                ## print('a.shape', a.shape) # t.shape torch.Size([8, 1, 1])
                x = observed_data[:,:,5:]
                ## print('x.shape', x.shape) # x.shape torch.Size([8, 1, 177])
                cond_obs = torch.cat([a,x], dim=2)
                ## print('cond_obs.shape', cond_obs.shape) # cond_obs.shape torch.Size([8, 1, 178])

                noisy_target = self.mapping_noise(current_sample) 
                diff_input = cond_obs + noisy_target
 
                # predicted = self.diffmodel(diff_input, side_info, t).to(self.device)
                predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)


                ## print('predicted.shape', predicted.shape) # predicted.shape torch.Size([8, 2])

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                ## print('coeff1', coeff1) # a number
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                ## print('coeff2', coeff2) # a number
           
                current_sample = current_sample.squeeze(1)

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                    ## print('if t current_sample.shape', current_sample.shape)

                current_sample = current_sample.unsqueeze(1)

            current_sample = current_sample.squeeze(1) #[8,2]
            imputed_samples[:, i] = current_sample.detach()
            # # print(imputed_samples.shape) # torch.Size([8, 15, 2])
        # # # print('imputed_samples', torch.mean(imputed_samples[:, 1], torch.mean(imputed_samples[:, 2])))
        return imputed_samples

    def forward(self, batch, is_train=1, propnet = None):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
    

        # In testing, using `gt_mask` (generated with fixed missing rate).
        if is_train == 0:
            cond_mask = gt_mask.clone()
        # In training, generate random mask
        else:
            # cond_mask = self.get_randmask(observed_mask)

            # Try both mask all factual or randomly mask partial factual y, keep t, x unmask.
            cond_mask = gt_mask.clone() # shape(8,1,30)
            cond_mask[:, :, 1] = 0 
            cond_mask[:, :, 2] = 0 


        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss(observed_data, cond_mask, gt_mask, side_info, is_train, set_t=-1, propnet=propnet) if is_train == 1 else self.calc_loss_valid
        
        return loss_func

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            # # # print('cond_mask.shape', cond_mask.shape) #(8,1,30)
            cond_mask[:,:,0] = 0 # Do not need to give factual t at test.
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class TabCSDI(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(TabCSDI, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # Insert K=1 axis. All mask now with shape (B, 1, L).
        observed_data = batch["observed_data"][:, np.newaxis, :] # shape (8,1,10)
        observed_data = observed_data.to(self.device).float()

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float() #shape (8,10)

        gt_mask = batch["gt_mask"][:, np.newaxis, :]

        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
        