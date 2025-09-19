import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl

# add paths in model/__init__.py for new models
from models import * 

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'

        if self.task in ('combined', 'modulation'):
            
            # Check if flow is enabled in specs
            use_flow = specs.get("use_normalizing_flow", False)
            flow_depth = specs.get("flow_depth", 2)
            flow_hidden_dim = specs.get("flow_hidden_dim", 512)
            
            self.sdf_model = SdfModel(specs=specs) 
            feature_dim = specs["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet 
            modulation_dim = feature_dim*3 # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            plane_resolution = specs["SdfModelSpecs"].get("plane_resolution", 64)
            
            # Eikonal regularization parameters
            self.use_eikonal = specs["SdfModelSpecs"].get("use_eikonal", False)
            self.eikonal_weight = specs["SdfModelSpecs"].get("eikonal_weight", 0.01)
            
            if use_flow:
                from models.autoencoder_flow import FlowBetaVAE
                self.vae_model = FlowBetaVAE(
                    in_channels=feature_dim * 3,
                    latent_dim=modulation_dim,
                    hidden_dims=hidden_dims,
                    kl_std=latent_std,
                    use_flow=True,
                    flow_depth=flow_depth,
                    flow_hidden_dim=flow_hidden_dim,
                    input_resolution=plane_resolution
                )
            else:
                self.vae_model = BetaVAE(
                in_channels=feature_dim * 3,
                latent_dim=modulation_dim,
                hidden_dims=hidden_dims,
                kl_std=latent_std,
                input_resolution=plane_resolution
            )
            #self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 
 
    def compute_gradient(self, xyz, pred_sdf):
        """
        Compute the gradient of SDF with respect to input points
        Args:
            xyz: input points (B, N, 3)
            pred_sdf: predicted SDF values (B, N, 1) or (B, N)
        Returns:
            gradients: (B, N, 3)
        """
        # Ensure pred_sdf has the right shape for grad computation
        if pred_sdf.dim() == 2:
            pred_sdf = pred_sdf.unsqueeze(-1)
        
        d_output = torch.ones_like(pred_sdf, requires_grad=False, device=pred_sdf.device)
        
        gradients = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=xyz,
            grad_outputs=d_output,
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        return gradients

    def eikonal_loss(self, gradients):
        """
        Compute eikonal loss to enforce unit norm gradients
        Args:
            gradients: (B, N, 3) gradient vectors
        Returns:
            eikonal_loss: scalar loss value
        """
        # Compute L2 norm of gradients
        gradient_norm = gradients.norm(2, dim=-1)  # (B, N)
        
        # Eikonal equation: ||∇f|| = 1
        eikonal = (gradient_norm - 1).pow(2).mean()
        
        return eikonal
    
    def training_step(self, x, idx):
        # Add this line at the very beginning
        torch.compiler.cudagraph_mark_step_begin()
        
        if self.task == 'combined':
            result = self.train_combined(x)
            return result
        elif self.task == 'modulation':
            result = self.train_modulation(x)
            return result
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] , 'fused': True},
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'], 'fused': True}
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'], 'fused' : True }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'], 'fused': True }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                 "lr_scheduler": {
                     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10000, min_lr=1e-6),
                     "monitor": "total"
                 }
        }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def train_modulation(self, x):

        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        # STEP 4: Add eikonal regularization if enabled
        if self.use_eikonal:
            # Detach the plane features to avoid second-order gradients through grid_sample
            xyz_eik = xyz.requires_grad_(True)
            reconstructed_plane_feature_detached = reconstructed_plane_feature.detach()
            pred_sdf_eik = self.sdf_model.forward_with_plane_features(
                reconstructed_plane_feature_detached, xyz_eik
            )
            gradients = self.compute_gradient(xyz_eik, pred_sdf_eik)
            eikonal = self.eikonal_loss(gradients)
            loss = loss + self.eikonal_weight * eikonal
            loss_dict = {"sdf": sdf_loss, "vae": vae_loss, "eikonal": eikonal, "total": loss}
        else:
            loss_dict = {"sdf": sdf_loss, "vae": vae_loss, "total": loss}
            
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        return loss


    def train_diffusion(self, x):

        self.train()

        pc = x['point_cloud'] # (B, 1024, 3) or False if unconditional 
        latent = x['latent'] # (B, D)

        # unconditional training if cond is None 
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None 

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)
            
        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        #print("plane feat shape: ", feat.shape)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF 
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 3.5: Compute eikonal loss if enabled
        eikonal_loss_1 = torch.tensor(0.0, device=xyz.device)
        if self.use_eikonal:
            # Detach the plane features to avoid second-order gradients through grid_sample
            xyz_eik = xyz.requires_grad_(True)
            reconstructed_plane_feature_detached = reconstructed_plane_feature.detach()
            pred_sdf_eik = self.sdf_model.forward_with_plane_features(
                reconstructed_plane_feature_detached, xyz_eik
            )
            gradients = self.compute_gradient(xyz_eik, pred_sdf_eik)
            eikonal_loss_1 = self.eikonal_loss(gradients)
        
        # STEP 4: use latent as input to diffusion model
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        
        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        # STEP 5.5: Compute eikonal loss for generated SDF if enabled
        eikonal_loss_2 = torch.tensor(0.0, device=xyz.device)
        if self.use_eikonal:
            # Detach the plane features to avoid second-order gradients through grid_sample
            xyz_eik = xyz.requires_grad_(True)
            generated_plane_feature_detach = generated_plane_feature.detach()
            pred_sdf_eik = self.sdf_model.forward_with_plane_features(
                generated_plane_feature_detach, xyz_eik
            )
            gradients_gen = self.compute_gradient(xyz_eik, pred_sdf_eik)
            eikonal_loss_2 = self.eikonal_loss(gradients_gen)
        
        # surface weight could prioritize points closer to surface but we did not notice better results when using it 
        #surface_weight = torch.exp(-50 * torch.abs(gt))
        #generated_sdf_loss = torch.mean( F.l1_loss(generated_sdf_pred, gt, reduction='none') * surface_weight )

        # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs file)
        # results could potentially improve with a grid search 
        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss
        
        if self.use_eikonal:
            # Add eikonal losses for both reconstructed and generated SDFs
            total_eikonal = eikonal_loss_1 + eikonal_loss_2
            loss = loss + self.eikonal_weight * total_eikonal
            
            loss_dict = {
                "total": loss,
                "sdf": sdf_loss,
                "vae": vae_loss,
                "diff": diff_loss,
                "gensdf": generated_sdf_loss,
                "eikonal_recon": eikonal_loss_1,
                "eikonal_gen": eikonal_loss_2,
            }
        else:
            loss_dict = {
                "total": loss,
                "sdf": sdf_loss,
                "vae": vae_loss,
                "diff": diff_loss,
                "gensdf": generated_sdf_loss,
            }
        
        #loss_dict =  {
        #                "total": loss,
        #                "sdf": sdf_loss,
        #                "vae": vae_loss,
        #                "diff": diff_loss,
        #                # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        #                # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        #                # visualizing loss curves can help with debugging if training is unstable
        #                #"diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
        #                #"diff1000": diff_1000_loss,
        #                "gensdf": generated_sdf_loss,
        #            }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss