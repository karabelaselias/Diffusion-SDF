#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import logging
import glob
from datetime import datetime

# New
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.pc_loader import PCloader
from dataloader.sdf_loader import SdfLoader
from dataloader.modulation_loader import ModulationLoader

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_world_size():
    world_size_keys = ("WORLD_SIZE", "SLURM_NTASKS", "JSM_NAMESPACE_SIZE")
    for key in world_size_keys:
        world_size = os.environ.get(key)
        if world_size is not None:
            return int(world_size)
    return 1


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def train(args, specs):

    # Determine stage from specs
    stage = specs.get('training_task', 'unknown')  # 'modulation', 'diffusion', or 'combined'
    
    
    # initialize dataset and loader
    split = json.load(open(specs["TrainSplit"], "r"))
    if specs['training_task'] == 'diffusion':
        train_dataset = ModulationLoader(specs["data_path"], pc_path=specs.get("pc_path",None), split_file=split, pc_size=specs.get("total_pc_size", None))
    else:
        train_dataset = SdfLoader(specs["DataSource"], split, pc_size=specs.get("PCsize",1024), grid_source=specs.get("GridSource", None), modulation_path=specs.get("modulation_path", None), augment=specs.get("augment_training", False))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
    )

    # Setup WandB
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = specs.get("wandb_project", "atria-3d-diffusion")  # Can set project name in specs

    # Default tags based on stage
    stage_tags = {
        'modulation': ['modulation', 'sdf-vae', 'stage1'],
        'diffusion': ['diffusion', 'latent-diffusion', 'stage2'],
        'combined': ['end2end', 'fine-tuning', 'stage3']
    }
        
    # Get tags from specs or use defaults
    tags = specs.get("wandb_tags", stage_tags.get(stage, [stage]))

    # Get group from specs or use stage name
    group = specs.get("wandb_group", stage)

    # loggers
    
    # TensorBoard logger (keeps your original functionality)
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(args.exp_dir, "logs"),
        name=None,  # Don't create additional subdirectory
        default_hp_metric=False
    )
    
    wandb_logger = pl_loggers.WandbLogger(
        project=project_name,
        name=f"{stage}_{timestamp}",
        group=group,  # Groups runs together
        tags=tags,    # Tags for filtering
        save_dir=os.path.join(args.exp_dir, "wandb"),
        log_model=False,
        config={
            **specs, 
            'exp_dir': args.exp_dir, 
            'batch_size': args.batch_size,
            'num_gpus': args.num_gpus,
            'workers': args.workers,
            'stage': stage,
            'timestamp': timestamp
        }
    )
    
    # CSV logger for easy metric inspection
    csv_logger = pl_loggers.CSVLogger(
        save_dir=args.exp_dir,
        name="csv_logs",
        flush_logs_every_n_steps=100  # Flush to disk every 100 steps
    )
    
    # Use both loggers
    loggers = [tb_logger, wandb_logger, csv_logger]
    
    callback = ModelCheckpoint(dirpath=args.exp_dir, 
                               filename='ckpt-{epoch:02d}-{total:.2f}',
                               save_top_k=3,
                               every_n_epochs=specs["log_freq"],
                               monitor='total',
                               mode='min',
                               verbose=True
                              )
    
    # pytorch lightning callbacks    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [callback, lr_monitor]

    model = CombinedModel(specs) if specs['SDFModel'] != 'vecset' else CombinedModelVecSet(specs)
    # Compile model if PyTorch 2.0+
    if torch.__version__ >= '2.0.0' and specs.get('compile_model', False) :
        print("Compiling model components...")
        from utils.compile_model import compile_model
        model = compile_model(
            model, 
            mode='default',
            compile_eikonal=specs["SdfModelSpecs"].get("use_eikonal", False)
        )
        #model = compile_model(model, mode='default')

    # note on loading from checkpoint:
    # if resuming from training modulation, diffusion, or end-to-end, just load saved checkpoint 
    # however, if fine-tuning end-to-end after training modulation and diffusion separately, will need to load sdf and diffusion checkpoints separately
    if args.resume == 'finetune':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)
        resume = None
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume)
        #ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None  

    # set a different seed for each device
    pl.seed_everything(42, workers=True)

    
    # precision 16 can be unstable (nan loss); recommend using 32
    from pytorch_lightning.strategies import DDPStrategy
    
    from torch.distributed.algorithms.ddp_comm_hooks import (
        default_hooks as default,
        powerSGD_hook as powerSGD,
    )
    
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=args.num_gpus, 
                         strategy=DDPStrategy(
                                gradient_as_bucket_view=True, 
                                static_graph=args.static_graph,
                                find_unused_parameters=False,
                                ddp_comm_hook=default.fp16_compress_hook 
                             ) if args.num_gpus > 1 else 'auto',
                         precision='bf16-mixed', 
                         max_epochs=specs["num_epochs"], 
                         callbacks=callbacks, 
                         log_every_n_steps=1,
                         logger=loggers,
                         default_root_dir=args.exp_dir,
                         gradient_clip_val=1.0,
                         num_nodes=args.num_nodes,
                         accumulate_grad_batches = args.accumulate_grad_batches if hasattr(args, 'accumulate_grad_batches') else 1
                         #accumulate_grad_batches = 2 if args.num_gpus > 1 else 1
                         )

    # sync batch norm
    if args.num_gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)
    
if __name__ == "__main__":

    import argparse
    # Silence all but rank 0
    if get_rank() != 0:
        logging.basicConfig(level=logging.WARNING)
        # Reduce PyTorch Lightning verbosity
        pl._logger.setLevel(logging.WARNING)

    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )

    arg_parser.add_argument(
        "--specs", "-s", required=True,
        help="The config specification for the experiment"
    )
    
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=32, type=int)
    arg_parser.add_argument("--workers", "-w", default=8, type=int)
    arg_parser.add_argument("--num_gpus", "-g", default=1, type=int)
    arg_parser.add_argument("--num_nodes", "-n", default=1, type=int)
    arg_parser.add_argument("--static-graph", action='store_true')

    args = arg_parser.parse_args()
    specs = json.load(open(args.specs))
    train(args, specs)
