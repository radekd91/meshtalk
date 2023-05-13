"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import torch as th
from models.vertex_unet import VertexUnet
from models.encoders import MultimodalEncoder
from training.dataset import DataReader
from training.forwarder import DiscreteExpressionForwarder
from training.trainer import Trainer

from gdl.datasets.MEADPseudo3DDM import MEADPseudo3DDM


config = {
    "artifacts_dir": "artifacts_dir",
    "vertex_mean": "assets/face_mean.npy",
    "vertex_std": "assets/face_std.npy",
    "dataloader_workers": 4,
    "expression_space": {
        "classes": 128,
        "heads": 16,
    },
    "train": {
        "learning_rate": 0.0001,
        "batch_size": 8,
        "segment_length": 32,
        "max_iterations": 50_000,
        "decrease_lr_iters": [40_000],
        "save_frequency": 10_000,
        "weight_decay": 0.001,
        "loss_terms": ["recon", "landmarks", "modality_crossing"],
        "loss_weights": {"recon": 1.0, "landmarks": 1.0, "modality_crossing": 1.0},
    },
}

os.system(f"mkdir -p {config['artifacts_dir']}")

# load mean, stddev, and masks
## TODO: this part is really unfortunate, I think we need to adjsut this based on our data, vertex_mean is probably a netural flame mesh, 
# std taken from the shape space, Timo might know the easiest way to get them
mean = th.from_numpy(np.load(config["vertex_mean"])) 
stddev = th.from_numpy(np.load(config["vertex_std"]))  
## TODO: This part is even more annoying, they have special keypoints and masks of mouth and eye regions (this would have to be adjusted to FLAME)
mouth_mask = np.loadtxt("assets/weighted_mouth_mask.txt").astype(np.float32).flatten() #
eye_mask = np.loadtxt("assets/weighted_eye_mask.txt", dtype=np.float32).flatten()
eye_keypoints = np.loadtxt("assets/eye_keypoints.txt", dtype=np.float32).flatten()

# ## MESHTALK template:
# # define train and validation dataset
# train_dataset = DataReader()
# val_dataset = DataReader()

# ## our adaptation: 
dm = MEADPseudo3DDM(
    # TODO: call with the same parameters as our final models 
)

dm.prepare_data()
dm.setup()

train_dataset = dm.training_set
val_dataset = dm.validation_set

# create models
geom_unet = VertexUnet(classes=config["expression_space"]["classes"],
                       heads=config["expression_space"]["heads"],
                       n_vertices=train_dataset.n_vertices, # TODO: hack in the FLAME vertices
                       mean=mean,
                       stddev=stddev,
                       )
print(f"geom_unet: {geom_unet.num_trainable_parameters()} trainable parameters")
encoder = MultimodalEncoder(classes=config["expression_space"]["classes"],
                            heads=config["expression_space"]["heads"],
                            expression_dim=128,
                            audio_dim=128,
                            n_vertices=train_dataset.n_vertices,  # TODO: hack in the FLAME vertices
                            mean=mean,
                            stddev=stddev
                            )
print(f"encoder: {encoder.num_trainable_parameters()} trainable parameters")

# train
forwarder = DiscreteExpressionForwarder(config, geom_unet, encoder, mouth_mask, eye_mask, eye_keypoints)
trainer = Trainer(config, forwarder, train_dataset, val_dataset)
trainer.train()
