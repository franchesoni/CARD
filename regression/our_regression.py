import time
import os
import numpy as np
import torch
from torch.utils import data
import logging

from utils import get_dataset, get_optimizer
from our_methods import get_method


class Runner:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.dataset_object = None
        self.method_name = args.loss.split(".")[1]

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        # first obtain test set for pre-trained model evaluation
        logging.info("Test set info:")
        test_set_object, test_set = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            test_set,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        # obtain training set
        logging.info("Training set info:")
        dataset_object, dataset = get_dataset(args, config, test_set=False)
        self.dataset_object = dataset_object
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        # load model
        assert dataset_object.train_dim_y == 1
        ys = dataset[:, -1]
        bounds = (min(ys.min(), -3 * ys.std()), max(ys.max(), 3 * ys.std()))
        # hparams = dict(n_components=10, n_bins=10, n_quantile_levels=10, bounds=bounds)
        hparams = dict(n_components=10, n_bins=2, n_quantile_levels=10, bounds=bounds)

        model = get_method(self.method_name)(
            [dataset_object.train_dim_x, 100, 50],
            **hparams
        ).to(config.device)
        optimizer = get_optimizer(self.config.optim, model.parameters())
        train_epochs = config.diffusion.nonlinear_guidance.n_pretrain_epochs
        model.train()
        for epoch in range(train_epochs):
            for step, xy_0 in enumerate(train_loader):
                xy_0 = xy_0.to(config.device)
                x_batch = xy_0[:, : -self.config.model.y_dim]
                y_batch = xy_0[:, -self.config.model.y_dim :]
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = model.loss(y_batch, pred)
                loss.backward()
                optimizer.step()
                print(f"epoch {epoch} step {step:03} loss {loss:.3f}", end="\r")
        states = [model.state_dict(), optimizer.state_dict(), hparams]
        torch.save(states, os.path.join(args.log_path, "ckpt.pth"))

    def test(self):
        """
        Evaluate model on regression tasks on test set.
        """
        print("- preparing...")
        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, dataset = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            dataset,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        self.dataset_object = dataset_object
        # set global prevision value for NLL computation if needed
        if args.nll_global_var:
            raise NotImplementedError("did you set global var? Not expected.")
            set_NLL_global_precision(test_var=args.nll_test_var)

        print("- loading...")

        if getattr(self.config.testing, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(log_path, "ckpt.pth"), map_location=self.device
            )
            ckpt_id = "last"
        else:
            states = torch.load(
                os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                map_location=self.device,
            )
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        hparams = states[2]

        model = get_method(self.method_name)(
            [dataset_object.train_dim_x, 100, 50], **hparams).to(config.device)
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)
        model.eval()

        print("- testing...")
        logscores = []
        import ipdb; ipdb.set_trace()
        with torch.no_grad():
            for step, xy_batch in enumerate(test_loader):
                # minibatch_start = time.time()
                xy_0 = xy_batch.to(self.device)
                current_batch_size = xy_0.shape[0]
                x_batch = xy_0[:, : -config.model.y_dim]
                y_batch = xy_0[:, -config.model.y_dim :]
                pred = model(x_batch)

                logscores.append(model.get_logscore_at_y(y_batch, pred).cpu())

        y_nll_all_steps_list = torch.concatenate(logscores).view(-1).numpy().tolist()
        # logging.info(f"y NLL at all steps: {y_nll_all_steps_list}.\n\n")

        return ([], [], [], y_nll_all_steps_list, [], [])
