import time
import os
import numpy as np
import torch
from torch.utils import data
import logging

from utils import get_dataset, get_optimizer
from our_methods import get_method
from model import EarlyStopping


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
        val_set_object, val_set = get_dataset(
            args, config, test_set=True, validation=True
        )
        train_subset_object, train_subset = get_dataset(
            args, config, test_set=False, validation=True
        )
        self.dataset_object = dataset_object
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = data.DataLoader(
            val_set,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        train_subset_loader = data.DataLoader(
            train_subset,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
            shuffle=True,
        )

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(args.log_path, "mytensorboard"))
        early_stopper = EarlyStopping(
            patience=config.diffusion.nonlinear_guidance.patience,
            delta=config.diffusion.nonlinear_guidance.delta,
        )
        # load model
        assert dataset_object.train_dim_y == 1
        ys = dataset[:, -1]
        bounds = (
            min(ys.min(), -4 * ys.std() + ys.mean()),
            max(ys.max(), 4 * ys.std() + ys.mean()),
        )
        hparams = dict(n_components=32, n_bins=32, n_quantile_levels=32, bounds=bounds)

        model = get_method(self.method_name)(
            [dataset_object.train_dim_x, 100, 50], **hparams
        ).to(config.device)
        optimizer = get_optimizer(self.config.optim, model.parameters())
        train_epochs = config.diffusion.nonlinear_guidance.n_pretrain_epochs
        model.train()
        for epoch in range(train_epochs):
            step_offset = len(train_subset_loader) * epoch
            for step, xy_0 in enumerate(train_subset_loader):
                xy_0 = xy_0.to(config.device)
                x_batch = xy_0[:, : -self.config.model.y_dim]
                y_batch = xy_0[:, -self.config.model.y_dim :]
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = model.loss(y_batch, pred)
                loss.backward()
                optimizer.step()
                print(f"epoch {epoch} step {step:03} loss {loss:.3f}", end="\r")
                writer.add_scalar("Loss/train_subset", loss, step_offset + step)
            with torch.no_grad():
                model.eval()
                val_losses = []
                for xy_0 in val_loader:
                    xy_0 = xy_0.to(config.device)
                    x_batch = xy_0[:, : -self.config.model.y_dim]
                    y_batch = xy_0[:, -self.config.model.y_dim :]
                    pred = model(x_batch)
                    loss = model.loss(y_batch, pred)
                    val_losses.append(loss)
                val_loss = torch.stack(val_losses).mean()
                model.train()
                writer.add_scalar("Loss/val", val_loss, step_offset)
                early_stopper(val_cost=val_loss, epoch=epoch)
            if early_stopper.early_stop:
                print("Early stopping")
                best_n_epochs = early_stopper.best_epoch
                break
        for epoch in range(best_n_epochs):
            step_offset = len(train_loader) * epoch
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
                writer.add_scalar("Loss/train", loss, step_offset + step)

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
            [dataset_object.train_dim_x, 100, 50], **hparams
        ).to(config.device)
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)
        model.eval()

        print("- testing...")
        logscores = []
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
