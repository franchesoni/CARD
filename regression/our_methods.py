from typing import Sequence
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

log2pi = torch.log(torch.tensor(2 * torch.pi))


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_fn=nn.LeakyReLU,
        dropout_p=0.0,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.dropout_p = dropout_p
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn())
                layers.append(nn.Dropout(p=self.dropout_p))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ProbabilisticMethod(ABC):
    @abstractmethod
    def get_F_at_y(self, batch_y, pred_params):
        pass

    @abstractmethod
    def get_logscore_at_y(self, batch_y, pred_params):
        pass

    @abstractmethod
    def loss(self, batch_y, pred_params):
        pass

    def forward(self, batch_x):
        pass

    def get_numerical_CRPS(
        self, batch_y, pred_params, lower, upper, count, divide=False
    ):
        assert batch_y.shape[1] == 1
        dys = torch.linspace(lower, upper, count)

        if divide:
            print("running with divide", divide, "-" * 10)
            # we do it ten points at a time to avoid memory issues
            for i in range(
                0, len(dys), divide
            ):  # (0, 10, 20, ..., ((count-1)//10)*10) < count
                print(f"getting numerical crps {i}...")
                divided_points = dys[i : i + divide]
                crps = self.get_numerical_CRPS(
                    batch_y,
                    pred_params,
                    divided_points[0],
                    divided_points[1],
                    len(divided_points),
                    divide=False,
                )
                if i == 0:
                    out = crps
                else:
                    out += crps
            return out

        dys = (
            dys.reshape(1, count).expand(pred_params.shape[0], count).to(batch_y.device)
        )
        Fy = self.get_F_at_y(dys, pred_params)  # (N, count)
        heavyside = 1 * (batch_y <= dys)  # (N, count)
        integrant = (Fy - heavyside) ** 2  # (N, count)
        crps = (dys[0, 1] - dys[0, 0]) * (
            integrant[:, 0] / 2 + integrant[:, 1:].sum(dim=1) + integrant[:, -1] / 2
        )  # (N,)
        return crps


class LaplaceLogScore(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(LaplaceLogScore, self).__init__()
        self.model = MLP(layer_sizes + [2], **kwargs)

    def get_F_at_y(self, batch_y, pred_params):
        mus, bs = pred_params[:, 0:1], pred_params[:, 1:]
        return 0.5 + 0.5 * (2 * (mus < batch_y) - 1) * (
            1 - torch.exp(-torch.abs(mus - batch_y) / bs)
        )

    def get_logscore_at_y(self, batch_y, pred_params):
        mus, bs = pred_params[:, 0:1], pred_params[:, 1:]
        return torch.log(2 * bs) + torch.abs(mus - batch_y) / bs

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(
            batch_y, pred_params
        ).mean()  # we should use sum but the mean has more deceent magnitude

    def forward(self, batch_x):
        params = self.model(batch_x)  # logits
        mu, blogits = params[:, 0], params[:, 1]
        blogits = nn.functional.softplus(blogits)  # w_b must be positive
        return torch.stack([mu, blogits], dim=1)


class LaplaceCRPS(LaplaceLogScore):
    def __init__(self, layer_sizes, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(LaplaceCRPS, self).__init__(layer_sizes)
        self.model = MLP(layer_sizes + [2], **kwargs)

    def get_crps_at_y(self, batch_y, pred_params):
        mus, bs = pred_params[:, 0:1], pred_params[:, 1:]
        crps = bs * (
            torch.abs(batch_y - mus) / bs
            + torch.exp(-torch.abs(batch_y - mus) / bs)
            - 3 / 4
        )
        return crps

    def loss(self, batch_y, pred_params):
        return self.get_crps_at_y(
            batch_y, pred_params
        ).mean()  # we should use sum but the mean has more deceent magnitude


class LaplaceGlobalWidth(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, train_width=True, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(LaplaceGlobalWidth, self).__init__()
        self.model = MLP(layer_sizes + [1], **kwargs)
        self.global_width = torch.tensor([1.0])
        self.train_width = train_width
        if train_width:
            self.global_width = nn.Parameter(self.global_width)

    def get_F_at_y(self, batch_y, pred_params):
        mus = pred_params
        bs = self.global_width
        return 0.5 + 0.5 * (2 * (mus < batch_y) - 1) * (
            1 - torch.exp(-torch.abs(mus - batch_y) / bs)
        )

    def get_logscore_at_y(self, batch_y, pred_params):
        self.global_width = self.global_width.to(pred_params.device)
        mus, bs = pred_params, self.global_width
        assert bs > 0, "Width must be positive"
        out = torch.log(2 * bs) + torch.abs(mus - batch_y) / bs
        return out

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(
            batch_y, pred_params
        ).mean()  # sum should be preferred if not for scale

    def forward(self, batch_x):
        return self.model(batch_x)


class MixtureDensityNetwork(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, n_components, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(MixtureDensityNetwork, self).__init__()
        self.model = MLP(layer_sizes + [3 * n_components], **kwargs)
        self.n_components = n_components

    def get_F_at_y(self, batch_y, pred_params):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components],
            pred_params[:, self.n_components : 2 * self.n_components],
            pred_params[:, 2 * self.n_components :],
        )
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert (
            len(batch_y.shape) == 2
        ), f"batch_y.shape is {batch_y.shape} but should be (N, 1)"
        batch_y = batch_y.unsqueeze(1)  # (N, 1, Y)
        pis, mus, sigmas = (
            pis.unsqueeze(2),
            mus.unsqueeze(2),
            sigmas.unsqueeze(2),
        )  # (N, K, 1)
        return (pis * 0.5 * (1 + torch.erf((batch_y - mus) / (sigmas * (2**0.5))))).sum(
            dim=1
        )  # (N, Y)

    def get_logscore_at_y(self, batch_y, pred_params):
        pis, mus, sigmas = (
            pred_params[:, : self.n_components],
            pred_params[:, self.n_components : 2 * self.n_components],
            pred_params[:, 2 * self.n_components :],
        )
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert (
            len(batch_y.shape) == 2
        ), f"batch_y.shape is {batch_y.shape} but should be (N, 1)"
        batch_y = batch_y.unsqueeze(1)  # (B, 1, Y)
        pis, mus, sigmas = (
            pis.unsqueeze(2),
            mus.unsqueeze(2),
            sigmas.unsqueeze(2),
        )  # (B, K, 1)
        log_prob_per_component = (
            -0.5 * (((batch_y - mus) / sigmas) ** 2) - torch.log(sigmas) - 0.5 * log2pi
        )  # (B, K, Y)
        weighted_log_prob = torch.log(pis) + log_prob_per_component
        neg_log_likelihood = -torch.logsumexp(weighted_log_prob, dim=1)  # (B, Y)
        return neg_log_likelihood

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).mean()

    def forward(self, batch_x):
        params = self.model(batch_x)  # logits
        pis, mus, sigmas = (
            params[:, : self.n_components],
            params[:, self.n_components : 2 * self.n_components],
            params[:, 2 * self.n_components :],
        )
        pis = nn.functional.softmax(pis, dim=1)  # pis must be positive and sum to 1
        sigmas = nn.functional.softplus(sigmas)  # w_b must be positive
        return torch.concatenate([pis, mus, sigmas], dim=1)

class Gaussian(MixtureDensityNetwork):
    def __init__(self, layer_sizes, n_components, **kwargs):
        # call super
        super(Gaussian, self).__init__(layer_sizes, n_components=1, **kwargs)

class KDESampler(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, n_components, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(KDESampler, self).__init__()
        self.model = MLP(layer_sizes + [n_components], **kwargs)
        self.n_components = n_components

    def get_F_at_y(self, batch_y, pred_params):
        mus = pred_params
        pis = torch.ones_like(mus) / mus.shape[1]
        sigmas = torch.sqrt(mus.var(dim=1, unbiased=True)) * torch.ones_like(mus)
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert (
            len(batch_y.shape) == 2
        ), f"batch_y.shape is {batch_y.shape} but should be (N, 1)"
        batch_y = batch_y.unsqueeze(1)  # (N, 1, Y)
        pis, mus, sigmas = (
            pis.unsqueeze(2),
            mus.unsqueeze(2),
            sigmas.unsqueeze(2),
        )  # (N, K, 1)
        return (pis * 0.5 * (1 + torch.erf((batch_y - mus) / (sigmas * (2**0.5))))).sum(
            dim=1
        )  # (N, Y)

    def get_logscore_at_y(self, batch_y, pred_params):
        mus = pred_params
        pis = torch.ones_like(mus) / mus.shape[1]
        sigmas = torch.sqrt(
            mus.var(dim=1, unbiased=True, keepdims=True)
        ) * torch.ones_like(mus)
        assert pis.shape[0] == batch_y.shape[0]
        assert pis.shape[1] == self.n_components
        assert (
            len(batch_y.shape) == 2
        ), f"batch_y.shape is {batch_y.shape} but should be (N, 1)"
        batch_y = batch_y.unsqueeze(1)  # (B, 1, Y)
        pis, mus, sigmas = (
            pis.unsqueeze(2),
            mus.unsqueeze(2),
            sigmas.unsqueeze(2),
        )  # (B, K, 1)
        log_prob_per_component = (
            -0.5 * (((batch_y - mus) / sigmas) ** 2) - torch.log(sigmas) - 0.5 * log2pi
        )  # (B, K, Y)
        weighted_log_prob = torch.log(pis) + log_prob_per_component
        neg_log_likelihood = -torch.logsumexp(weighted_log_prob, dim=1)  # (B, Y)
        return neg_log_likelihood

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).mean()

    def forward(self, batch_x):
        return self.model(batch_x)  # samples


class CategoricalCrossEntropy(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, n_bins, bounds, **kwargs):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(CategoricalCrossEntropy, self).__init__()
        self.bin_borders = torch.linspace(bounds[0], bounds[1], n_bins + 1)
        self.B = len(self.bin_borders) - 1
        self.bin_widths = self.bin_borders[1:] - self.bin_borders[:-1]
        assert self.bin_widths.min() > 0, "Bin borders must be strictly increasing"
        self.model = MLP(layer_sizes + [self.B], **kwargs)

    def prepare_params(self, pred_params):
        return dict(
            cdf_at_borders=None,
            bin_masses=pred_params,
            bin_borders=self.bin_borders.reshape(1, -1),
        )

    def get_F_at_y(self, batch_y, pred_params):
        return get_F_at_y_PL(batch_y, **self.prepare_params(pred_params))

    def get_logscore_at_y(self, batch_y, pred_params):
        return get_logscore_at_y_PL(batch_y, **self.prepare_params(pred_params))

    def loss(self, batch_y, pred_params):
        return self.get_logscore_at_y(batch_y, pred_params).mean()

    def forward(self, batch_x):
        logits = self.model(batch_x)
        masses = nn.functional.softmax(logits, dim=1)
        return masses


class PinballLoss(ProbabilisticMethod, nn.Module):
    def __init__(
        self, layer_sizes, n_quantile_levels, bounds, predict_residuals=False, **kwargs
    ):
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(PinballLoss, self).__init__()
        quantile_levels = torch.linspace(0, 1, n_quantile_levels + 2)[1:-1]
        self.quantile_levels = torch.sort(quantile_levels)[0]
        assert self.quantile_levels.min() > 0, "Quantiles must be in [0, 1]"
        assert self.quantile_levels.max() < 1, "Quantiles must be in [0, 1]"
        self.lower, self.upper = torch.tensor(bounds)
        self.lower, self.upper = torch.tensor(self.lower), torch.tensor(self.upper)
        self.quantile_levels = torch.concatenate(
            (torch.tensor([0]), self.quantile_levels, torch.tensor([1]))
        )
        self.B = len(self.quantile_levels) - 1
        self.model = MLP(layer_sizes + [len(quantile_levels)], **kwargs)
        self.predict_residuals = predict_residuals
        self.do_sort = True  # because we don't care about the gradient

    def prepare_params(self, pred_params):
        self.lower, self.upper = self.lower.to(pred_params.device), self.upper.to(
            pred_params.device
        )
        self.quantile_levels = self.quantile_levels.to(pred_params.device)
        cdf_at_borders = self.quantile_levels.reshape(1, -1)
        bin_masses = None
        N, Q = pred_params.shape  # Q = number of quantiles
        bin_borders = torch.concatenate(
            (
                self.lower.reshape(1, 1).expand(N, 1),
                pred_params,
                self.upper.reshape(1, 1).expand(N, 1),
            ),
            dim=1,
        )
        if self.do_sort:
            bin_borders = torch.sort(bin_borders, dim=1)[
                0
            ]  # we can do this in pinball as we don't care about this gradient
        return dict(
            cdf_at_borders=cdf_at_borders,
            bin_masses=bin_masses,
            bin_borders=bin_borders,
        )

    def get_F_at_y(self, batch_y, pred_params):
        return get_F_at_y_PL(batch_y, **self.prepare_params(pred_params))

    def get_logscore_at_y(self, batch_y, pred_params):
        logscore = get_logscore_at_y_PL(batch_y, **self.prepare_params(pred_params))
        return logscore

    def loss(self, batch_y, pred_params):
        batch_y = batch_y.reshape(-1, 1)
        self.quantile_levels = self.quantile_levels.to(batch_y.device)
        pinball = (self.quantile_levels[1:-1] - 1 * (batch_y < pred_params)) * (
            batch_y - pred_params
        )
        return pinball.sum(dim=1).mean()

    def forward(self, batch_x):
        out = self.model(batch_x)
        if self.predict_residuals:
            residuals = torch.concatenate(
                (out[:, :1], nn.functional.softplus(out[:, 1:])), dim=1
            )
            out = torch.cumsum(residuals, dim=1)
        return out


class CRPSHist(CategoricalCrossEntropy):
    def loss(self, batch_y, pred_params):
        return get_crps_PL(batch_y, **self.prepare_params(pred_params)).mean()


class CRPSQR(PinballLoss):
    def loss(self, batch_y, pred_params):
        self.do_sort = False
        bin_borders = self.prepare_params(pred_params)["bin_borders"]
        self.do_sort = True
        bin_widths = bin_borders[:, 1:] - bin_borders[:, :-1]  # (N, B)
        if (bin_widths < 0).any():  # bins are unordered, crps can't be computed
            return super().loss(batch_y, pred_params)
        else:
            return get_crps_PL(batch_y, **self.prepare_params(pred_params)).mean()


class LogScoreQR(PinballLoss):
    def loss(self, batch_y, pred_params):
        self.do_sort = False
        bin_borders = self.prepare_params(pred_params)["bin_borders"]
        self.do_sort = True
        bin_widths = bin_borders[:, 1:] - bin_borders[:, :-1]  # (N, B)
        if (bin_widths < 0).any():  # bins are unordered, crps can't be computed
            return super().loss(batch_y, pred_params)
        else:
            return get_logscore_at_y_PL(
                batch_y, **self.prepare_params(pred_params)
            ).mean()


class IQN(ProbabilisticMethod, nn.Module):
    def __init__(self, layer_sizes, n_layers_h=3, cos_n=64, **kwargs):
        # here we init the iqn, which is composed by h and phi. These are separate networks.
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(IQN, self).__init__()
        self.embedding_dim = layer_sizes[-1]
        self.g = MLP(layer_sizes, **kwargs)
        self.h = MLP(
            [self.embedding_dim] * n_layers_h + [1]
        )  # just two layers as in https://github.com/BY571/IQN-and-Extensions/blob/master/IQN-DQN.ipynb
        self.cos_n = cos_n
        pis = torch.tensor([torch.pi * i for i in range(self.cos_n)]).view(
            1, self.cos_n
        )
        self.register_buffer("pis", pis)
        self.cos_embed = nn.Linear(self.cos_n, self.embedding_dim)

    def sample_taus(self, batch_size, n_taus=8):
        taus = torch.rand(batch_size, n_taus).unsqueeze(2)  # (N, n_taus, 1)
        return taus

    def get_tau_embedding(self, taus):
        # taus must be (N, n_taus, Y)
        taus_shape = taus.shape
        taus = taus.reshape(-1, 1)  # (N*n_taus*Y, 1)
        return nn.functional.relu(
            self.cos_embed(torch.cos(taus * self.pis).view(-1, self.cos_n))
        ).view(*taus_shape, self.embedding_dim)

    def forward(self, batch_x):
        return self.g(batch_x)  # (N, embedding_dim)

    def loss(self, batch_y, pred_params):
        # pred_params is (N, D)
        N, Y = batch_y.shape
        taus = self.sample_taus(batch_y.shape[0]).to(batch_y.device)  # (N, n_taus, 1)
        tau_embedding = self.get_tau_embedding(taus)  # (N, n_taus, 1, embedding_dim)
        assert tau_embedding.shape[2] == 1
        tau_embedding = tau_embedding.squeeze(2)  # (N, n_taus, embedding_dim)
        quantiles = self.h(tau_embedding * pred_params.unsqueeze(1))  # (N, n_taus, 1)
        pinball = (taus - 1 * (batch_y.unsqueeze(1) < quantiles)) * (
            batch_y.unsqueeze(1) - quantiles
        )  # (N, n_taus, Y)
        return pinball.sum(dim=1).mean()

    def prepare_params(self, batch_y, pred_params):
        batch_y = batch_y.contiguous()
        N, Y = batch_y.shape
        D = pred_params.shape[1]
        pred_params = pred_params.expand(N, D)  # (N, D)
        return N, Y, D, batch_y, pred_params

    def get_dist(self, batch_y, pred_params, samples_per_step=8, num_steps=4):
        N, Y, D, batch_y, pred_params = self.prepare_params(batch_y, pred_params)

        def quantile_fn(taus):
            tau_embedding = self.get_tau_embedding(taus)  # (N, n_taus, Y, E)
            return self.h(tau_embedding * pred_params[:, None, None]).squeeze(-1)

        def search_step(lower_taus, upper_taus, y_values):
            taus = (
                torch.linspace(0, 1, samples_per_step)
                .reshape(1, samples_per_step, 1)
                .expand(N, samples_per_step, 1)
                .to(batch_y.device)
            )  # (N, samples_per_step, 1)
            taus = (
                lower_taus + (upper_taus - lower_taus) * taus
            )  # (N, samples_per_step, Y)
            quantiles = quantile_fn(taus)  # (N, samples_per_step, Y)
            right_indices = torch.searchsorted(
                quantiles.permute(0, 2, 1), y_values.unsqueeze(2)
            )  # (N, Y, 1)
            left_indices = right_indices - 1
            right_indices, left_indices = right_indices.squeeze(2).unsqueeze(
                1
            ), left_indices.squeeze(2).unsqueeze(
                1
            )  # (N, 1, Y)
            right_indices, left_indices = right_indices.expand(
                N, samples_per_step, Y
            ), left_indices.expand(
                N, samples_per_step, Y
            )  # (N, samples_per_step, Y)
            new_upper_taus = torch.gather(
                taus, 1, torch.clamp(right_indices, max=samples_per_step - 1)
            )
            new_upper_taus[right_indices > samples_per_step - 1] = 1
            new_lower_taus = torch.gather(taus, 1, torch.clamp(left_indices, min=0))
            new_lower_taus[left_indices < 0] = 0
            return new_lower_taus, new_upper_taus

        # Initial bounds
        lower_taus = torch.zeros((N, 1, Y)).to(batch_y.device)
        upper_taus = torch.ones((N, 1, Y)).to(batch_y.device)

        # Perform search steps
        for _ in range(num_steps):
            lower_taus, upper_taus = search_step(lower_taus, upper_taus, batch_y)

        F_at_y = (lower_taus + upper_taus) / 2
        f_at_y = 1 / ((upper_taus - lower_taus) * Y)

        return F_at_y, f_at_y

    def get_dist2(self, batch_y, pred_params, tau_samples=2001):
        # batch_y is (N, Y)
        # pred_params is (N, D)
        # for each y we need to find the tau that gives the quantile
        # in other words we need to set the output to y and find the best tau. if we assume monotonicity this is a binary search.
        N, Y, D, batch_y, pred_params = self.prepare_params(batch_y, pred_params)

        def quantile_fn(taus):
            tau_embedding = self.get_tau_embedding(
                taus.unsqueeze(-1)
            )  # (N, n_taus, embedding_dim)
            return self.h(tau_embedding * pred_params.unsqueeze(1)).squeeze(
                -1
            )  # (N, n_taus)

        sampled_taus = (
            torch.linspace(0, 1, tau_samples).unsqueeze(0).expand(N, tau_samples)
        ).to(
            batch_y.device
        )  # (N, initial_samples)
        sampled_quantiles = quantile_fn(sampled_taus)  # (N, initial_samples)
        right_taus = torch.searchsorted(sampled_quantiles, batch_y)  # (N, Y)
        left_taus = right_taus - 1
        right_taus, left_taus = right_taus / tau_samples, left_taus / tau_samples
        F_at_y = (left_taus + right_taus) / 2
        f_at_y = (right_taus - left_taus) / (
            sampled_taus[0, 1] - sampled_taus[0, 0]
        )  # (N, Y)
        return F_at_y, f_at_y

    def get_F_at_y(self, batch_y, pred_params):
        return self.get_dist(batch_y, pred_params)[0]

    def get_logscore_at_y(self, batch_y, pred_params):
        return -torch.log(self.get_dist(batch_y, pred_params)[1])


class MCD(CategoricalCrossEntropy):
    def __init__(self, layer_sizes, n_bins, bounds, dropout_p=0, n_preds=100, **kwargs):
        # here we init the iqn, which is composed by h and phi. These are separate networks.
        """
        `layer_sizes` is a list of neurons for each layer, the first element being the dimension of the input.
        It does not include the last layer.
        """
        super(MCD, self).__init__(layer_sizes, n_bins, bounds)
        self.model = MLP(layer_sizes + [1], dropout_p=dropout_p, **kwargs)
        self.mse_loss = nn.MSELoss()
        self.bounds = bounds
        self.dropout_p = dropout_p
        self.n_preds = n_preds
        self.best_std = None

    def forward(self, batch_x):
        preds = self.model(batch_x)  # logits
        return preds

    def predict_ensemble(self, x, std=torch.tensor(0.05)):
        if self.best_std is not None:
            std_ = self.best_std
        else:
            std_ = std
        preds = torch.stack(
            [self.model(x) + torch.normal(0, std_) for _ in range(self.n_preds)], dim=1
        )
        # preds is (BS, n_preds, 1)
        # print(f"one prediction is {preds[0, :10]}")
        preds = preds.squeeze(2)  # (BS, n_preds)
        # Calculate histogram
        probs = torch.zeros((preds.shape[0], self.B))  # (BS, num_bins)
        for n in range(preds.shape[0]):
            hist = torch.histc(
                preds[n], bins=self.B, min=self.bounds[0], max=self.bounds[1]
            )
            probs[n] = hist / hist.sum()
        return probs

    def loss(self, batch_y, pred_params):
        # pred_params is (N, D))
        return self.mse_loss(batch_y, pred_params)

    def turn_on_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()


def handle_input(batch_y, cdf_at_borders, bin_masses, bin_borders):
    # reshape inputs and return shapes too. Returns:
    # batch_y (N, Y)
    # cdf_at_borders (N, B+1)
    # bin_masses (N, B)
    # bin_borders (N, B+1)
    assert not (
        torch.isnan(batch_y).any()
        or (torch.isnan(cdf_at_borders).any() if cdf_at_borders is not None else False)
        or (torch.isnan(bin_masses).any() if bin_masses is not None else False)
        or torch.isnan(bin_borders).any()
    ), "NaN in input"
    assert (bin_masses is None and cdf_at_borders is not None) or (
        bin_masses is not None and cdf_at_borders is None
    )
    batch_y = batch_y.contiguous()
    N, Y = batch_y.shape
    assert (cdf_at_borders is None and bin_masses.shape[0] in [1, N]) or (
        bin_masses is None and cdf_at_borders.shape[0] in [1, N]
    )
    assert bin_borders.shape[0] in [1, N]
    assert bin_borders.shape[1] == (
        cdf_at_borders.shape[1]
        if cdf_at_borders is not None
        else bin_masses.shape[1] + 1
    )
    B = (
        cdf_at_borders.shape[1] - 1
        if cdf_at_borders is not None
        else bin_masses.shape[1]
    )
    # the F is in fact a linear interpolation of the cdf or the quantiles
    if cdf_at_borders is None:
        cdf_at_borders = torch.cat(
            (
                torch.zeros((N, 1), device=bin_masses.device),
                torch.cumsum(bin_masses, dim=1),
            ),
            dim=1,
        )  # (N, B+1)
    else:
        cdf_at_borders = cdf_at_borders.expand(N, B + 1).contiguous()
        bin_masses = cdf_at_borders[:, 1:] - cdf_at_borders[:, :-1]  # (N, B)
    cdf_at_borders = cdf_at_borders.float()  # else the comparison below might fail
    bin_borders = bin_borders.expand(N, B + 1).contiguous()
    bin_widths = bin_borders[:, 1:] - bin_borders[:, :-1]  # (N, B)
    assert torch.isclose(cdf_at_borders[0, 0], torch.tensor(0.0)) and torch.isclose(
        cdf_at_borders[-1, -1], torch.tensor(1.0)
    ), f"CDF at borders is {cdf_at_borders[0, 0]} and {cdf_at_borders[-1, -1]} but should be 0 and 1"
    bin_borders = bin_borders.contiguous()
    # send all tensors to the right device
    cdf_at_borders, bin_masses, bin_borders, bin_widths = (
        cdf_at_borders.to(batch_y.device),
        bin_masses.to(batch_y.device),
        bin_borders.to(batch_y.device),
        bin_widths.to(batch_y.device),
    )
    return N, Y, B, batch_y, cdf_at_borders, bin_masses, bin_borders, bin_widths


def get_logscore_at_y_PL(batch_y, cdf_at_borders, bin_masses, bin_borders):
    # unpack from handle_input
    N, Y, B, batch_y, cdf_at_borders, bin_masses, bin_borders, bin_widths = (
        handle_input(batch_y, cdf_at_borders, bin_masses, bin_borders)
    )

    #
    # -- SUGGESTED ASSERTIONS --
    #

    # 1) Check that batch_y has expected shape (N, Y).
    assert batch_y.shape == (N, Y), (
        f"batch_y must have shape (N, Y).  " f"Got {batch_y.shape}, expected ({N}, {Y})"
    )

    # 2) cdf_at_borders should have shape (N, B+1).
    #    We also often expect it to be sorted along dim=1, and possibly first=0, last=1, etc.
    assert cdf_at_borders.shape == (N, B + 1), (
        f"cdf_at_borders must have shape (N, B+1).  "
        f"Got {cdf_at_borders.shape}, expected ({N}, {B+1})"
    )
    # Check that it is non-decreasing in each row (typical for a CDF).
    assert (
        cdf_at_borders[:, :-1] <= cdf_at_borders[:, 1:]
    ).all(), "cdf_at_borders must be sorted non-decreasing along dim=1."

    # 3) bin_borders should have shape (N, B+1).
    #    Usually we also expect it to be sorted along dim=1.
    assert bin_borders.shape == (N, B + 1), (
        f"bin_borders must have shape (N, B+1).  "
        f"Got {bin_borders.shape}, expected ({N}, {B+1})"
    )
    # Check that it is sorted non-decreasing in each row (typical for bin edges).
    assert (
        bin_borders[:, :-1] <= bin_borders[:, 1:]
    ).all(), "bin_borders must be sorted non-decreasing along dim=1."

    # 4) If bin_masses is not None, check that it has shape (N, B).
    if bin_masses is not None:
        assert bin_masses.shape == (N, B), (
            f"bin_masses must have shape (N, B).  "
            f"Got {bin_masses.shape}, expected ({N}, {B})"
        )

    #
    # -- FUNCTION LOGIC --
    #

    # If bin_masses is None, compute from the difference in CDF
    if bin_masses is None:
        bin_masses = cdf_at_borders[:, 1:] - cdf_at_borders[:, :-1]  # (N, B)

    # -- We are discarding bin_widths returned by handle_input, as you said that's fine.
    bin_widths = bin_borders[:, 1:] - bin_borders[:, :-1]  # shape (N, B)

    # Optional: check that bin_widths are strictly positive (if you expect strict monotonic edges)
    if not (bin_widths > 0).all():
        assert not (bin_widths < 0).any(), "All bin_widths must be >= 0."
        bin_widths += 1e-4
        bin_borders = torch.concatenate((bin_borders[:,0:1], bin_borders[:,0:1] + torch.cumsum(bin_widths + 1e-4, dim=1)), dim=1)
    bin_masses = bin_masses + 1e-5
    bin_masses = bin_masses / bin_masses.sum(dim=1, keepdims=True)
    assert (bin_masses > 0).all(), "All bin_masses must be > 0."

    bin_densities = bin_masses / bin_widths
    # Optional: check that densities are >= 0
    assert (bin_densities >= 0).all(), "bin_densities must be >= 0."

    # handle the squeeze logic in searchsorted
    #   This line tries to make bin_borders 1D if shape[0] == 1 and Y>1.
    #   If you truly want that logic, just add an assertion to confirm the scenario:
    if bin_borders.shape[0] == 1 and Y > 1:
        # We expect bin_borders.shape == (1, B+1)
        # Just to be sure:
        assert bin_borders.shape == (
            1,
            B + 1,
        ), "bin_borders squeeze logic triggered, but shape is not (1, B+1)."
        borders_for_search = bin_borders.squeeze(dim=0)  # shape becomes (B+1,)
    else:
        borders_for_search = bin_borders

    # Now we do searchsorted
    y_bin = torch.searchsorted(borders_for_search, batch_y) - 1
    assert y_bin.min() >= 0, "y_bin must be >= 0."
    assert y_bin.max() < B, "y_bin must be < B."

    # Check that y_bin shape is (N, Y)
    assert y_bin.shape == (N, Y), (
        f"searchsorted result must have shape (N, Y).  Got {y_bin.shape}, "
        f"expected ({N}, {Y}).  Possibly shape mismatch with bin_borders."
    )

    # reshape so we can index in the bin dimension
    bin_densities = bin_densities.reshape(N, B, 1)  # => (N, B, 1)
    y_bin = y_bin.reshape(N, 1, y_bin.shape[1])  # => (N, 1, Y)

    # final log_score
    log_score = -(
        torch.log(bin_densities)
        * (y_bin == torch.arange(B, device=batch_y.device).reshape(1, B, 1))
    ).sum(dim=1)

    # log_score should be shape (N, Y)
    assert log_score.shape == (N, Y), (
        f"Final log_score must be (N, Y).  Got {log_score.shape}, "
        f"expected ({N}, {Y})."
    )

    return log_score


def get_values_at(tensor, k):
    """k is (N, Y), we index it with n, y.
    We return out[n,y] = tensor[n, k[n, y]]"""
    assert tensor.shape[0] == k.shape[0]
    assert len(k.shape) == 2
    N, Y = k.shape
    return tensor[torch.arange(N).reshape(N, 1).expand(N, Y), k]


def get_F_at_y_PL(batch_y, cdf_at_borders, bin_masses, bin_borders):
    N, Y, B, batch_y, cdf_at_borders, bin_masses, bin_borders, bin_widths = (
        handle_input(batch_y, cdf_at_borders, bin_masses, bin_borders)
    )
    # bin index for each y
    purek = torch.searchsorted(bin_borders, batch_y) - 1
    k = torch.clamp(purek, 0, B - 1)  # (N, Y) or (N,)
    cdf_at_borderk = get_values_at(cdf_at_borders, k)  # (N, Y)
    cdf_at_y = cdf_at_borderk + (
        (
            (batch_y - get_values_at(bin_borders, k))  # (N, Y)
            * (get_values_at(cdf_at_borders, k + 1) - cdf_at_borderk)
            / (get_values_at(bin_widths, k) + 1e-45)
        )
        * (1 * (0 < get_values_at(bin_widths, k)))
    )
    cdf_at_y[purek < 0] = 0
    return cdf_at_y


def get_crps_PL(batch_y, cdf_at_borders, bin_masses, bin_borders):
    # the idea is that we compute the crps for a flexible input
    # batch_y is (N, Y)
    # cdf_at_borders is (N, B+1) or (1, B+1)
    # bin_masses is (N, B) or (1, B)
    # bin_borders is (N, B+1) or (1, B+1)
    # the output should always be (N, Y)
    # what we are computing is (in latex):

    # A = b_B-y
    # B = \sum_{i=1}^{i=B} \int_{b_{i-1}}^{b_i} F(y')^2dy' (`int_Fysquared_bis.sum(dim=1)`)
    # C = \int_{y}^{b_k} F(y')dy'                          (`int_Fy_bis` masked and summed)
    # D = \sum_{i=k+1}^{i=B} \int_{b_{i-1}}^{b_{i}} F(y')dy'

    # crps = A + B - 2 * (C + D)             when b_0 < y < b_B
    # crps = -A + B                          when b_B < y (note that abs(A) = -A in this case and abs(A) = A in the other two cases)
    # crps = A + B - 2 * D_                  when y < b_0  (D_ = `int_Fy_bis.sum(dim=1)`)

    # crps = abs(A) + B - 2 * (C + D) * (1*(y < b_B)))  (with careful treatment of C and D)

    N, Y, B, batch_y, cdf_at_borders, bin_masses, bin_borders, bin_widths = (
        handle_input(batch_y, cdf_at_borders, bin_masses, bin_borders)
    )  # batch_y is (N, Y) and the other are (N, B(+1))
    assert (
        bin_borders == torch.sort(bin_borders, dim=1)[0]
    ).all(), "bin borders must be ordered"
    assert (bin_widths >= 0).all(), "bin borders must be ordered"

    # Compute A
    partA = torch.abs(bin_borders[:, -1:] - batch_y)  # (N, Y)
    # Compute B
    int_Fysquared_bis = (
        -1
        / 3
        * (
            cdf_at_borders[:, :-1] ** 2
            + cdf_at_borders[:, :-1] * cdf_at_borders[:, 1:]
            + cdf_at_borders[:, 1:] ** 2
        )
        * (-bin_widths)
    )  # (N, B)
    partB = int_Fysquared_bis.sum(1, keepdims=True)  # (N, 1)

    # Compute C and D
    int_Fy_bis = (
        (cdf_at_borders[:, :-1] + cdf_at_borders[:, 1:]) / 2 * (bin_widths)
    )  # (N, B)
    # Find the bin index for each y
    purekm1 = (
        torch.searchsorted(bin_borders, batch_y) - 1
    )  # (N, Y)  (starts with 0 (unless y < b_0) and contains the index of the bin y is included in)
    # note that k-1 is the index of the bin that ends at b_k
    km1 = torch.clamp(purekm1, 0, B - 1)  # (N,) or (N, Y)

    cdf_at_borderkm1 = get_values_at(cdf_at_borders, km1)  # (N, Y)
    cdf_at_borderk = get_values_at(cdf_at_borders, km1 + 1)  # (N, Y)
    bin_widths_km1 = get_values_at(
        bin_widths.expand(N, B), km1
    )  # (N, Y)  width of interval [b_km1, b_k] where y is included
    cdf_at_y = cdf_at_borderkm1 + (
        (
            (batch_y - get_values_at(bin_borders, km1))  # (N, Y)
            * (cdf_at_borderk - cdf_at_borderkm1)
            / (bin_widths_km1 + 1e-45)
        )
        * (1 * (0 < bin_widths_km1))
    )  # (N, Y)

    partC = (
        (cdf_at_y + cdf_at_borderk)  # (N, Y)
        / 2
        * (get_values_at(bin_borders, km1 + 1) - batch_y)
    ) * (
        (bin_borders[:, 0:1] < batch_y)
    ).float()  # non-zero if b_0 < y < b_B
    mask = torch.arange(B, device=batch_y.device).reshape(1, 1, B) > purekm1.view(
        N, Y, 1
    )  # (N, Y, B)
    partD = torch.sum(int_Fy_bis.view(N, 1, B) * mask, dim=2)  # (N, Y, B)  # (N, Y)

    crps = (
        partA + partB - 2 * (partC + partD) * (batch_y < bin_borders[:, -1:])
    )  # (N, Y)
    return crps


def test_laplace():
    torch.autograd.set_detect_anomaly(True)
    for fixed_pred_params, method in [
        (torch.tensor([[0.2, 0.5]]), LaplaceLogScore([12, 128, 128])),
        (torch.tensor([[0.2]]), LaplaceGlobalWidth([12, 128, 128])),
        (torch.tensor([[0.2]]), LaplaceGlobalWidth([12, 128, 128], train_width=False)),
    ]:
        method.train()  # try train mode
        batch_x = torch.rand(size=(128, 12))  # dummy input
        pred_params = method(batch_x)  # try forward
        target = torch.rand(size=(128,))  # dummy target
        loss = method.loss(target, pred_params)  # try loss
        # try learning step
        optim = torch.optim.SGD(method.parameters(), lr=1e-3)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # see if the F value is correct
        if isinstance(method, LaplaceGlobalWidth):
            method.global_width.data = torch.tensor([0.5])
        CDFaty = method.get_F_at_y(torch.tensor([[0.7]]), pred_params=fixed_pred_params)
        assert torch.isclose(CDFaty, torch.tensor([[0.816]]), atol=1e-3)
        # see if the f value is correct
        logscoreaty = method.get_logscore_at_y(
            torch.tensor([[0.7]]), pred_params=fixed_pred_params
        )
        assert torch.isclose(logscoreaty, torch.tensor([[1.0]]), atol=1e-3)


def test_gaussian():
    torch.autograd.set_detect_anomaly(True)
    method = MixtureDensityNetwork([12, 128, 128], 2)
    fixed_pred_params = torch.tensor([[0.5, 0.5, 0, 2.0, 1.0, 0.5]]).expand(
        4, 6
    )  # batch size will be 4
    method.train()
    batch_x = torch.rand(size=(128, 12))
    pred_params = method(batch_x)
    target = torch.rand(size=(128, 1))
    loss = method.loss(target, pred_params)
    optim = torch.optim.SGD(method.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # correct values come from wolframalpha
    # see if the F value is correct
    CDFaty = method.get_F_at_y(
        torch.tensor([-1, 0, 1, 2]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(4)
    assert torch.allclose(
        CDFaty, torch.tensor([0.0793, 0.2500, 0.4320, 0.7386]), atol=1e-3
    )
    # see if the f value is correct
    logscoreaty = method.get_logscore_at_y(
        torch.tensor([-1, 0, 1, 2]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(4)
    assert torch.allclose(
        logscoreaty, torch.tensor([2.1121, 1.6114, 1.7431, 0.8535]), atol=1e-3
    )


def test_categorical_cross_entropy():
    torch.autograd.set_detect_anomaly(True)

    # Assuming `MLP` and `CategoricalCrossEntropy` are defined as shown earlier
    # Setup model
    method = CategoricalCrossEntropy(
        [12, 128, 128], bin_borders=[0.0, 1.0, 2.0, 3.0, 4.0]
    )
    method.train()

    # Fixed predicted parameters for testing (batch size 4, B=4 bins)
    fixed_pred_params = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],  # distribution over bins
            [0.2, 0.2, 0.4, 0.2],
            [0.25, 0.25, 0.25, 0.25],
        ]
    ).expand(3, 4)

    # Create random batch_x for a forward pass
    batch_x = torch.rand(size=(128, 12))
    pred_params = method(batch_x)

    # Create random target values within the range of bin_borders
    target = torch.rand(size=(128, 1)) * 4  # since bin borders range from 0 to 4

    # Calculate loss
    loss = method.loss(target, pred_params)

    # Optimizer setup
    optim = torch.optim.SGD(method.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Testing the CDF values
    CDFaty = method.get_F_at_y(
        torch.tensor([0.5, 1.5, 2.5]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(3)
    assert torch.allclose(
        CDFaty, torch.tensor([0.05, 0.3, 0.625]), atol=1e-3
    ), f"CDF values are incorrect: {CDFaty}"

    # Testing the logscore values
    logscoreaty = method.get_logscore_at_y(
        torch.tensor([0.5, 1.5, 2.5]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(3)
    assert torch.allclose(
        logscoreaty, torch.tensor([2.3026, 1.6094, 1.3863]), atol=1e-3
    ), f"Log score values are incorrect: {logscoreaty}"


def test_pinball():
    torch.autograd.set_detect_anomaly(True)

    # Assuming `MLP` and `CategoricalCrossEntropy` are defined as shown earlier
    # Setup model
    method = PinballLoss(
        [12, 128, 128], quantile_levels=[0.2, 0.5, 0.8], bounds=[0.0, 1.0]
    )
    method.train()

    # Fixed predicted parameters for testing (batch size 3, B=4 bins)
    fixed_pred_params = torch.tensor(
        [
            [0.2, 0.5, 0.8],  # distribution over bins
            [0.2, 0.8, 0.9],
            [0.4, 0.5, 0.6],
        ]
    ).expand(3, 3)

    # Create random batch_x for a forward pass
    batch_x = torch.rand(size=(128, 12))
    pred_params = method(batch_x)

    # Create random target values within the range of bin_borders
    target = torch.rand(size=(128, 1))  # since bin borders range from 0 to 4

    # Calculate loss
    loss = method.loss(target, pred_params)

    # Optimizer setup
    optim = torch.optim.SGD(method.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Testing the CDF values
    CDFaty = method.get_F_at_y(
        torch.tensor([0.1, 0.5, 0.6]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(3)
    assert torch.allclose(
        CDFaty, torch.tensor([0.1, 0.35, 0.8]), atol=1e-3
    ), f"CDF values are incorrect: {CDFaty}"

    # Testing the logscore values
    logscoreaty = method.get_logscore_at_y(
        torch.tensor([0.1, 0.5, 0.6]).reshape(-1, 1), pred_params=fixed_pred_params
    ).reshape(3)
    assert torch.allclose(
        logscoreaty, torch.tensor([0, 0.6931, -1.0986]), atol=1e-3
    ), f"Log score values are incorrect: {logscoreaty}"


def test_crps():
    import numpy as np

    bs = np.array([0, 0.25, 0.5, 0.75, 1])
    B = len(bs) - 1
    pdf = [0.85, 0.25, 0.45, 0.25]
    pdf = pdf / np.sum(pdf)
    y = 0.3

    assert B == len(pdf)

    def integrate(low, high, points, func):
        dx = (high - low) / (points - 1)
        values = [func(low + i * dx) for i in range(points)]
        return np.trapz(values, dx=dx)

    cdf_values = np.array([0] + list(np.cumsum(pdf)))

    def cdf_func(x):
        return np.interp(x, bs, cdf_values)

    sq_diff = lambda x: (cdf_func(x) - 1 * (y <= x)) ** 2
    print("crps:", integrate(-1, 1, 100000, sq_diff))

    numerical = CategoricalCrossEntropy(
        layer_sizes=[1], bin_borders=bs
    ).get_numerical_CRPS(
        batch_y=torch.tensor([[y]]),
        pred_params=torch.tensor(pdf).view(1, -1),
        lower=-1,
        upper=1,
        count=10000,
    )
    print("our numerical crps", numerical)

    ourcrps = get_crps_PL(
        batch_y=torch.tensor([[y]]),
        cdf_at_borders=torch.tensor(np.array([cdf_values])),
        bin_masses=None,
        bin_borders=torch.tensor(np.array([bs])),
    )
    print("our crps:", ourcrps)


def test_all():
    test_laplace()
    test_gaussian()
    test_categorical_cross_entropy()
    test_pinball()
    test_crps()


if __name__ == "__main__":
    test_all()

method_names = [
    "laplacescore",
    "laplacecrps",
    "laplacewb",
    "mdn",
    "kde",
    "ce",
    "pinball",
    "crpshist",
    "crpsqr",
    "iqn",
]


def get_method(method_name):
    if method_name == "laplacescore":
        return LaplaceLogScore
    elif method_name == "laplacecrps":
        return LaplaceCRPS
    elif method_name == "laplacewb":
        return LaplaceGlobalWidth
    elif method_name == "mdn":
        return MixtureDensityNetwork
    elif method_name == "gaussian":
        return Gaussian
    elif method_name == "kde":
        return KDESampler
    elif method_name == "ce":
        return CategoricalCrossEntropy
    elif method_name == "pinball":
        return PinballLoss
    elif method_name == "crpshist":
        return CRPSHist
    elif method_name == "crpsqr":
        return CRPSQR
    elif method_name == "logscoreqr":
        return LogScoreQR
    elif method_name == "iqn":
        return IQN
    elif method_name == "mcd":
        return MCD
    else:
        raise ValueError(f"Unknown method name {method_name}")
