import argparse
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tqdm
from matplotlib.gridspec import GridSpec
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR


class BaseRewardModel(nn.Module):
    network: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        *,
        state_dim: int,
        output_dim: int = 1,
        num_layers: int = 4,
        hidden_dim: int = 512,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        for layer_index in range(num_layers):
            if layer_index == 0:
                layers.append(nn.Linear(state_dim, hidden_dim))
            elif layer_index == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_index < num_layers - 1:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the log probability of the given preference comparisons according to the
        model. If preferences[i] == 0, then state0 is preferred to state1, and vice
        versa.
        """

        reward0 = self.forward(state0)
        reward1 = self.forward(state1)
        reward_diff = reward0 - reward1
        reward_diff[preferences == 1] *= -1
        return -F.softplus(-reward_diff)


class MeanAndVarianceRewardModel(BaseRewardModel):
    def __init__(self, *args, max_std=np.inf, **kwargs):
        super().__init__(*args, output_dim=2, **kwargs)
        self.max_std = max_std

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean_and_log_std = self.network(state)
        # return mean_and_log_std
        mean = mean_and_log_std[:, 0]
        log_std = mean_and_log_std[:, 1] - 2
        log_std = log_std.clamp(max=np.log(self.max_std))
        return torch.stack([mean, log_std], dim=1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        output0 = self.forward(state0)
        output1 = self.forward(state1)
        mean0 = output0[:, 0]
        log_std0 = output0[:, 1]
        mean1 = output1[:, 0]
        log_std1 = output1[:, 1]

        diff_mean = mean0 - mean1
        diff_mean[preferences == 1] *= -1
        var_combined = torch.exp(log_std0) ** 2 + torch.exp(log_std1) ** 2
        # p: torch.Tensor = Normal(0, torch.sqrt(var_combined)).cdf(diff_mean)
        z = diff_mean / torch.sqrt(var_combined)
        # Based on approximation here: https://stats.stackexchange.com/a/452121
        return -F.softplus(-z * np.sqrt(2 * np.pi))
        # logp = torch.log(p.clamp(min=1e-4))
        # return logp


class CategoricalRewardModel(BaseRewardModel):
    comparison_matrix: torch.Tensor

    def __init__(
        self, *args, num_atoms: Optional[int] = None, state_dim: int, **kwargs
    ):
        if num_atoms is None:
            if state_dim == 1:
                num_atoms = 20
            else:
                num_atoms = 8
        super().__init__(
            *args,
            output_dim=num_atoms,
            use_batchnorm=True,
            state_dim=state_dim,
            **kwargs,
        )

        comparison_matrix = torch.empty((num_atoms, num_atoms))
        atom_values = torch.linspace(0, 1, num_atoms)
        comparison_matrix[:] = atom_values[None, :] > atom_values[:, None]
        comparison_matrix[atom_values[None, :] == atom_values[:, None]] = 0.5
        self.register_buffer("comparison_matrix", comparison_matrix)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(state), dim=-1)

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        dist0 = self.forward(state0)
        dist1 = self.forward(state1)
        prob1 = ((dist0 @ self.comparison_matrix) * dist1).sum(dim=1)
        prob = prob1.clone()
        prob[preferences == 0] = (1 - prob1)[preferences == 0]
        return prob.log()


class ClassifierRewardModel(BaseRewardModel):
    def __init__(self, *args, state_dim, **kwargs):
        super().__init__(*args, state_dim=state_dim * 2, **kwargs)

    def forward(self, state0: torch.Tensor, state1: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.network(torch.cat([state0, state1], dim=-1))

    def preference_logp(
        self, state0: torch.Tensor, state1: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the log probability of the given preference comparisons according to the
        model. If preferences[i] == 0, then state0 is preferred to state1, and vice
        versa.
        """

        logits = self.forward(state0, state1)
        logits[preferences == 0] *= -1
        return -F.softplus(-logits)


def train_rlhf(
    *,
    reward_model: BaseRewardModel,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_state: Callable[[int], torch.Tensor],
    batch_size: int,
    lr: float,
    num_iterations: int,
    device: torch.device,
) -> BaseRewardModel:
    """
    Trains and returns a reward function using RLHF.
    Args:
        reward_fn: The ground truth reward/utility function, which can be randomized.
        state_dim: The dimension of the state space.
        sample_state: A function that samples states from the state space, i.e.
            sample_state(n) returns n samples from the state space.
        batch_size: The batch size for training the reward function.
        num_iterations: The number of iterations to train the reward function.
    Returns:
        The trained reward function as a PyTorch neural network.
    """

    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=(1e-5 / lr) ** (1 / num_iterations))
    reward_model.to(device).train()
    progress_bar = tqdm.tqdm(range(num_iterations))
    for _ in progress_bar:
        optimizer.zero_grad()
        state0 = sample_state(batch_size).to(device)
        state1 = sample_state(batch_size).to(device)
        rewards0 = reward_fn(state0)
        rewards1 = reward_fn(state1)
        preferences = (rewards1 > rewards0).long()
        loss = -reward_model.preference_logp(state0, state1, preferences).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_description(
            f"loss = {loss.item():.2f}    lr = {scheduler.get_lr()[0]:.2e}"  # type: ignore
        )

    return reward_model


def reward_fn_1d(state: torch.Tensor) -> torch.Tensor:
    state = state.squeeze(-1)
    rewards = state.clone()
    double_rewards = torch.rand(rewards.shape, device=rewards.device) < 0.5
    rewards[(state >= 0.8) & double_rewards] *= 2
    rewards[(state >= 0.8) & ~double_rewards] *= 0
    return rewards


def reward_fn_2d(state: torch.Tensor) -> torch.Tensor:
    x, y = state[..., 0], state[..., 1]
    b = torch.rand(x.shape, device=x.device) < 1 - x * y
    rewards = torch.empty_like(x)
    rewards[b] = (y / (1 - x * y))[b]
    rewards[~b] = 0
    return rewards


def main(  # noqa: C901
    *,
    env_name: Literal["1d", "2d"],
    batch_size: int,
    lr: float,
    num_iterations: int,
    out_dir: str,
    reward_model_kwargs: Dict[str, Any] = {},
):
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    state_dim: int

    if env_name == "1d":
        reward_fn = reward_fn_1d
        state_dim = 1
    elif env_name == "2d":
        reward_fn = reward_fn_2d
        state_dim = 2

    sample_state = lambda n: torch.rand(n, state_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_models: Dict[str, BaseRewardModel] = {}
    reward_model_class: Type[BaseRewardModel]
    for reward_model_class in [
        BaseRewardModel,
        MeanAndVarianceRewardModel,
        CategoricalRewardModel,
        ClassifierRewardModel,
    ]:
        print(f"Training {reward_model_class.__name__}...")
        kwargs = dict(reward_model_kwargs)
        # if env_name == "1d" and reward_model_class is MeanAndVarianceRewardModel:
        #     kwargs["max_std"] = 1
        reward_model = reward_model_class(state_dim=state_dim, **kwargs)
        reward_model = train_rlhf(
            reward_model=reward_model,
            reward_fn=reward_fn,
            sample_state=sample_state,
            batch_size=batch_size,
            lr=lr,
            num_iterations=num_iterations,
            device=device,
        )
        reward_model.eval()
        reward_models[reward_model_class.__name__] = reward_model

    experiment_dir = os.path.join(
        out_dir,
        env_name,
        f"{batch_size}_{lr}_{num_iterations}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving results to {experiment_dir}...")

    # Matplotlib configuration
    latex_preamble = r"""
    \usepackage{times}
    \usepackage{amsmath}
    \usepackage{mathptmx}
    \usepackage{xcolor}

    \newcommand{\oracle}[1]{{O_{#1}}}
    \newcommand{\utility}{u}
    \newcommand{\exutility}{\bar{\utility}}
    \newcommand{\learnedutility}{\hat{\utility}}
    \newcommand{\loss}{L}
    \newcommand{\noise}{\epsilon}
    \newcommand{\noisedist}{\mathcal{D}_\noise}
    \newcommand{\bordacount}{\text{BC}}
    \newcommand{\altspace}{\mathcal{A}}
    \newcommand{\alta}{a}
    \newcommand{\altb}{b}
    \newcommand{\altc}{c}
    \newcommand{\unseenspace}{\mathcal{Z}}
    \newcommand{\unseen}{z}
    \newcommand{\unseendist}{\mathcal{D}_\unseen}
    \newcommand{\comparisonprob}{p}
    \newcommand{\btlprob}{\comparisonprob^\text{BTL}}
    \newcommand{\uniformdist}{\text{Unif}}
    \newcommand{\bernoulli}{\mathcal{B}}
    \newcommand{\learneddist}{\smash{\hat{\mathcal{D}}}}
    """
    matplotlib.rc("text", usetex=True)
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rc("text.latex", preamble=latex_preamble)
    matplotlib.rc("font", size=9)
    matplotlib.rc("pgf", rcfonts=False, texsystem="pdflatex", preamble=latex_preamble)
    figure = plt.figure(figsize=(5.5, 1.2))
    cmap = "Greys"
    color = "k"
    title_kwargs = dict(fontsize=9)

    if env_name == "1d":
        x = torch.linspace(0, 1, 100, device=device)
        num_rows = 1
        num_cols = 3

        # Combine first three subplots into one
        combined_ax = figure.add_subplot(num_rows, num_cols, 1)
        combined_ax.set_title("Normal preference learning", **title_kwargs)
        combined_ax.set_xlabel(r"$\alta$")
        combined_ax.set_xlim(0, 1)

        # Base Reward Model Output
        rewards = reward_models["BaseRewardModel"](x[:, None]).detach().cpu().tolist()
        combined_ax.plot(
            x.cpu(),
            rewards,
            color="k",
            label=r"$\learnedutility(\alta)$",
        )
        combined_ax.set_ylim(
            [
                -0.2 * max(rewards) + 1.2 * min(rewards),
                1.2 * max(rewards) - 0.2 * min(rewards),
            ]
        )
        combined_ax.annotate(
            r"$\learnedutility(\alta)$",
            xy=(x[65].item(), rewards[65]),
            textcoords="offset points",
            xytext=(-3, 3),
            color="k",
            ha="right",
        )

        # combined_ax.set_ylabel(r"$\text{Learned Utility}(\alpha)$", color=color)
        # combined_ax.tick_params(axis='y', labelcolor=color)

        # Creating twin X-axis for Borda Count and Expected Utility
        twin_ax = combined_ax.twinx()

        # Borda Count
        bc = torch.empty_like(x)
        bc[x < 0.8] = (0.1 + x)[x < 0.8]
        bc[x >= 0.8] = (0.275 + 0.25 * x)[x >= 0.8]
        twin_ax.plot(x.cpu(), bc.cpu(), color="tab:blue", label=r"$\bordacount(\alta)$")
        twin_ax.annotate(
            r"$\bordacount(\alta)$",
            xy=(x[35].item(), bc[35].item()),
            textcoords="offset points",
            xytext=(-3, 3),
            color="tab:blue",
            ha="right",
        )

        # Expected Utility Function
        twin_ax.plot(x.cpu(), x.cpu(), color="tab:red", label=r"$\exutility(\alta)$")
        twin_ax.set_yticks([])
        twin_ax.annotate(
            r"$\exutility(\alta)$",
            xy=(x[50].item(), x[50].item()),
            textcoords="offset points",
            xytext=(3, -5),
            color="tab:red",
            ha="left",
        )
        combined_ax.set_yticks([])

        # twin_ax.set_ylabel(r"$\text{Utility Measures}$", color='C1')
        # twin_ax.tick_params(axis='y', labelcolor='C1')

        # Adding legend
        # lines, labels = combined_ax.get_legend_handles_labels()
        # lines2, labels2 = twin_ax.get_legend_handles_labels()
        # twin_ax.legend(
        #     lines + lines2,
        #     labels + labels2,
        #     loc="upper left",
        #     handlelength=1,
        #     frameon=False,
        #     borderaxespad=0.1,
        # )

        # Mean and Variance Subplot
        mean_and_variance_ax = figure.add_subplot(num_rows, num_cols, 2)
        mean_and_variance_ax.set_title("DPL (mean and variance)", **title_kwargs)
        mean_and_variance_ax.set_xlabel(r"$\alta$")
        mean_and_variance_ax.set_ylabel(r"$\hat{\mu}(\alta) \pm \hat{\sigma}(\alta)$")
        mean_and_variance_ax.set_xlim(0, 1)
        mean_and_log_std = (
            reward_models["MeanAndVarianceRewardModel"](x[:, None]).detach().cpu()
        )
        mean = mean_and_log_std[:, 0]
        std = torch.exp(mean_and_log_std[:, 1])
        mean_and_variance_ax.plot(x.cpu(), mean, color=color)
        ymin = mean.min() - 1
        ymax = mean.max() + 1
        mean_and_variance_ax.fill_between(
            x.cpu(),
            np.clip(mean - std, ymin, ymax),
            np.clip(mean + std, ymin, ymax),
            alpha=0.2,
            color=color,
        )
        mean_and_variance_ax.set_ylim(mean.min() - 1, mean.max() + 1)

        # Categorical Subplot
        categorical_ax = figure.add_subplot(num_rows, num_cols, 3)
        categorical_ax.set_title("DPL (categorical)", **title_kwargs)
        categorical_ax.set_xlabel(r"$\alta$")
        categorical_ax.set_ylabel(r"$\hat{p}(\utility \mid \alta)$")
        categorical_ax.set_xlim(0, 1)
        dist = reward_models["CategoricalRewardModel"](x[:, None]).detach().cpu()
        categorical_ax.imshow(
            dist.transpose(0, 1),
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="auto",
            cmap=cmap,
        )

        figure.tight_layout(pad=0.1)
        figure.subplots_adjust(wspace=0.7, left=0.05, right=0.95)
    elif env_name == "2d":
        x = torch.linspace(0, 1, 20, device=device)
        y = torch.linspace(0, 1, 20, device=device)
        xx, yy = torch.meshgrid(x, y)
        xxyy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        num_rows = 2
        num_cols = 3

        def setup_ax(ax):
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        true_reward_ax = figure.add_subplot(num_rows, num_cols, 1)
        true_reward_ax.set_title(
            "True utility function\n$\\utility(x, y)$", **title_kwargs
        )
        setup_ax(true_reward_ax)
        true_reward_ax.contourf(xx.cpu(), yy.cpu(), yy.cpu(), cmap=cmap)

        borda_count_ax = figure.add_subplot(num_rows, num_cols, 2)
        borda_count_ax.set_title("Borda count\n$\\bordacount(x, y)$", **title_kwargs)
        setup_ax(borda_count_ax)
        bc = (
            0.25
            + yy
            - 0.25 * yy**2
            - 0.125 * xx * yy
            - xx * yy**2
            + 0.25 * xx * yy**3
        )
        borda_count_ax.contourf(xx.cpu(), yy.cpu(), bc.cpu(), cmap=cmap)

        base_ax = figure.add_subplot(num_rows, num_cols, 3)
        base_ax.set_title(
            "Preference learning\n$\\learnedutility(x, y)$", **title_kwargs
        )
        setup_ax(base_ax)
        uhat = reward_models["BaseRewardModel"](xxyy).detach().cpu().reshape(xx.size())
        base_ax.contourf(xx.cpu(), yy.cpu(), uhat, cmap="Blues")

        mean_ax = figure.add_subplot(num_rows, num_cols, 4)
        var_ax = figure.add_subplot(num_rows, num_cols, 5)
        mean_ax.set_title("\n$\\learnedutility(x, y)$", **title_kwargs)
        var_ax.set_title("\n$\\hat{\\sigma}(x, y)$", **title_kwargs)
        setup_ax(mean_ax)
        setup_ax(var_ax)
        mean_and_log_std = (
            reward_models["MeanAndVarianceRewardModel"](xxyy)
            .detach()
            .cpu()
            .reshape(xx.size() + (2,))
        )
        mean = mean_and_log_std[..., 0]
        std = torch.exp(mean_and_log_std[..., 1])
        mean_ax.contourf(xx.cpu(), yy.cpu(), mean, cmap=cmap)
        var_ax.contourf(xx.cpu(), yy.cpu(), std, cmap=cmap)
        mean_and_var_ax = figure.add_subplot(
            GridSpec(num_rows, num_cols - 1, width_ratios=[2, 1])[2], frameon=False
        )
        mean_and_var_ax.set_title(
            "Preference learning (mean and variance)\n", **title_kwargs
        )
        mean_and_var_ax.xaxis.set_visible(False)
        mean_and_var_ax.yaxis.set_visible(False)

        categorical_ax = figure.add_subplot(num_rows, num_cols, 6)
        categorical_ax.set_title("Preference learning (categorical)", **title_kwargs)
        setup_ax(categorical_ax)
        bins = 4
        x = (torch.arange(bins, device=device).float() + 0.5) / bins
        y = (torch.arange(bins, device=device).float() + 0.5) / bins
        xx, yy = torch.meshgrid(x, y)
        xxyy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        dists = (
            reward_models["CategoricalRewardModel"](xxyy)
            .detach()
            .cpu()
            .reshape(*xx.size(), -1)
        )
        for i in range(1, bins):
            categorical_ax.axvline(i / bins, color="black", linewidth=1)
            categorical_ax.axhline(i / bins, color="black", linewidth=1)
        for i in range(bins):
            for j in range(bins):
                dist = dists[i, j]
                num_atoms = dist.size()[0]
                categorical_ax.bar(
                    np.linspace(i / bins, (i + 1) / bins, num_atoms + 3)[1:-2],
                    dist / dists.max() / bins,
                    width=1 / bins / num_atoms,
                    bottom=j / bins,
                    align="edge",
                    color=color,
                )

        figure.tight_layout(pad=0.1)
        figure.subplots_adjust(hspace=1, wspace=0.5)

    figure.savefig(os.path.join(experiment_dir, f"{env_name}.png"), dpi=300)
    figure.savefig(os.path.join(experiment_dir, f"{env_name}.pgf"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="1d")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    main(
        env_name=args.env,
        batch_size=args.batch_size,
        lr=args.lr,
        num_iterations=args.num_iterations,
        out_dir=args.out_dir,
    )
