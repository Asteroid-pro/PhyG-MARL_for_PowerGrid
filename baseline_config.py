from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    episodes: int = 1000
    steps: int = 200
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    entropy_coef: float = 0.003
    ppo_epochs: int = 5
    recurrent_sequence_length: int = 32
    recurrent_burn_in: int = 0
    hidden_dim: int = 128
    test_seed_start: int = 60000
    test_seed_count: int = 100
    evaluation_interval: int = 50

    @property
    def test_seeds(self) -> list[int]:
        return list(range(self.test_seed_start, self.test_seed_start + self.test_seed_count))


METHODS = ("mappo", "ippo", "ppo", "gru_mappo", "gat_mappo", "qmix", "mat", "phyg_v2_2_2")
TRAINABLE_METHODS = METHODS[:-1]
SEEDS = (42, 43, 44)
