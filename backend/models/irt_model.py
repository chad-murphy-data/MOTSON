"""
MOTSON Bayesian IRT Model

Implements the full Bayesian IRT approach:
- theta: ability to win matches (how good the team is at "answering" opponents)
- b_home: difficulty to beat at home (how hard it is for opponents to win at this ground)
- b_away: difficulty to beat away (how hard it is for opponents to win against this team on the road)

Key insight: In football, every team is simultaneously:
- A "test-taker" trying to beat their opponent (using their theta)
- A "test question" being answered by their opponent (represented by their b)

Match prediction:
    m_home = theta_home - b_away_opponent  (home team's attack margin)
    m_away = theta_away - b_home_opponent  (away team's attack margin)
    gap = m_home - m_away

Gap is then mapped to [P(home), P(draw), P(away)] using empirically calibrated anchor points.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Use CPU for JAX (more compatible, fast enough for 20 teams)
jax.config.update('jax_platform_name', 'cpu')


@dataclass
class IRTParameters:
    """Container for IRT parameter estimates with uncertainties."""
    theta: Dict[str, float]  # Team -> theta mean
    theta_se: Dict[str, float]  # Team -> theta standard error
    b_home: Dict[str, float]  # Team -> b_home mean
    b_home_se: Dict[str, float]  # Team -> b_home standard error
    b_away: Dict[str, float]  # Team -> b_away mean
    b_away_se: Dict[str, float]  # Team -> b_away standard error


# Empirically calibrated anchor table for gap -> probabilities
# Symmetric around 0 - home advantage is captured in b_home vs b_away differential
GAP_ANCHORS = [
    # (gap, P(home_win), P(draw), P(away_win))
    (-2.0, 0.02, 0.05, 0.93),
    (-1.5, 0.03, 0.07, 0.90),
    (-1.0, 0.05, 0.10, 0.85),
    (-0.5, 0.20, 0.27, 0.53),
    (0.0, 0.33, 0.34, 0.33),
    (0.5, 0.53, 0.27, 0.20),
    (1.0, 0.85, 0.10, 0.05),
    (1.5, 0.90, 0.07, 0.03),
    (2.0, 0.93, 0.05, 0.02),
]


def gap_to_probabilities(gap: float) -> Tuple[float, float, float]:
    """
    Convert gap to (home_win_prob, draw_prob, away_win_prob) using anchor interpolation.

    Args:
        gap: m_home - m_away where m_home = theta_home - b_away_opponent

    Returns:
        Tuple of (home_win_prob, draw_prob, away_win_prob)
    """
    # Clamp gap to anchor range
    gap = max(-2.0, min(2.0, gap))

    # Find surrounding anchors
    for i in range(len(GAP_ANCHORS) - 1):
        g1, h1, d1, a1 = GAP_ANCHORS[i]
        g2, h2, d2, a2 = GAP_ANCHORS[i + 1]

        if g1 <= gap <= g2:
            # Linear interpolation
            t = (gap - g1) / (g2 - g1)
            h = h1 + t * (h2 - h1)
            d = d1 + t * (d2 - d1)
            a = a1 + t * (a2 - a1)

            # Normalize to sum to 1 (handles floating point errors)
            total = h + d + a
            return h / total, d / total, a / total

    # Fallback (shouldn't reach here)
    return 0.33, 0.34, 0.33


def gap_to_logits(gap: jnp.ndarray) -> jnp.ndarray:
    """
    Convert gap to logits for categorical distribution (JAX-compatible).

    For MCMC, we need a differentiable function. We approximate the anchor table
    with a smooth function that captures the key behaviors:
    - High positive gap -> high home win probability
    - High negative gap -> high away win probability
    - Near zero -> roughly even with slight draw boost
    - Draw probability peaks at gap=0 and decreases with |gap|

    Returns logits [away_win, draw, home_win]
    """
    # Scale factor for sharpness of probability transitions
    k = 1.5

    # Home/away logits: linear in gap
    logit_home = k * gap
    logit_away = -k * gap

    # Draw logit: peaks at 0, decreases with |gap|
    # Use negative quadratic to create peak
    logit_draw = 0.2 - 0.6 * jnp.abs(gap)

    return jnp.stack([logit_away, logit_draw, logit_home], axis=-1)


def irt_model_gpcm(
    home_team_ids: jnp.ndarray,
    away_team_ids: jnp.ndarray,
    outcomes: jnp.ndarray = None,
    n_teams: int = 20,
    theta_prior_mean: jnp.ndarray = None,
    theta_prior_std: float = 0.5,
    b_prior_std: float = 0.5,
):
    """
    NumPyro model for MOTSON IRT.

    Each team has:
    - theta: ability to win (unbounded, mean ~0)
    - b_home: difficulty to beat at home
    - b_away: difficulty to beat away

    Args:
        home_team_ids: Array of home team indices for each match
        away_team_ids: Array of away team indices for each match
        outcomes: Array of outcomes (0=away_win, 1=draw, 2=home_win), None for prediction
        n_teams: Number of teams
        theta_prior_mean: Prior means for theta (from 5-year GPCM), None for uninformative
        theta_prior_std: Prior std for theta
        b_prior_std: Prior std for b parameters
    """
    # Priors for team parameters
    if theta_prior_mean is not None:
        # Informative priors from historical data
        theta = numpyro.sample(
            "theta",
            dist.Normal(theta_prior_mean, theta_prior_std)
        )
    else:
        # Uninformative priors for fresh estimation
        theta = numpyro.sample(
            "theta",
            dist.Normal(jnp.zeros(n_teams), theta_prior_std)
        )

    # b_home and b_away priors - centered on theta (teams that win also tend to not lose)
    # But with independent variation to capture teams with different profiles
    b_home = numpyro.sample(
        "b_home",
        dist.Normal(theta, b_prior_std)  # Prior centered on theta
    )

    b_away = numpyro.sample(
        "b_away",
        dist.Normal(theta - 0.15, b_prior_std)  # Away defense typically slightly weaker
    )

    # Match predictions
    # m_home = theta[home] - b_away[away]  (home team's attack vs away team's away defense)
    # m_away = theta[away] - b_home[home]  (away team's attack vs home team's home defense)
    m_home = theta[home_team_ids] - b_away[away_team_ids]
    m_away = theta[away_team_ids] - b_home[home_team_ids]

    # Gap from home team's perspective
    gap = m_home - m_away

    # Convert to logits
    logits = gap_to_logits(gap)

    # Likelihood
    with numpyro.plate("matches", len(home_team_ids)):
        numpyro.sample("outcome", dist.Categorical(logits=logits), obs=outcomes)


def fit_irt_model(
    matches: List[Dict],
    team_to_idx: Dict[str, int],
    theta_prior_mean: Dict[str, float] = None,
    theta_prior_std: float = 0.5,
    n_warmup: int = 500,
    n_samples: int = 1000,
    seed: int = 42,
) -> IRTParameters:
    """
    Fit the IRT model using MCMC.

    Args:
        matches: List of match dicts with keys: home_team, away_team, outcome (0/1/2)
        team_to_idx: Mapping from team name to index
        theta_prior_mean: Prior means from 5-year estimation (optional)
        theta_prior_std: Prior standard deviation for theta
        n_warmup: Number of warmup samples
        n_samples: Number of posterior samples
        seed: Random seed

    Returns:
        IRTParameters with posterior means and standard errors
    """
    n_teams = len(team_to_idx)
    idx_to_team = {v: k for k, v in team_to_idx.items()}

    # Convert matches to arrays
    home_ids = jnp.array([team_to_idx[m["home_team"]] for m in matches])
    away_ids = jnp.array([team_to_idx[m["away_team"]] for m in matches])
    outcomes = jnp.array([m["outcome"] for m in matches])

    # Convert prior means to array if provided
    prior_mean_array = None
    if theta_prior_mean:
        prior_mean_array = jnp.array([
            theta_prior_mean.get(idx_to_team[i], 0.0)
            for i in range(n_teams)
        ])

    logger.info(f"Fitting IRT model on {len(matches)} matches for {n_teams} teams...")

    # Run MCMC
    kernel = NUTS(irt_model_gpcm)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=1,  # Single chain for speed
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        home_team_ids=home_ids,
        away_team_ids=away_ids,
        outcomes=outcomes,
        n_teams=n_teams,
        theta_prior_mean=prior_mean_array,
        theta_prior_std=theta_prior_std,
    )

    # Extract posterior samples
    samples = mcmc.get_samples()

    # Calculate means and standard errors
    theta_samples = samples["theta"]
    b_home_samples = samples["b_home"]
    b_away_samples = samples["b_away"]

    theta = {idx_to_team[i]: float(theta_samples[:, i].mean()) for i in range(n_teams)}
    theta_se = {idx_to_team[i]: float(theta_samples[:, i].std()) for i in range(n_teams)}

    b_home = {idx_to_team[i]: float(b_home_samples[:, i].mean()) for i in range(n_teams)}
    b_home_se = {idx_to_team[i]: float(b_home_samples[:, i].std()) for i in range(n_teams)}

    b_away = {idx_to_team[i]: float(b_away_samples[:, i].mean()) for i in range(n_teams)}
    b_away_se = {idx_to_team[i]: float(b_away_samples[:, i].std()) for i in range(n_teams)}

    # Center theta (mean = 0)
    mean_theta = np.mean(list(theta.values()))
    theta = {k: v - mean_theta for k, v in theta.items()}
    # Also shift b values to maintain relative relationships
    b_home = {k: v - mean_theta for k, v in b_home.items()}
    b_away = {k: v - mean_theta for k, v in b_away.items()}

    logger.info(f"IRT fitting complete. Theta range: [{min(theta.values()):.3f}, {max(theta.values()):.3f}]")

    return IRTParameters(
        theta=theta,
        theta_se=theta_se,
        b_home=b_home,
        b_home_se=b_home_se,
        b_away=b_away,
        b_away_se=b_away_se,
    )


def bayesian_blend(
    prior_mean: float,
    prior_std: float,
    likelihood_mean: float,
    likelihood_std: float,
    min_prior_weight: float = 0.5,
) -> Tuple[float, float]:
    """
    Combine prior (gravity) and likelihood (current season) using Bayesian updating.

    Posterior = (prior_mean/prior_var + likelihood_mean/likelihood_var) / (1/prior_var + 1/likelihood_var)

    With a floor on prior weight to ensure gravity is always at least 50%.

    Args:
        prior_mean: Prior mean (from 5-year estimation)
        prior_std: Prior standard deviation
        likelihood_mean: Current season estimate
        likelihood_std: Standard error from current season
        min_prior_weight: Minimum weight for prior (default 0.5)

    Returns:
        (posterior_mean, posterior_std)
    """
    prior_var = prior_std ** 2
    likelihood_var = likelihood_std ** 2

    # Calculate natural weights
    prior_precision = 1 / prior_var
    likelihood_precision = 1 / likelihood_var
    total_precision = prior_precision + likelihood_precision

    natural_prior_weight = prior_precision / total_precision
    natural_likelihood_weight = likelihood_precision / total_precision

    # Apply floor on prior weight
    if natural_prior_weight < min_prior_weight:
        prior_weight = min_prior_weight
        likelihood_weight = 1 - min_prior_weight
    else:
        prior_weight = natural_prior_weight
        likelihood_weight = natural_likelihood_weight

    # Blended estimate
    posterior_mean = prior_weight * prior_mean + likelihood_weight * likelihood_mean

    # Posterior variance (using actual weights, not floored)
    posterior_var = 1 / total_precision
    posterior_std = np.sqrt(posterior_var)

    return posterior_mean, posterior_std


def predict_match(
    home_team: str,
    away_team: str,
    params: IRTParameters,
) -> Tuple[float, float, float, float]:
    """
    Predict match outcome probabilities.

    Args:
        home_team: Home team name
        away_team: Away team name
        params: IRT parameters

    Returns:
        (home_win_prob, draw_prob, away_win_prob, gap)
    """
    # Attack margins
    m_home = params.theta[home_team] - params.b_away[away_team]
    m_away = params.theta[away_team] - params.b_home[home_team]

    # Gap from home team's perspective
    gap = m_home - m_away

    # Convert to probabilities
    h_prob, d_prob, a_prob = gap_to_probabilities(gap)

    return h_prob, d_prob, a_prob, gap


def calculate_informativeness(theta_diff: float, scale: float = 1.0) -> float:
    """
    Calculate how informative a match is based on theta difference.

    Matches between similar teams are more informative (could go either way).
    Matches between mismatched teams are less informative (predictable).

    Args:
        theta_diff: Absolute difference in team thetas
        scale: Scale parameter for decay (default 1.0)

    Returns:
        Informativeness weight in [0, 1]
    """
    return np.exp(-(theta_diff ** 2) / (2 * scale ** 2))
