import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
from tqdm.auto import tqdm
import warnings

from config import Config


def _calculate_ig_single(model, single_input_tree, baseline_tree, target_idx, m_steps):
    """Core Integrated Gradients calculation for one sequence."""
    key = jax.random.PRNGKey(0)

    @eqx.filter_grad
    def grad_target_output_fn(interpolated_input):
        pred = model(interpolated_input, key=key)
        return pred[target_idx]

    alphas = jnp.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Interpolate inputs: Pytree leaves become shape (m_steps+1, *original_shape)
    interpolated_inputs = jax.vmap(
        lambda alpha: jax.tree.map(
            lambda x, b: b + alpha * (x - b),
            single_input_tree,
            baseline_tree,
        )
    )(alphas)

    # Calculate gradients at each interpolated step
    # Pytree leaves become shape (m_steps+1, *original_shape)
    # static_keys = ["graph"]
    # data_axes_spec = {key: (None if key in static_keys else 0) for key in interpolated_inputs}
    interpolated_grads = jax.vmap(grad_target_output_fn)(interpolated_inputs)

    # Average grads using trapezoidal rule approximation
    def trapezoid_avg(g):
        integral_approx = jnp.sum(g[1:] + g[:-1], axis=0) / 2.0
        return integral_approx / m_steps

    avg_grads = jax.tree.map(trapezoid_avg, interpolated_grads)

    # Calculate diff and final attribution: (input - baseline) * avg_grads
    input_diff = jax.tree.map(
        lambda x, b: x - b,
        single_input_tree,
        baseline_tree,
    )
    ig_attribs = jax.tree.map(
        lambda diff, avg_grad: diff * avg_grad,
        input_diff,
        avg_grads,
    )

    return ig_attribs


@eqx.filter_jit
def _get_batch_ig(model, batch, target_idx, m_steps):
    """Calculate IG attributions for a batch, focusing on last time step of dynamic features."""
    baseline = jax.tree.map(jnp.zeros_like, batch)
    if 'graph' in baseline.keys():
        baseline['graph'] = batch['graph']

    # Create an in_axes pytree that matches the batch structure.
    # Use `None` for static keys and `0` for keys to be batched over.
    # TODO: If we start training with mixes of different basins we will need to fix this.
    static_keys = ["graph"]
    data_axes_spec = {key: (None if key in static_keys else 0) for key in batch}

    # Vmap the single calculation over the batch
    batched_ig_fn = jax.vmap(
        _calculate_ig_single,
        in_axes=(None, data_axes_spec, data_axes_spec, None, None),
    )
    batch_ig_attribs_tree = batched_ig_fn(
        model,
        batch,
        baseline,
        target_idx,
        m_steps,
    )

    # Extract and concatenate dynamic features from the last time step
    feat_imp_list = []
    dynamic_feature_keys = list(batch["dynamic"].keys())  # Ensure consistent order

    for feat_group in dynamic_feature_keys:
        # Summing attributions across the sequence length dimension (axis=1)
        summed_attributions = jnp.nansum(
            batch_ig_attribs_tree["dynamic"][feat_group],
            axis=1,
        )
        feat_imp_list.append(summed_attributions)

    # Concatenate along the feature dimension: -> (batch, total_dynamic_features)
    final_ig_attributions = jnp.concatenate(
        feat_imp_list,
        axis=-1,
    )
    return final_ig_attributions


@eqx.filter_jit
def _get_batch_predictions(model, batch, target_idx):
    """Helper to get predictions for a batch."""
    key = jax.random.PRNGKey(0)  # keys are only used for dropout.

    def predict_single(single_input):
        return model(single_input, key=key)[target_idx]

    

    # Create an in_axes pytree that matches the batch structure.
    # Use `None` for static keys and `0` for keys to be batched over.
    # TODO: If we start training with mixes of different basins we will need to fix this.
    static_keys = ["graph"]
    axes_spec = {key: (None if key in static_keys else 0) for key in batch}
    # Vmap the prediction function
    return jax.vmap(predict_single, in_axes=(axes_spec,))(batch)


def get_intgrads_df(cfg: Config, model, dataloader, target, m_steps=50, max_iter=np.inf):
    """
    Calculates feature importance using Integrated Gradients.

    Args:
        cfg: Configuration object.
        model: The trained Equinox model.
        target: The target variable name.
        m_steps: Number of steps for the IG approximation.
        max_iter: Maximum number of batches to process.

    Returns:
        A pandas DataFrame with integrated gradients attributions.
    """
    # Set model to inference mode (no dropout)
    model = eqx.nn.inference_mode(model)

    targets = cfg.features.target
    target_idx = targets.index(target)

    results = {
        "basin": [],
        "date": [],
        target: [],
        "ig_attribs": [],
    }

    if (max_iter != np.inf) and (max_iter < len(dataloader)):
        n_iter = max_iter
    else:
        n_iter = len(dataloader)

    pbar = tqdm(
        dataloader,
        total=n_iter,
        disable=cfg.quiet,
        desc=f"Calculating IG for {target}",
    )

    # --- Main Loop ---
    for i, (basin, date, batch) in enumerate(pbar):
        if i >= n_iter:
            break

        # Remove auxiliary data if necessary before passing to model/IG calc
        if "dynamic_dt" in batch:
            batch.pop("dynamic_dt")

        # Calculate Predictions (more efficient to do it once here)
        batch_preds = _get_batch_predictions(model, batch, target_idx)
        y_denorm = dataloader.dataset.denormalize(batch_preds, target)

        # Calculate Integrated Gradients Attributions
        batch_ig_attributions = _get_batch_ig(model, batch, target_idx, m_steps)

        # Store results (use device_get to move from JAX arrays to host)
        results["basin"].extend(list(basin))
        results["date"].extend(list(date))
        results["ig_attribs"].append(jax.device_get(batch_ig_attributions))
        results[target].append(jax.device_get(y_denorm))

    final_ig_attributions = np.concatenate(results["ig_attribs"], axis=0)
    final_target_values = np.concatenate(results[target], axis=0)

    # Get feature names in the correct order
    dynamic_features = dataloader.dataset.features["dynamic"]
    feature_names = [f for k in dynamic_features for f in dynamic_features[k]]

    # Create DataFrame
    output_df = pd.DataFrame(
        {
            "basin": results["basin"],
            "date": results["date"],
            target: final_target_values,
        }
    )
    ig_cols_df = pd.DataFrame(final_ig_attributions, columns=feature_names)

    final_df = pd.concat(
        [
            output_df.reset_index(drop=True),
            ig_cols_df.reset_index(drop=True),
        ],
        axis=1,
    )
    final_df.set_index(["basin", "date"], inplace=True)

    return final_df


def save_all_intgrads(cfg: Config, model, dataloader, save_dir, m_steps=50):
    for target in cfg.features.target:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = get_intgrads_df(cfg, model, dataloader, target, m_steps)

        with open(save_dir / f"{target}_integrated_gradients.pkl", "wb") as file:
            pickle.dump(data, file)
