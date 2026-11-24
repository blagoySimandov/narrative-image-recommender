import itertools
from collections.abc import Sequence

import altair as alt
import numpy as np
import polars as pl
from scipy import stats

from dataset_loaders import load_yfcc_human_summaries


def extended_tau(list_a, list_b):
    """Calculate the extended Kendall tau from two top-k lists of equal length."""
    n = len(list_a)
    if len(list_b) != n:
        raise ValueError("Both lists must have the same length")

    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).with_columns(
        pl.col("rank_a", "rank_b").fill_null(n)
    )

    n_pad = 2 * n - ranks.height
    dummy_df = pl.DataFrame(
        {"rank_a": [n] * n_pad, "rank_b": [n] * n_pad},
        schema={"rank_a": pl.Int64, "rank_b": pl.Int64},
    )
    total_df = pl.concat([ranks.select("rank_a", "rank_b"), dummy_df])

    tau = stats.kendalltau(
        total_df["rank_a"].to_numpy(), total_df["rank_b"].to_numpy()
    ).statistic
    return scale_tau(n, tau)


def scale_tau(length, value):
    """Scale an extended tau correlation such that it falls in [-1, +1]."""
    n_0 = 2 * length * (2 * length - 1)
    n_a = length * (length - 1)
    n_d = n_0 - n_a
    min_tau = (2.0 * n_a - n_0) / n_d
    return 2 * (value - min_tau) / (1 - min_tau) - 1


def create_rank(a):
    """Convert an ordered list to a DataFrame of (key, rank)."""
    return pl.DataFrame(
        {"key": a, "rank": pl.int_range(len(a), eager=True, dtype=pl.Int64)}
    )


def join_ranks(rank_a, rank_b):
    """Full outer join of two rank DataFrames on key."""
    return rank_a.rename({"rank": "rank_a"}).join(
        rank_b.rename({"rank": "rank_b"}), on="key", how="full", coalesce=True
    )


def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set)


def f_measure(pred, ref):
    pred_set, ref_set = set(pred), set(ref)
    return 2 * len(pred_set & ref_set) / (len(pred_set) + len(ref_set))


METRICS = {
    "Extended Kendall Tau": extended_tau,
    "Jaccard": jaccard,
    "F-measure": f_measure,
}
BASELINE_NAMES = ("Random", "Human")


def human_summaries(city: str, k: int) -> list[np.ndarray]:
    return [np.asarray(r)[:k] for r in load_yfcc_human_summaries(city)]


def _random_scores(fn, human, all_ids, k, n_trials):
    rng = np.random.default_rng(0)
    trials = [
        [fn(rng.choice(all_ids, size=k, replace=False), h) for h in human]
        for _ in range(n_trials)
    ]
    return np.mean(trials, axis=0).tolist()


def _human_scores(fn, human):
    pairs = itertools.combinations(range(len(human)), 2)
    return [float(fn(human[i], human[j])) for i, j in pairs]


def _check_inputs(systems, human, all_ids, k):
    assert all(len(ids) == k for ids in systems.values()), "all systems need the same k"
    assert all(len(h) == k for h in human), "a human summary is shorter than k"
    missing = set().union(*map(set, human)) - set(all_ids)
    assert not missing, f"{len(missing)} human-picked images missing from the dataset"


def score_systems(
    systems: dict[str, Sequence[str]], city: str, all_ids, n_trials: int = 200
) -> pl.DataFrame:
    """One row per (metric, comparison, human summary or pair). Systems map name -> ordered image_ids."""
    k = len(next(iter(systems.values())))
    human = human_summaries(city, k)
    _check_inputs(systems, human, all_ids, k)
    rows = []
    for metric, fn in METRICS.items():
        values = {
            "Random": _random_scores(fn, human, all_ids, k, n_trials),
            "Human": _human_scores(fn, human),
        }
        values |= {
            name: [float(fn(ids, h)) for h in human] for name, ids in systems.items()
        }
        rows += [
            {"metric": metric, "comparison": name, "value": v}
            for name, vals in values.items()
            for v in vals
        ]
    return pl.DataFrame(rows)


def summarise(scores: pl.DataFrame) -> pl.DataFrame:
    names = scores["comparison"].unique(maintain_order=True).to_list()
    systems = [n for n in names if n not in BASELINE_NAMES]
    means = (
        scores.group_by("metric", "comparison", maintain_order=True)
        .agg(pl.col("value").mean())
        .pivot(on="comparison", index="metric", values="value")
    )
    return means.select("metric", "Random", *systems, "Human")


def normalise(summary: pl.DataFrame) -> pl.DataFrame:
    """(system - random) / (human - random). 0 is random, 1 is human agreement."""
    systems = [c for c in summary.columns if c != "metric" and c not in BASELINE_NAMES]
    return summary.select(
        "metric",
        *[
            (
                (pl.col(s) - pl.col("Random")) / (pl.col("Human") - pl.col("Random"))
            ).alias(s)
            for s in systems
        ],
    )


def comparison_chart(scores: pl.DataFrame):
    order = [
        "Random",
        *[
            c
            for c in scores["comparison"].unique(maintain_order=True)
            if c not in BASELINE_NAMES
        ],
        "Human",
    ]
    base = alt.Chart(scores).encode(
        x=alt.X("comparison:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)),
        color=alt.Color("comparison:N", sort=order, legend=None),
    )
    points = (
        base.mark_circle(size=55, opacity=0.55)
        .encode(
            y=alt.Y("value:Q", title="score"),
            xOffset=alt.XOffset("jitter:Q"),
            tooltip=["comparison", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .transform_calculate(jitter="random()")
    )
    mean = base.mark_tick(thickness=3, size=34).encode(y="mean(value):Q")
    err = base.mark_errorbar(extent="stdev").encode(y=alt.Y("value:Q", title="score"))
    return (
        alt.layer(points, err, mean, data=scores)
        .properties(width=220, height=260)
        .facet(column=alt.Column("metric:N", title=None))
        .resolve_scale(y="independent")
    )
