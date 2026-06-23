from math import log2
import random
import csv
from statistics import median


# ============================================================
# Global parameters
# ============================================================

EPSILON = 1e-6
N = 10_000_000
STEP_LIMIT = 10000
RANDOM_SEED = 42

RAW_OUTPUT_FILE = "aas_spine_raw_rows_10M.csv"
GROUP_SUMMARY_FILE = "aas_spine_group_summary_10M.csv"
M_GROUP_SUMMARY_FILE = "aas_spine_m_group_summary_10M.csv"


# ============================================================
# Core Collatz functions
# ============================================================

def v2(n: int) -> int:
    """
    Compute the 2-adic valuation v_2(n).

    That is, return the largest exponent a such that 2^a divides n.
    """
    if n == 0:
        raise ValueError("v2(0) is undefined.")

    n = abs(n)
    return (n & -n).bit_length() - 1


def T(n: int) -> int:
    """
    Accelerated odd Collatz map:

        T(n) = (3n + 1) / 2^{v2(3n+1)}.

    Input n is assumed to be positive and odd.
    """
    x = 3 * n + 1
    return x >> v2(x)


def tau(n: int, limit: int = STEP_LIMIT):
    """
    First-descent time:

        tau(n) = min{r >= 1 : T^r(n) < n}.

    Returns None if no descent is found within the step limit.
    """
    x = n

    for r in range(1, limit + 1):
        x = T(x)

        if x < n:
            return r

    return None


def iterates(n: int, r: int):
    """
    Return the first r accelerated odd iterates:

        [T(n), T^2(n), ..., T^r(n)].
    """
    x = n
    out = []

    for _ in range(r):
        x = T(x)
        out.append(x)

    return out


def kappa(m: int) -> int:
    """
    Exact computation of

        kappa_m = ceil(log2(3^m / (2^m - 1))).

    This uses exact integer comparisons and avoids floating-point error.
    """
    k = 0

    while (1 << k) * ((1 << m) - 1) < 3 ** m:
        k += 1

    return k


# ============================================================
# AAS functions
# ============================================================

def aas_from_deltas(
    deltas,
    epsilon: float = EPSILON,
    redundancy=None,
    weights=None
):
    """
    Compute redundancy-neutral or redundancy-adjusted AAS.

    Given violation magnitudes Delta_i >= 0, define

        x_i = 1 / (1 + Delta_i)

    and

        phi(x_i) = -log2((x_i + epsilon) / (1 + epsilon)).

    Then

        AAS = sum_i omega_i (1 - R_i) phi(x_i).

    In the default setting, redundancy is neutral:

        R_i = 0.

    Equal weights are used unless another weight vector is supplied.
    """
    r = len(deltas)

    if weights is None:
        weights = [1 / r] * r

    if redundancy is None:
        redundancy = [0.0] * r

    total = 0.0

    for delta, omega, R in zip(deltas, weights, redundancy):
        x = 1 / (1 + delta)
        phi = -log2((x + epsilon) / (1 + epsilon))
        total += omega * (1 - R) * phi

    return total


def evaluate_case(m: int, q: int, limit: int = STEP_LIMIT):
    """
    Evaluate one spine case:

        n = 2^m q - 1.

    The function computes:

        - kappa_m
        - certified residue q0
        - certification status
        - s = v2(3^m q - 1)
        - T^m(n)
        - tau(n)
        - delay = tau(n) - m
        - five metamorphic violation magnitudes
        - redundancy-neutral AAS
    """
    km = kappa(m)
    modulus = 1 << km
    q0 = pow(3 ** m, -1, modulus)

    n = (1 << m) * q - 1

    s = v2((3 ** m) * q - 1)

    orbit = iterates(n, m)
    Tm = orbit[m - 1]

    tau_n = tau(n, limit=limit)

    if tau_n is None:
        delay = None
        descent_status = "unresolved"
    elif tau_n == m:
        delay = 0
        descent_status = "exact_at_m"
    elif tau_n > m:
        delay = tau_n - m
        descent_status = "delayed"
    else:
        # This should not occur for spine elements because tau(n) >= m,
        # but it is retained as a defensive check.
        delay = tau_n - m
        descent_status = "early_descent"

    # --------------------------------------------------------
    # MR1: Certified residue proximity
    #
    # Certified condition:
    #
    #     q ≡ q0 mod 2^{kappa_m}
    #
    # Delta_residue is a cyclic normalized distance from q0.
    # --------------------------------------------------------

    residue_distance = (q - q0) % modulus
    cyclic_distance = min(residue_distance, modulus - residue_distance)
    delta_residue = cyclic_distance / (modulus / 2)

    # --------------------------------------------------------
    # MR2: Valuation threshold deficit
    #
    # Expected:
    #
    #     v2(3^m q - 1) >= kappa_m.
    # --------------------------------------------------------

    delta_valuation = max(0, km - s) / km

    # --------------------------------------------------------
    # MR3: Pre-descent resistance
    #
    # Expected:
    #
    #     T^j(n) > n for 1 <= j <= m-1.
    # --------------------------------------------------------

    pre_descent = orbit[:m - 1]

    if all(x > n for x in pre_descent):
        delta_predescent = 0.0
    else:
        delta_predescent = max((n - x) / n for x in pre_descent if x <= n)

    # --------------------------------------------------------
    # MR4: Exact exit descent at time m
    #
    # Expected:
    #
    #     T^m(n) < n.
    # --------------------------------------------------------

    delta_exit = max(0, Tm - n) / n

    # --------------------------------------------------------
    # MR5: Exact first-descent time
    #
    # Expected:
    #
    #     tau(n) = m.
    # --------------------------------------------------------

    if tau_n is None:
        delta_tau = 1.0
    elif tau_n == m:
        delta_tau = 0.0
    else:
        delta_tau = min(1.0, abs(tau_n - m) / m)

    deltas = [
        delta_residue,
        delta_valuation,
        delta_predescent,
        delta_exit,
        delta_tau
    ]

    aas = aas_from_deltas(deltas)

    certified = (q % modulus == q0)

    return {
        "m": m,
        "kappa": km,
        "modulus": modulus,
        "q0": q0,
        "q": q,
        "n": n,
        "certified": certified,
        "s": s,
        "T_m": Tm,
        "tau": tau_n,
        "delay": delay,
        "descent_status": descent_status,
        "Delta_residue": delta_residue,
        "Delta_valuation": delta_valuation,
        "Delta_predescent": delta_predescent,
        "Delta_exit": delta_exit,
        "Delta_tau": delta_tau,
        "AAS": aas
    }


# ============================================================
# Group generation
# ============================================================

def generate_certified_qs(m: int, N: int):
    """
    Generate all certified q-values satisfying

        q = q0 + 2^{kappa_m} t

    and

        n = 2^m q - 1 <= N.
    """
    km = kappa(m)
    modulus = 1 << km
    q0 = pow(3 ** m, -1, modulus)

    q_max = (N + 1) // (1 << m)

    qs = []

    if q0 <= q_max:
        q = q0

        while q <= q_max:
            qs.append(q)
            q += modulus

    return qs


def generate_boundary_qs(m: int, N: int):
    """
    Generate all boundary-shifted control q-values:

        q = q0 + 1 + 2^{kappa_m} t.

    These lie immediately next to the certified residue class but
    are not themselves certified.
    """
    km = kappa(m)
    modulus = 1 << km
    q0 = pow(3 ** m, -1, modulus)

    q_start = q0 + 1
    q_max = (N + 1) // (1 << m)

    qs = []
    q = q_start

    while q <= q_max:
        qs.append(q)
        q += modulus

    return qs


def generate_full_noncertified_qs(m: int, N: int):
    """
    Generate all non-certified q-values satisfying

        1 <= q <= q_max,

    where

        n = 2^m q - 1 <= N,

    and

        q not congruent to q0 mod 2^{kappa_m}.

    This is the full finite-range complement of the certified
    residue class inside the m-th resistance spine.
    """
    km = kappa(m)
    modulus = 1 << km
    q0 = pow(3 ** m, -1, modulus)

    q_max = (N + 1) // (1 << m)

    qs = []

    for q in range(1, q_max + 1):
        if q % modulus != q0:
            qs.append(q)

    return qs


def generate_random_spine_qs(m: int, N: int, sample_per_m: int = 200):
    """
    Generate sampled random spine controls.

    These are random q-values in

        1 <= q <= q_max,

    where

        n = 2^m q - 1 <= N.

    Certified q-values are excluded when possible.
    """
    km = kappa(m)
    modulus = 1 << km
    q0 = pow(3 ** m, -1, modulus)

    q_max = (N + 1) // (1 << m)

    if q_max <= 0:
        return []

    qs = set()
    attempts = 0
    max_attempts = sample_per_m * 50

    while len(qs) < sample_per_m and attempts < max_attempts:
        q = random.randint(1, q_max)

        if q % modulus != q0:
            qs.add(q)

        attempts += 1

    return sorted(qs)


# ============================================================
# Running summary helpers
# ============================================================

def empty_stats():
    """
    Initialize a running statistics dictionary.
    """
    return {
        "cases": 0,

        "resolved_count": 0,
        "unresolved_count": 0,
        "exact_at_m_count": 0,
        "delayed_count": 0,
        "early_descent_count": 0,

        "sum_tau": 0.0,
        "values_tau": [],
        "max_tau": 0,

        "sum_delay": 0.0,
        "values_delay": [],
        "max_delay": 0,

        "sum_AAS": 0.0,
        "values_AAS": [],
        "max_AAS": 0.0,

        "sum_Delta_valuation": 0.0,
        "sum_Delta_exit": 0.0,
        "sum_Delta_tau": 0.0,

        "AAS_positive_count": 0
    }


def update_stats(stats, row):
    """
    Update running statistics with one evaluated row.
    """
    stats["cases"] += 1

    tau_n = row["tau"]
    delay = row["delay"]
    descent_status = row["descent_status"]

    if tau_n is None:
        stats["unresolved_count"] += 1
    else:
        stats["resolved_count"] += 1
        stats["sum_tau"] += tau_n
        stats["values_tau"].append(tau_n)
        stats["max_tau"] = max(stats["max_tau"], tau_n)

        if delay is not None:
            stats["sum_delay"] += delay
            stats["values_delay"].append(delay)
            stats["max_delay"] = max(stats["max_delay"], delay)

    if descent_status == "exact_at_m":
        stats["exact_at_m_count"] += 1
    elif descent_status == "delayed":
        stats["delayed_count"] += 1
    elif descent_status == "early_descent":
        stats["early_descent_count"] += 1

    stats["sum_AAS"] += row["AAS"]
    stats["values_AAS"].append(row["AAS"])
    stats["max_AAS"] = max(stats["max_AAS"], row["AAS"])

    stats["sum_Delta_valuation"] += row["Delta_valuation"]
    stats["sum_Delta_exit"] += row["Delta_exit"]
    stats["sum_Delta_tau"] += row["Delta_tau"]

    if row["AAS"] > 1e-12:
        stats["AAS_positive_count"] += 1


def finalize_stats(group, stats):
    """
    Convert running statistics into a final summary row.
    """
    cases = stats["cases"]
    resolved = stats["resolved_count"]

    if cases == 0:
        return {
            "group": group,
            "cases": 0,

            "resolved_count": 0,
            "unresolved_count": 0,
            "exact_at_m_count": 0,
            "delayed_count": 0,
            "early_descent_count": 0,

            "mean_tau": None,
            "median_tau": None,
            "max_tau": None,

            "mean_delay": None,
            "median_delay": None,
            "max_delay": None,

            "mean_AAS": None,
            "median_AAS": None,
            "max_AAS": None,

            "mean_Delta_valuation": None,
            "mean_Delta_exit": None,
            "mean_Delta_tau": None,

            "AAS_positive_count": 0
        }

    return {
        "group": group,
        "cases": cases,

        "resolved_count": stats["resolved_count"],
        "unresolved_count": stats["unresolved_count"],
        "exact_at_m_count": stats["exact_at_m_count"],
        "delayed_count": stats["delayed_count"],
        "early_descent_count": stats["early_descent_count"],

        "mean_tau": stats["sum_tau"] / resolved if resolved else None,
        "median_tau": median(stats["values_tau"]) if stats["values_tau"] else None,
        "max_tau": stats["max_tau"] if stats["values_tau"] else None,

        "mean_delay": stats["sum_delay"] / resolved if resolved else None,
        "median_delay": median(stats["values_delay"]) if stats["values_delay"] else None,
        "max_delay": stats["max_delay"] if stats["values_delay"] else None,

        "mean_AAS": stats["sum_AAS"] / cases,
        "median_AAS": median(stats["values_AAS"]),
        "max_AAS": stats["max_AAS"],

        "mean_Delta_valuation": stats["sum_Delta_valuation"] / cases,
        "mean_Delta_exit": stats["sum_Delta_exit"] / cases,
        "mean_Delta_tau": stats["sum_Delta_tau"] / cases,

        "AAS_positive_count": stats["AAS_positive_count"]
    }


def write_csv(filename, rows):
    """
    Write a list of dictionaries to CSV.
    """
    if not rows:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Full AAS experiment
# ============================================================

def run_full_aas_experiment(
    N: int = 10_000_000,
    random_per_m: int = 200,
    write_raw_rows: bool = True,
    include_full_noncertified: bool = True
):
    """
    Full AAS experiment.

    Original Paper 2 groups are preserved:

    1. certified

       q ≡ q0 mod 2^{kappa_m}

    2. boundary_shifted

       q ≡ q0 + 1 mod 2^{kappa_m}

    3. random_spine

       sampled random non-certified q-values in the spine.

    Paper 3 extension:

    4. full_noncertified

       all q-values satisfying

           q not congruent to q0 mod 2^{kappa_m}

       in the finite range n = 2^m q - 1 <= N.

    Output files:

        aas_spine_raw_rows_10M.csv
        aas_spine_group_summary_10M.csv
        aas_spine_m_group_summary_10M.csv
    """
    random.seed(RANDOM_SEED)

    all_rows = []

    group_stats = {
        "certified": empty_stats(),
        "boundary_shifted": empty_stats(),
        "random_spine": empty_stats()
    }

    if include_full_noncertified:
        group_stats["full_noncertified"] = empty_stats()

    m_group_stats = {}

    m = 2

    while (1 << m) - 1 <= N:
        print(f"\nStarting m={m}")

        qs_certified = generate_certified_qs(m, N)
        qs_boundary = generate_boundary_qs(m, N)
        qs_random = generate_random_spine_qs(
            m,
            N,
            sample_per_m=random_per_m
        )

        groups = {
            "certified": qs_certified,
            "boundary_shifted": qs_boundary,
            "random_spine": qs_random
        }

        if include_full_noncertified:
            qs_full_noncertified = generate_full_noncertified_qs(m, N)
            groups["full_noncertified"] = qs_full_noncertified

        for group_name, qs in groups.items():
            print(f"  group={group_name}, cases={len(qs)}")

            key = (m, group_name)

            if key not in m_group_stats:
                m_group_stats[key] = empty_stats()

            for idx, q in enumerate(qs, start=1):
                row = evaluate_case(m, q)
                row["group"] = group_name

                update_stats(group_stats[group_name], row)
                update_stats(m_group_stats[key], row)

                if write_raw_rows:
                    all_rows.append(row)

                if idx % 100000 == 0:
                    print(f"    processed {idx}/{len(qs)}")

        print(f"Finished m={m}")

        m += 1

    group_summary = [
        finalize_stats(group, stats)
        for group, stats in group_stats.items()
    ]

    m_group_summary = []

    for (m, group), stats in sorted(m_group_stats.items()):
        row = finalize_stats(group, stats)
        row["m"] = m
        m_group_summary.append(row)

    if write_raw_rows:
        write_csv(RAW_OUTPUT_FILE, all_rows)

    write_csv(GROUP_SUMMARY_FILE, group_summary)
    write_csv(M_GROUP_SUMMARY_FILE, m_group_summary)

    print("\n=== GROUP SUMMARY FULL ===")

    for row in group_summary:
        print(row)

    print("\nCSV files written:")

    if write_raw_rows:
        print(RAW_OUTPUT_FILE)

    print(GROUP_SUMMARY_FILE)
    print(M_GROUP_SUMMARY_FILE)

    return group_summary, m_group_summary


# ============================================================
# Run experiment
# ============================================================

group_summary_full, m_group_summary_full = run_full_aas_experiment(
    N=N,
    random_per_m=200,
    write_raw_rows=True,
    include_full_noncertified=True
)
