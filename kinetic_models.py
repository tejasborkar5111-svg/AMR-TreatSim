"""
kinetic_models.py

Core engineering models for predicting ARB/ARG removal across treatment
technologies. Each function returns a LOG10 removal value (the standard
way disinfection/removal performance is reported in wastewater engineering).

Effluent concentration is recovered as:
    effluent = influent * 10 ** (-log_removal)

All formulas here are standard, textbook environmental engineering models
(Chick-Watson, CT concept, pseudo-first-order decay) - not from any single
paper. Replace the default rate constants (k) with values YOUR team
extracts from literature (see literature_dataset.csv) or measures in the
lab (UV calibration experiment).
"""

import math


# ---------------------------------------------------------------------------
# Shared helper: generic first-order log10 removal
# ---------------------------------------------------------------------------
def first_order_log10_removal(k: float, exposure: float) -> float:
    """
    Generic pseudo-first-order removal model.

    Many disinfection/oxidation processes follow:
        C/C0 = exp(-k * exposure)
    which converts to log10 removal as:
        log10(C0/C) = k * exposure / ln(10)

    Args:
        k: rate constant (units depend on process - see each wrapper below)
        exposure: dose, CT value, or contact time (units must match k)

    Returns:
        log10 removal (e.g. 3.0 means 99.9% removal)
    """
    if k < 0 or exposure < 0:
        raise ValueError("k and exposure must be non-negative")
    return (k * exposure) / math.log(10)


# ---------------------------------------------------------------------------
# UV Disinfection - Chick-Watson model
# ---------------------------------------------------------------------------
def uv_log_removal(k: float, uv_dose: float) -> float:
    """
    Chick-Watson model for UV disinfection.

    Args:
        k: UV inactivation rate constant, cm2/mJ
           (default literature range ~0.05-0.25 for ARB; get YOUR own
           value from the Week 3 lab experiment)
        uv_dose: UV dose delivered, mJ/cm2 (= intensity x exposure time)

    Returns:
        log10 removal
    """
    return first_order_log10_removal(k, uv_dose)


# ---------------------------------------------------------------------------
# Ozonation - CT concept (concentration x time)
# ---------------------------------------------------------------------------
def ozone_log_removal(k: float, concentration_mg_l: float, contact_time_min: float) -> float:
    """
    CT-based first-order model for ozone disinfection.

    Args:
        k: ozone rate constant, L/(mg*min)
        concentration_mg_l: residual ozone concentration, mg/L
        contact_time_min: contact time, minutes

    Returns:
        log10 removal
    """
    ct = concentration_mg_l * contact_time_min
    return first_order_log10_removal(k, ct)


# ---------------------------------------------------------------------------
# Advanced Oxidation Processes (AOP) - pseudo-first-order vs contact time
# ---------------------------------------------------------------------------
def aop_log_removal(k: float, contact_time_min: float) -> float:
    """
    Pseudo-first-order hydroxyl-radical degradation model, simplified to
    contact time (since OH-radical exposure is rarely measured directly
    in undergrad-accessible literature - contact time is the common proxy).

    Args:
        k: pseudo-first-order rate constant, min^-1
        contact_time_min: contact/reaction time, minutes

    Returns:
        log10 removal
    """
    return first_order_log10_removal(k, contact_time_min)


# ---------------------------------------------------------------------------
# Membrane Bioreactor (MBR) - empirical (no clean kinetic model exists)
# ---------------------------------------------------------------------------
def mbr_log_removal(reported_log_removal: float) -> float:
    """
    MBR removal is governed by membrane pore size and biofilm rejection,
    not a simple rate law - literature reports removal directly rather
    than a rate constant. This function is a pass-through / lookup wrapper
    so MBR fits the same interface as the other technologies.

    Args:
        reported_log_removal: log10 removal value taken from your
            literature_dataset.csv (average across matching rows for the
            target organism/gene type)

    Returns:
        log10 removal
    """
    return reported_log_removal


# ---------------------------------------------------------------------------
# Hybrid systems - sequential stages, log removals add
# ---------------------------------------------------------------------------
def hybrid_log_removal(stage_log_removals: list) -> float:
    """
    For sequential treatment stages, survival fractions multiply, which
    means log10 removals simply ADD.

    Args:
        stage_log_removals: list of log10 removal values, one per stage
            e.g. [mbr_log_removal(1.5), ozone_log_removal(k, C, t)]

    Returns:
        combined log10 removal across all stages
    """
    return sum(stage_log_removals)


# ---------------------------------------------------------------------------
# Utility: convert log10 removal <-> effluent concentration / percent
# ---------------------------------------------------------------------------
def effluent_concentration(influent: float, log_removal: float) -> float:
    """Given influent concentration and log10 removal, return effluent conc."""
    return influent * (10 ** (-log_removal))


def log_removal_to_percent(log_removal: float) -> float:
    """Convert log10 removal to a percent removal value."""
    return (1 - 10 ** (-log_removal)) * 100


def percent_to_log_removal(percent: float) -> float:
    """Convert percent removal to log10 removal."""
    if not (0 <= percent < 100):
        raise ValueError("percent must be in [0, 100)")
    return -math.log10(1 - percent / 100)


# ---------------------------------------------------------------------------
# Self-test / demo - run this file directly to see sample outputs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== AMR-TreatSim kinetic models: demo run (calibrated from real literature) ===\n")

    influent_conc = 1e6  # example: 1,000,000 CFU/100mL ARB in raw wastewater

    # UV - calibrated from PubMed 24946583: 20 mJ/cm2 dose gave 3.7-log removal
    # of the most UV-resistant ARB class studied (rifampicin-resistant bacteria).
    # Back-calculated k = 3.7 * ln(10) / 20 = 0.426 cm2/mJ
    uv_k = 0.426
    uv_dose = 20  # mJ/cm2
    uv_lr = uv_log_removal(uv_k, uv_dose)
    print(f"UV (source: PubMed 24946583): k={uv_k:.3f} cm2/mJ, dose={uv_dose} mJ/cm2 "
          f"-> log removal={uv_lr:.2f}, percent={log_removal_to_percent(uv_lr):.3f}%")

    # Ozone - calibrated from Antibiotics 2022;11(2):210: mean rate constant
    # ~1.0 min^-1 for O3-based systems on antimicrobial-resistant bacteria,
    # reported reaching >99.9% removal after 10 min contact.
    ozone_k = 1.0  # min^-1 (time-based pseudo-first-order, as reported)
    ozone_contact_time = 10  # min
    ozone_lr = first_order_log10_removal(ozone_k, ozone_contact_time)
    print(f"Ozone (source: Antibiotics 2022;11(2):210): k={ozone_k} min^-1, "
          f"t={ozone_contact_time} min -> log removal={ozone_lr:.2f}, "
          f"percent={log_removal_to_percent(ozone_lr):.3f}%")

    # AOP - directly from Adamou et al. 2025 (ACS EST Water): best mineral-based
    # AOP (rNAu-1) variant achieved 2.9-log removal of ARGs after 30 min.
    aop_lr = 2.9
    print(f"AOP (source: Adamou et al. 2025, ACS EST Water): reported log removal={aop_lr:.2f}, "
          f"percent={log_removal_to_percent(aop_lr):.3f}%")

    # MBR - from full-scale MBR ARG removal study: range 1.5-7.3 log,
    # using the midpoint as a representative value.
    mbr_lr = mbr_log_removal(reported_log_removal=4.4)
    print(f"MBR (source: full-scale MBR ARG study, DOI S0048969718342013): "
          f"log removal={mbr_lr:.2f}, percent={log_removal_to_percent(mbr_lr):.3f}%")

    # Hybrid: MBR + Ozone in sequence (modeled, not from a single hybrid-system paper)
    hybrid_lr = hybrid_log_removal([mbr_lr, ozone_lr])
    print(f"Hybrid (MBR+Ozone, MODELED): combined log removal={hybrid_lr:.2f}, "
          f"percent={log_removal_to_percent(hybrid_lr):.3f}%")

    print("\nAll values above are now traceable to real papers in literature_dataset.csv,")
    print("except Hybrid, which is a modeled combination flagged for validation.")
    print("Next: replace uv_k with your own Week 3 lab-calibrated value once ready.")
