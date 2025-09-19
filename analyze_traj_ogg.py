def analyze_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""

    # Featurize trajectories.
    results = {}
    results["featurization"] = {
        "traj": featurize_trajectory(traj_md),
        "ref_traj": featurize_trajectory(ref_traj_md),
    }
    py_logger.info(f"Featurization complete.")

    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"]
    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_feats = ref_traj_results["feats"]["torsions"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]

    assert traj_feats.describe() == ref_traj_feats.describe(), "Featurization of trajectories does not match."
    feats = traj_feats

    # Compute feature histograms.
    results["feature_histograms"] = {
        "traj": compute_feature_histograms(traj_featurized_dict),
        "ref_traj": compute_feature_histograms(ref_traj_featurized_dict),
    }
    py_logger.info(f"Feature histograms computed.")

    # We will compare the trajectory as well as the (shortened) reference trajectories.
    trajs_to_compare = {
        "traj": traj_featurized,
        "ref_traj": ref_traj_featurized,
        "ref_traj_10x": ref_traj_featurized[: len(ref_traj_featurized) // 10],
        "ref_traj_100x": ref_traj_featurized[: len(ref_traj_featurized) // 100],
        "ref_traj_1000x": ref_traj_featurized[: len(ref_traj_featurized) // 1000],
    }

    # Compute PMFs.
    results["PMFs"] = {}
    for key, traj in trajs_to_compare.items():
        results["PMFs"][key] = compute_dihedral_PMFs(traj, feats)
    py_logger.info(f"PMFs computed.")

    # Compute JSDs.
    results["JSD_torsions"] = {}
    for key, traj in trajs_to_compare.items():
        results["JSD_torsions"][key] = compute_JSD_torsions(
            traj,
            ref_traj_featurized,
            feats,
        )
    py_logger.info(f"JSD of torsion distributions computed.")

    # Compute JSDs of torsions against time.
    results["JSD_torsions_against_time"] = {}
    for key, traj in trajs_to_compare.items():
        results["JSD_torsions_against_time"][key] = compute_JSD_torsions_against_time(
            traj,
            ref_traj_featurized,
            feats,
        )
    py_logger.info(f"JSD of torsion distributions as a function of time computed.")

    # Compute torsion decorrelations.
    results["torsion_decorrelations"] = compute_torsion_decorrelations(
        traj_featurized,
        ref_traj_featurized,
        feats,
    )
    py_logger.info(f"Torsion decorrelations computed.")

    # TICA analysis.
    results["TICA"] = compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info(f"TICA computed.")

    traj_tica = results["TICA"]["traj"]
    ref_traj_tica = results["TICA"]["ref_traj"]

    traj_ticas_to_compare = {
        "traj": traj_tica,
        "ref_traj": ref_traj_tica,
        "ref_traj_10x": ref_traj_tica[: len(ref_traj_tica) // 10],
        "ref_traj_100x": ref_traj_tica[: len(ref_traj_tica) // 100],
        "ref_traj_1000x": ref_traj_tica[: len(ref_traj_tica) // 1000],
    }

    # Compute TICA stats.
    results["TICA_histograms"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["TICA_histograms"][key] = compute_TICA_histogram_for_plotting(tica)
    py_logger.info(f"Histograms of TICA projections computed.")

    results["JSD_TICA"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_TICA"][key] = compute_JSD_TICA(
            tica,
            ref_traj_tica,
        )
    py_logger.info(f"JSD of TICA projections computed.")

    results["JSD_TICA_against_time"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_TICA_against_time"][key] = compute_JSD_TICA_against_time(
            tica,
            ref_traj_tica,
        )
    py_logger.info(f"JSD of TICA projections as a function of time computed.")

    # Compute autocorrelation stats.
    results["TICA_decorrelations"] = compute_TICA_decorrelations(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"TICA decorrelations computed.")

    # Compute MSM.
    # Sometimes, this fails because the reference trajectory is too short.
    try:
        MSM_info = get_MSM_after_KMeans(ref_traj_tica)
        results["MSM"] = MSM_info
    except IndexError:
        py_logger.warning(f"MSM information could not be computed.")
        return results

    results["JSD_MSM"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_MSM"][key] = compute_JSD_MSM(
            tica,
            ref_traj_tica,
            MSM_info,
        )
    py_logger.info(f"JSD of MSM state probabilities computed.")

    results["JSD_MSM_against_time"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_MSM_against_time"][key] = compute_JSD_MSM_against_time(
            tica,
            ref_traj_tica,
            MSM_info,
        )
    py_logger.info(f"JSD of MSM state probabilities as a function of time computed.")

    results["MSM_matrices"] = {}
    for key, tica in traj_ticas_to_compare.items():
        try:
            results["MSM_matrices"][key] = compute_MSM_transition_and_flux_matrices(
                tica,
                MSM_info,
            )
        except (ValueError, RuntimeError):
            py_logger.warning(f"MSM matrices could not be computed for {key}.")
            continue
    py_logger.info(f"MSM matrices computed.")

    return results
