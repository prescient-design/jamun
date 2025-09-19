def analyze_trajectories_sdf(traj_md: md.Trajectory, sdf_file: str) -> Dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""

    # Featurize trajectories.
    results = {}
    results["featurization"] = analysis_utils_sdf.featurize_trajectories_macrocycle(traj_md, sdf_file)

    py_logger.info(f"Analysis_utils_sdf Featurization complete.")
    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"] 

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    py_logger.info(f"Featurization complete.")

    py_logger.info(f"this is the traj results: {traj_results}")
    py_logger.info(f"this is the ref traj results: {ref_traj_results}")
    py_logger.info(f"this is the traj feats: {traj_feats}")
    py_logger.info(f"this is the ref traj feats: {ref_traj_results['feats']}")
    py_logger.info(f"this is the traj featurized dict: {traj_featurized_dict}")
    py_logger.info(f"this is the ref traj featurized dict: {ref_traj_featurized_dict}")

    py_logger.info(f"this is the traj featurized: {traj_featurized}") 
    py_logger.info(f"this is the ref traj featurized: {ref_traj_featurized}")

    # Compute feature histograms.
    results["feature_histograms"] = compute_feature_histograms(
        traj_featurized_dict,
        ref_traj_featurized_dict,
    )
    py_logger.info(f"Feature histograms computed.")

    # Compute PMFs.
    results["PMFs"] = analysis_utils_sdf.compute_PMFs_mc(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"PMFs computed.")

    # Compute JSDs.
    results["JSD_torsion_stats"] = analysis_utils_sdf.compute_JSD_torsion_stats_mc(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD torsion stats computed.")

    # Compute JSDs of torsions against time.
    results["JSD_torsion_stats_against_time"] = analysis_utils_sdf.compute_JSD_torsion_stats_against_time(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD torsion stats as a function of time computed.")

    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]

    # TICA analysis.
    results["TICA"] = analysis_utils.compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info(f"TICA computed.")

    traj_tica = results["TICA"]["traj_tica"]
    ref_traj_tica = results["TICA"]["ref_traj_tica"]

    # Compute TICA stats.
    results["TICA_stats"] = analysis_utils.compute_TICA_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"TICA stats computed.")

    # Compute autocorrelation stats.
    results["autocorrelation_stats"] = analysis_utils.compute_autocorrelation_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"Autocorrelation stats computed.")

    # Compute MSM stats.
    # Sometimes, this fails because the reference trajectory is too short.
    try:
        results["MSM_stats"] = compute_MSM_stats(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info(f"MSM stats computed.")

        # Compute JSDs against time.
        results["JSD_MSM_stats_against_time"] = analysis_utils.compute_JSD_MSM_stats_against_time(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info(f"JSD MSM stats as a function of time computed.")
   
    except IndexError:
        py_logger.warning(f"MSM stats could not be computed.")

    return results
