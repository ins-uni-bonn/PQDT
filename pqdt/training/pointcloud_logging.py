def log_reconstruction_meshes(logger, step, stage, target, source, rebuild, coarse, up_factors=None):
    if logger is None:
        return

    from pqdt.utils.miscs import cat_vertices_color

    experiment = logger.experiment
    vis, color = cat_vertices_color(target.permute(0, 2, 1), rebuild.permute(0, 2, 1), n_upsample=up_factors)
    experiment.add_mesh(f"{stage}/recon", vertices=vis, colors=color, global_step=step)

    vis, color = cat_vertices_color(target.permute(0, 2, 1), coarse.permute(0, 2, 1))
    experiment.add_mesh(f"{stage}/coarse_recon", vertices=vis, colors=color, global_step=step)

    vis, color = cat_vertices_color(target.permute(0, 2, 1), source.permute(0, 2, 1))
    experiment.add_mesh(f"{stage}/input", vertices=vis, colors=color, global_step=step)
