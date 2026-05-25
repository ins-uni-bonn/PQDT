from pqdt.utils.miscs import fps_subsample


def compute_pq_reconstruction_losses(loss_fn, pq, upsampled, target):
    if not upsampled:
        raise ValueError("upsampled must contain at least one prediction")

    loss_pq = loss_fn(pq, fps_subsample(target, pq.shape[1]))
    upsampled_losses = [
        loss_fn(
            prediction,
            target if index == len(upsampled) - 1 else fps_subsample(target, prediction.shape[1]),
        )
        for index, prediction in enumerate(upsampled)
    ]
    return {
        "loss": loss_pq + sum(upsampled_losses),
        "loss_pq": loss_pq,
        "loss_fine": upsampled_losses[-1],
    }
