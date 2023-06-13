import numpy as np


def numpy_collate(batch):
    # return jax.tree_map(lambda tensor: np.ndarray(tensor), batch)
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return dict((key, numpy_collate([d[key] for d in batch])) for key in batch[0])
    else:
        return np.array(batch)
