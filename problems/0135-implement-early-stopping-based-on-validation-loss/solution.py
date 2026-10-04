from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    cnt = 0
    best = val_losses[0]
    best_epoch = 0

    for i in range(1, len(val_losses)):
        if best - val_losses[i] > min_delta:
            best = val_losses[i]
            best_epoch = i
            cnt = 0
        else:
            cnt += 1

        if cnt >= patience:
            return i, best_epoch 

    return len(val_losses) - 1, best_epoch