import numpy as np


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=300,
                      lr_decay_iter=1, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    if (iter % lr_decay_iter != 0) or (iter > max_iter):
     	return optimizer.param_groups[0]['lr']

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    
    return lr


def poly_lr_scheduler_with_backbone(optimizer, init_lr, iter, max_iter=50, lr_decay_iter=1, power=0.9):
    '''
    polynomial learning rate scheduler with support for different scaling factors across parameter groups 
    '''
  
    if (iter % lr_decay_iter != 0) or (iter > max_iter):
        return [group['lr'] for group in optimizer.param_groups]

    new_base_lr = init_lr * (1 - iter / max_iter) ** power

    for group in optimizer.param_groups:
        scale = group['initial_lr'] / init_lr
        group['lr'] = new_base_lr * scale

    return [group['lr'] for group in optimizer.param_groups]


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)
        

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def mean_iou(per_class_iou):
    valid_ious = per_class_iou[~np.isnan(per_class_iou)]
    valid_class = len(valid_ious)
    return np.sum(valid_ious)/valid_class
