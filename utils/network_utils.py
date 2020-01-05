from torch.optim import lr_scheduler

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict

def get_lambda_scheduler(optimizer, args):
    def lambda_rule(i):
        lr_l = 1.0 - max(0, i + 1 - args.niter) / float(args.niter_decay)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def get_step_scheduler(optimizer, args):
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=args.lr_step,
                                    gamma=0.1,
                                    last_epoch=-1)
    return scheduler

def load_state_dict(self, net, path):
    state_dict = remove_module_key(torch.load(path))
    net.load_state_dict(state_dict)
