import torch
import copy


x = torch.tensor([[ 31,  -1,  68,  49,  -1],
        [187,  -1,  -1,  -1,  -1],
        [ 53,   9, 225,  -1,  -1],
        [ 96,  -1,  -1,   2, 142],
        [247,  -1,  -1,  -1,  -1],
        [ 69,   6,   9,  -1,  74],
        [102,  -1,  -1,  -1,  -1],
        [ -1,  -1,  68,  -1,  -1],
        [ -1,  -1,  -1,  82, 164],
        [227,  -1,  46,  -1,  60]])

y = copy.deepcopy(x)
y[y > 0] = 1
y[y < 0] = -9e9

idx_vec = torch.tensor([i for i in range(y.shape[1])])
y += idx_vec
_, indices = torch.topk(y, 1)
vals = torch.gather(x, 1, indices)
print('vals: {}'.format(vals))