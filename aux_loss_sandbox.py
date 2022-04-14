import torch
import copy


def find_most_probably_span_len(poss_start_idxs, poss_end_idxs, min_start_idx=None):
    poss_start_idxs = torch.cat([poss_start_idxs.unsqueeze(-1) for _ in range(poss_start_idxs.shape[-1])], dim=-1)
    poss_end_idxs = poss_end_idxs.unsqueeze(1)

    if min_start_idx is not None:
        # print('poss_start_idxs:\n{}'.format(poss_start_idxs))
        min_start_idx = torch.repeat_interleave(
            torch.repeat_interleave(
                min_start_idx, poss_start_idxs.shape[-1], dim=1
            ).unsqueeze(-1), poss_start_idxs.shape[-1], dim=-1
        )
        # print('minimum_length:\n{}'.format(min_start_idx))

        poss_start_idxs[poss_start_idxs <= min_start_idx] = 9e9
        # input('poss_start_idxs:\n{}'.format(poss_start_idxs))

    z = poss_end_idxs - poss_start_idxs
    z_orig = copy.deepcopy(z)

    z[z < 0] = -9e9
    z[z >= 0] = 1
    # print('z:\n{}'.format(z))
    adj_z = torch.tensor([i for i in range(poss_start_idxs.shape[-1])])
    adj_z = [adj_z + i for i in range(poss_start_idxs.shape[-1])]

    adj_z = torch.cat([s.unsqueeze(1) for s in adj_z], dim=1)
    # print('adj_z:\n{}'.format(adj_z))

    new_z = z + adj_z
    # print('new_z:\n{}'.format(new_z))

    new_z = new_z.view(new_z.shape[0], -1)
    max_idxs = torch.argmax(new_z, dim=-1)
    # print('max_idxs:\n{}'.format(max_idxs))
    z_orig = z_orig.view(z_orig.shape[0], -1)

    # print('z_orig.shape: {}'.format(z_orig.shape))
    # print('max_idxs.shape: {}'.format(max_idxs.shape))

    best_lengths = torch.gather(z_orig, 1, max_idxs.unsqueeze(-1))
    # print('best_lengths:\n{}'.format(best_lengths))
    return best_lengths


x = torch.tensor([[134, 145, 74, 222, 266],
                  [142, 106, 234, 41, 154],
                  [127, 59, 153, 170, 174],
                  [157, 87, 147, 233, 237],
                  [128, 130, 20, 155, 150],
                  [153, 135, 60, 237, 143],
                  [144, 77, 45, 71, 225],
                  [139, 162, 153, 142, 71],
                  [122, 236, 131, 143, 233],
                  [136, 234, 171, 175, 170]])

y = torch.tensor([[67, 136, 96, 115, 128],
                  [134, 70, 118, 99, 98],
                  [160, 144, 68, 28, 110],
                  [93, 262, 72, 200, 114],
                  [167, 168, 100, 130, 94],
                  [42, 256, 68, 44, 89],
                  [113, 118, 130, 112, 98],
                  [164, 128, 112, 107, 125],
                  [129, 157, 93, 5, 89],
                  [231, 90, 111, 223, 69]])

z = torch.tensor([[13],
                  [16],
                  [12],
                  [16],
                  [12],
                  [12],
                  [16],
                  [10],
                  [12],
                  [13]])

# print('x.shape: {}'.format(x.shape))
# print('y.shape: {}'.format(y.shape))

something_something = find_most_probably_span_len(x, y, min_start_idx=z)
print(something_something)
