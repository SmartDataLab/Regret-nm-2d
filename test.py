#%%
from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import generate_dataset_1x2, generate_dataset_nxk
from regretnet import RegretNet, train_loop, test_loop
# from torch.utils.tensorboard import SummaryWriter
from datasets import Dataloader
from util import plot_12_model, plot_payment, plot_loss, plot_regret
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=6000)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--misreport-lr', type=float, default=2e-2)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--reserved-price', type=float, default=0)
parser.add_argument('--p_activation', default="")
parser.add_argument('--a_activation', default="")
parser.add_argument('--n-agents', default="")
parser.add_argument('--model', default="")

#%%
# if __name__ == "__main__":
# args = parser.parse_args()
# print(args.random_seed)
#Sample testing scripts
# python test.py --p_activation frac_sigmoid --a_activation softmax \
# --model model/2x2_softmax_in_scratch.pt --n-agents 2
# python test.py --p_activation frac_sigmoid_linear --a_activation sparsemax \
# --model model/2x2_sparsemax_in_linsigpmt_scratch_fast.pt --n-agents 2
class Arg:
    def __init__(self):
        self.random_seed=0
        self.test_num_examples = 6000
        self.batch_size = 256
        self.test_batch_size = 256
        self.misreport_lr = 2e-2
        self.misreport_iter = 25
        self.test_misreport_iter = 1000
        self.reserved_price = 0
        self.model = "model/2x2_sparsemax_in_linsigpmt_scratch_fast.pt"
        self.a_activation = "sparsemax"
        self.p_activation = "frac_sigmoid_linear"
        self.n_agents =2
args = Arg()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

checkpoint = torch.load(args.model, map_location='cpu')
print("Architecture:")
print(json.dumps(checkpoint['arch'], indent=4, sort_keys=True))
print("Training Args:")
print(json.dumps(vars(checkpoint['args']), indent=4, sort_keys=True))
#%%

# override p_activation
if args.p_activation != "":
    checkpoint['arch']['p_activation'] = args.p_activation


model = RegretNet(**checkpoint['arch']).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'], strict=False)
test_data = generate_dataset_nxk(checkpoint['arch']['n_agents'],
                                    checkpoint['arch']['n_items'], args.test_num_examples).to(DEVICE)
args.n_agents = checkpoint['arch']['n_agents']
args.n_items = checkpoint['arch']['n_items']
test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

pay, alloc, pay_result, result = test_loop(model, test_loader, args, device=DEVICE)
print(f"Experiment:{checkpoint['name']}")
print(json.dumps(result, indent=4, sort_keys=True))
print(pay_result)
#%%

# visualize additive utility 1x2 uniform distribution
x1 = (2.0 - np.sqrt(2.0)) / 3.0
x2 = 2.0 / 3.0
points = [(x1, 1.0), (x1, x2), (x2, x1), (x2, 0)]
x = list(map(lambda x: x[0], points))
y = list(map(lambda x: x[1], points))

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
plt.plot(x, y, linewidth=2, linestyle='--', c='black')

img = ax.imshow(alloc[23].cpu().detach().numpy()[::1, 0, :], extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, cmap='YlOrRd')

plt.text(0.25, 0.25, s='0', color='black', fontsize='10', fontweight='bold')
plt.text(0.65, 0.65, s='1', color='black', fontsize='10', fontweight='bold')

ax.set_xlabel('$v_1$')
ax.set_ylabel('$v_2$')
plt.title('01 Prob. of allocating item 1')
_ = plt.colorbar(img, fraction=0.046, pad=0.04)
fig.set_size_inches(4, 3)
plt.savefig('alloc1.png', bbox_inches='tight', pad_inches=0.05)



# %%
