#%%
# print("import packages")
from argparse import ArgumentParser
# print("import packages")
import torch
# print("import packages")
import numpy as np
# print("import packages")
import matplotlib.pyplot as plt

# print("import packages")
from datasets import generate_dataset_1x2, generate_dataset_nxk
# print("import packages")
# from regretnet import RegretNet, train_loop, test_loop
from regretnet_nm_2d import RegretNet_NM_2D, train_loop, test_loop
# print("import packages")
# print("import packages")
from datasets import Dataloader
# print("import packages")
from util import plot_12_model, plot_payment, plot_loss, plot_regret
# print("import packages")
import json
# print("tes")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#%%

from torch.utils.tensorboard import SummaryWriter
#%%
#%%
parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=400)
parser.add_argument('--test-num-examples', type=int, default=100)
parser.add_argument('--n-agents', type=int, default=10)
parser.add_argument('--k-positions', type=int, default=5)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--reserved-price', type=float, default=0)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--test-batch-size', type=int, default=20)
parser.add_argument('--model-lr', type=float, default=1e-2)
parser.add_argument('--misreport-lr', type=float, default=0.1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--rho', type=float, default=50.0)
parser.add_argument('--rho-incr-iter', type=int, default=5)
parser.add_argument('--rho-incr-amount', type=float, default=5.0)
parser.add_argument('--rho-ir', type=float, default=1.0)
parser.add_argument('--rho-incr-iter-ir', type=int, default=5)
parser.add_argument('--rho-incr-amount-ir', type=float, default=5.0)
parser.add_argument('--rho-rp', type=float, default=1.0)
parser.add_argument('--payment_power', type=float, default=0.)
parser.add_argument('--lagr_update_iter', type=int, default=6.0)
parser.add_argument('--lagr_update_iter_ir', type=int, default=6.0)
parser.add_argument('--lagr_update_iter_rp', type=int, default=6.0)
parser.add_argument('--ir_penalty_power', type=float, default=2)
parser.add_argument('--rp_penalty_power', type=float, default=2)
parser.add_argument('--resume', default="")

# architectural arguments
parser.add_argument('--p_activation', default='full_relu_div')
# parser.add_argument('--p_activation', default='full_linear')
parser.add_argument('--a_activation', default='full_linear')
# parser.add_argument('--a_activation', default='full_relu_clipped')
# parser.add_argument('--a_activation', default='full_sigmoid_linear')
parser.add_argument('--hidden_layer_size', type=int, default=100)
parser.add_argument('--n_hidden_layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--rs_loss', action='store_true')

parser.add_argument('--teacher_model', default="")
parser.add_argument('--name', default='testing_name')
#%%
args = parser.parse_args(args=[])
#%%
# if __name__ == "__main__":
print("enter the programm")
# args = parser.parse_args()
print("parser args")
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# writer = SummaryWriter(log_dir=f"run/{args.name}")
writer = None
print("initial writer finish")

K_POSITION = 20
POSITION_RANGE = [0,1]
model = RegretNet_NM_2D(args.n_agents, args.n_items, args.k_positions, activation='relu', hidden_layer_size=args.hidden_layer_size,
                    n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                    a_activation=args.a_activation, separate=args.separate).to(DEVICE)
print("initial model finish")
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

if args.teacher_model != "":
    checkpoint = torch.load(args.teacher_model)
    teachermodel = RegretNet_NM_2D(**checkpoint['arch'])
    teachermodel.load_state_dict(checkpoint['state_dict'], strict=False)
    teachermodel.to(DEVICE)
else:
    teachermodel=None

print("load model parameter finish")
train_data = generate_dataset_nxk(args.n_agents, args.n_items, args.num_examples).to(DEVICE)
train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)
print("generate data finsh")
test_data = generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples).to(DEVICE)
test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

#%%
# test_loop(
#     model,
#     test_loader,
#     args,
#     device=DEVICE
# )
#%%

print("start train loop")
loss_list = train_loop(
    model, train_loader, test_loader, POSITION_RANGE, args, device=DEVICE, writer=writer
)
#%%
import matplotlib.pyplot as plt
plt.plot(loss_list)
# writer.close()
#%%
# plot the result 
from util import plot_12_model, plot_payment, plot_loss, plot_regret
#%%
#%%
pay, alloc, pay_result, result = test_loop(model, test_loader, args, device=DEVICE)
#%%
# print(f"Experiment:{args.name}")
# print(json.dumps(result, indent=4, sort_keys=True))
# plot_payment(model, grid_width=0.01, name=args.name)
# plot_12_model(model, grid_width=0.01, name=args.name)
alloc[0]
#%%
# %%
# save the model
# plot_12_model(model)
# alloc[0].cpu().detach().numpy()
c_dict = ["red", "yellow", "green", "orange", "black"]
def plot_space_distribution(allocs):
    for alloc in allocs:
        alloc_ = alloc.cpu().detach().numpy()
        for i in range(alloc_.shape[1]):
            first_c = alloc_[:,i,:]
            plt.scatter(first_c[:,0], first_c[:,1],c = c_dict[i],vmin=0,vmax=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig("dist_%s.png" % alloc_.shape[1])
    plt.show()

# %%
plot_space_distribution(alloc)
# %%
def plot_one_case(data, alloc, pay, id = 0):
    plt.scatter(data[:,0], data[:,1], c="blue")
    plt.scatter(alloc[0,0], alloc[0,1], s = pay[0] * 500, c="red")
    plt.scatter(alloc[1,0], alloc[1,1], s = pay[1] * 500, c="yellow")
    plt.scatter(alloc[2,0], alloc[2,1], s = pay[2] * 500, c="green")
    plt.savefig("case_%s.png" % id)
    plt.show()

# %%
data = next(test_loader)
data
#%%
allocs, payments = model(data)
allocs
#%%
id_ = 4
plot_one_case(data.cpu().detach().numpy()[id_],
        allocs.cpu().detach().numpy()[id_],
        payments.cpu().detach().numpy()[id_],id_)
# %%
