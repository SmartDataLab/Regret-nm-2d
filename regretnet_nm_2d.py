import json
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as tqdm

import ibp as ibp
from datasets import generate_dataset_1x2, generate_dataset_nxk
from util import plot_12_model, plot_loss, plot_payment, plot_regret

K = 20
class RegretNet_NM_2D(nn.Module):
    def __init__(self, n_agents, loc_n_dim, k_locations = K, hidden_layer_size=128, clamp_op=None, n_hidden_layers=2,
                 activation='tanh', p_activation=None, a_activation='softmax', separate=False):
        super(RegretNet_NM_2D, self).__init__()

        # 竞标者总效用等于单个竞标者效用的和
        self.activation = activation
        if activation == 'tanh':
            self.act = ibp.Tanh
        else:
            self.act = ibp.ReLU
        self.clamp_opt = clamp_op
        if clamp_op is None:
            def cl(x):
                x.clamp_min_(0.0)
                x.clamp_max_(1.0)
            self.clamp_op = cl
        else:
            self.clamp_op = clamp_op

        self.p_activation = p_activation
        self.a_activation = a_activation
        self.n_agents = n_agents
        self.loc_n_dim = loc_n_dim
        self.k_locations = k_locations

        self.input_size = self.n_agents * self.loc_n_dim
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers=n_hidden_layers
        self.separate = separate
        # 输出为获得每个物品的竞拍人及每个竞拍人付出的支付价格

        self.payments_size = self.k_locations
        # 20
        self.allocations_size = self.k_locations * self.loc_n_dim

        if self.a_activation == 'softmax':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim)),
                                                  ibp.Softmax(dim=1),
                                                  ibp.View_Cut()]
        elif self.a_activation == 'sparsemax':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim)),
                                                  ibp.Sparsemax(dim=1),
                                                  ibp.View_Cut()]
        elif self.a_activation == 'full_sigmoid_linear':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.SigmoidLinear(),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim))]
        elif self.a_activation == 'full_relu_clipped':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.ReLUClipped(lower=0, upper=1),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim))]
        elif self.a_activation == 'full_relu_div':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.ReLUClipped(lower=0, upper=1),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim)),
                                                  ibp.Allo_Div(dim=1)]
        elif self.a_activation == 'full_linear':
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                                  ibp.View((-1, self.k_locations, self.loc_n_dim))]
        else:
            raise ValueError(f"{self.a_activation} behavior is not defined")

        if p_activation == 'full_sigmoid':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.hidden_layer_size),
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.Sigmoid()
            ]
        elif p_activation == 'full_sigmoid_linear':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.SigmoidLinear(mult=loc_n_dim)
            ]
        elif p_activation == 'full_relu':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.ReLU()
            ]
        elif p_activation == 'full_relu_clipped':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.ReLUClipped(lower=0, upper=loc_n_dim)
            ]
        elif p_activation == 'frac_relu_clipped':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.ReLUClipped(lower=0, upper=1)
            ]
        elif p_activation == 'frac_sigmoid_linear':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.SigmoidLinear()
            ]
        elif p_activation == 'full_linear':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size)
            ]
        elif p_activation == 'frac_sigmoid':
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.Sigmoid()
            ]
        elif p_activation == 'full_relu_div':
            self.payment_head = [ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.ReLUClipped(lower=0.1, upper=0.9),ibp.Allo_Div(dim=1)]
        else:
            raise ValueError('payment activation behavior is not defined')

        if separate:
            self.nn_model = ibp.Sequential(
                *([ibp.Identity()])
            )
            self.payment_head = [ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                [l for i in range(n_hidden_layers)
                                 for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                self.payment_head

            self.payment_head = ibp.Sequential(*self.payment_head)
            self.allocation_head = [ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                   [l for i in range(n_hidden_layers)
                                    for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                   self.allocation_head
            self.allocation_head = ibp.Sequential(*self.allocation_head)
        else:
            self.nn_model = ibp.Sequential(
                *([ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] +
                  [l for i in range(n_hidden_layers)
                   for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
            )
            self.allocation_head = ibp.Sequential(*self.allocation_head)
            self.payment_head = ibp.Sequential(*self.payment_head)

    def glorot_init(self):
        # reinitializes with Glorot (aka Xavier) uniform initialization

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):

        # tensor x 的形状为 [batch_size, n_agents, n_items]
        # 转化为 [batch_size, n_agents * n_items]
        # output的形状为 [batch_size, n_agents, n_items],
        # 对每个拍卖品的分配需要经过 softmax 或 doubly stochastic

        x = reports.view(-1, self.n_agents * self.loc_n_dim)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        if self.p_activation in ['frac_sigmoid', 'frac_relu_clipped']:
            payments = self.payment_head(x) * torch.sum(
                allocs * reports, dim=2
            )
        elif self.p_activation == 'frac_sigmoid_linear':
            payments = self.payment_head(x) * torch.sum(
                allocs * reports, dim=2
            )
        else:
            payments = self.payment_head(x)

        return allocs, payments

    def interval(self, reports_upper, reports_lower):
        upper = reports_upper.view(-1, self.n_agents * self.loc_n_dim)
        lower = reports_lower.view(-1, self.n_agents * self.loc_n_dim)
        upper, lower = self.nn_model.interval(upper, lower)

        allocs_upper, allocs_lower = self.allocation_head.interval(upper, lower)

        if self.p_activation == 'frac_sigmoid':
            raise ValueError("Interval for frac_sigmoid is currently not implemented")
            payments_upper, payments_lower = frac_upper * torch.sum(allocs_upper[:, :-1, :] * reports_upper, dim=2), \
                                frac_lower * torch.sum(allocs_lower[:, :-1, :] * reports_lower, dim=2)
        else:
            payments_upper, payments_lower = self.payment_head.interval(upper, lower)
        return (allocs_upper, allocs_lower), (payments_upper, payments_lower)
    def reg(self, reports_upper, reports_lower):
        reg = 0
        upper = reports_upper.view(-1, self.n_agents * self.loc_n_dim)
        lower = reports_lower.view(-1, self.n_agents * self.loc_n_dim)
        reg += self.nn_model.reg(upper, lower)
        upper, lower = self.nn_model.interval(upper, lower)
        reg += self.allocation_head.reg(upper, lower)
        reg += self.payment_head.reg(upper, lower)
        return reg





def calc_agent_util(valuations, agent_allocations, payments):
    # valuations size [-1, n_agents, loc_n_dim]
    # agent_allocations size [-1, k_postion, loc_n_dim]
    # payments size [-1, n_agents]
    # TODO(sujinhua): to change teh payment as the investment
    # 计算allocs时需要将dummy dim去掉
    # valuations is agent positon
    # use manhatta distance to calculate the position
    # calculate the 
    # 原始
    # util_from_items = torch.sum(agent_allocations * valuations, dim=2)
    # report location x alloca
    agent_postion_c_dist = torch.cdist(valuations, agent_allocations)
    util_from_items,_ =  torch.min(agent_postion_c_dist, dim=2)
    Max_dist = 1.5
    # util = f(dist) f = C - x
    util_from_items = Max_dist - util_from_items

    return util_from_items# - payments

def calc_agent_util_bound(valuations, agent_allocations_upper, payments_lower):
    # valuations size [-1, n_agents, n_items]
    # agent_allocations_bounds size [-1*n_agents, n_agents, n_items]
    # payments size [-1*n_agents, n_agents]
    n_agents = valuations.shape[1]
    n_items = valuations.shape[2]

    util_from_items_upper = torch.sum(agent_allocations_upper.reshape(-1, n_agents, n_agents, n_items) * valuations[:, None, :, :], dim=3)
    util_upper = util_from_items_upper - payments_lower.reshape(-1, n_agents, n_agents)
    util_upper = util_upper[:, range(n_agents), range(n_agents)]
    return util_upper


def create_combined_misreports(misreports, valuations):
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2]

    # 构建mask计算所有情况的和，从而将竞标者的效用情况结合起来
    mask = torch.zeros(
        (misreports.shape[0], n_agents, n_agents, n_items), device=misreports.device
    )
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0

    tiled_mis = misreports.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)
    tiled_true = valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)

    return mask * tiled_mis + (1.0 - mask) * tiled_true

def create_misreports_bound(valuations, eps=.1):
    n_agents = valuations.shape[1]
    n_items = valuations.shape[2]

    # 构建mask计算所有情况的和，从而将竞标者的出价情况结合起来
    mask = torch.zeros(
        (valuations.shape[0], n_agents, n_agents, n_items), device=valuations.device
    )
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0
    tiled_upper = torch.min(valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)+eps, torch.tensor(1.).to(valuations.device))
    tiled_lower = torch.max(valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)-eps, torch.tensor(0.).to(valuations.device))
    tiled_true = valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)
    tiled_upper, tiled_lower = mask * tiled_upper + (1.0 - mask) * tiled_true, \
                  mask * tiled_lower + (1.0 - mask) * tiled_true

    return tiled_upper, tiled_lower


def optimize_misreports(
    model, current_valuations, current_misreports, misreport_iter=10, lr=1e-1
):
    # misreports 和 valuations 的tensor大小相同，初始结果也相同

    current_misreports.requires_grad_(True)

    for i in range(misreport_iter):
        model.zero_grad()
        agent_utils = tiled_misreport_util(current_misreports, current_valuations, model)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), current_misreports)

        with torch.no_grad():
            current_misreports += lr * misreports_grad
            model.clamp_op(current_misreports)

    return current_misreports


def tiled_misreport_util_bound(current_valuations, model, eps=0.1):
    n_agents = current_valuations.shape[1]
    n_items = current_valuations.shape[2]
    agent_idx = list(range(n_agents))

    batch_upper, batch_lower = create_misreports_bound(current_valuations, eps=eps)
    flatbatch_tiled_misreports_upper, flatbatch_tiled_misreports_lower = batch_upper.view(-1, n_agents, n_items), batch_lower.view(-1, n_agents, n_items)


    (allocs_upper, allocs_lower), (payments_upper, payments_lower) = model.interval(flatbatch_tiled_misreports_upper, flatbatch_tiled_misreports_lower)

    payments_upper, payments_lower = payments_upper.view(-1, n_agents, n_agents), payments_lower.view(-1, n_agents, n_agents)

    allocations_upper, allocations_lower = allocs_upper.view(-1, n_agents, n_agents, n_items), allocs_lower.view(-1, n_agents, n_agents, n_items)

    misreport_util = calc_agent_util_bound(
        current_valuations, allocations_upper, payments_lower
    )

    return misreport_util


def tiled_misreport_util(current_misreports, current_valuations, model):
    n_agents = current_valuations.shape[1]
    n_items = current_valuations.shape[2]
    # loc_n_dim

    agent_idx = list(range(n_agents))
    tiled_misreports = create_combined_misreports(
        current_misreports, current_valuations
    )
    flatbatch_tiled_misreports = tiled_misreports.view(-1, n_agents, n_items)
    allocations, payments = model(flatbatch_tiled_misreports)
    # reshaped_payments = payments.view(
    #     -1, n_agents, n_agents
    # )
    reshaped_allocations = allocations.view(-1, n_agents, allocations.shape[1], n_items)
    # 将agent's payments 和 allocations分出来
    # agent_payments = reshaped_payments[:, agent_idx, agent_idx]
    agent_allocations = torch.mean(reshaped_allocations,dim= 1)
    agent_utils = calc_agent_util(
        current_valuations, agent_allocations, payments
    )
    # agent_utils size [-1, n_agents]
    return agent_utils

def calc_rp_loss(model, payments, rp_limit, rp_lagr_mults, rho):
    # 构建mask计算所有情况下的求和
    mask = torch.ones(
        (1, model.n_agents), device = payments.device)
    edge = torch.zeros(
        (1, model.n_agents), device = payments.device)
    ReLU_layer = torch.nn.ReLU()
    # max_rp_operator = ReLU_layer(rp_lagr_mults + rho * (rp_limit - payments))
    max_rp_operator = torch.max(edge, rp_lagr_mults + rho * (rp_limit - payments))
    rp_decomposed = max_rp_operator**2 - rp_lagr_mults**2
    rp_loss = rp_decomposed.sum(dim=1).mean()
    return rp_loss

def test_loop(
    model,
    loader,
    args,
    device='cpu'
):
    total_regret = 0.0
    n_agents = model.n_agents
    total_regret_by_agt = [0. for i in range(n_agents)]
    total_regret_sq = 0.0
    total_regret_sq_by_agt = [0. for i in range(n_agents)]
    total_payment = 0.0
    regret_max = 0
    n_count = 0
    print(args)
    # ouput payments and allocations directly
    payment_tot = []
    alloc_tot = []
    payment_list = []

    for i, batch in tqdm(enumerate(loader)):
        batch = batch.to(device)
        misreport_batch = batch.clone().detach()
        n_count += batch.shape[0]
        rp_limit = torch.full((batch.shape[0], args.n_agents), args.reserved_price).to(device)

        optimize_misreports(model, batch, misreport_batch, misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model(batch)

        payment_tot.append(payments)
        alloc_tot.append(allocs)

        truthful_util = calc_agent_util(batch, allocs, payments)

        # misreport_allocs, misreport_payments = model(misreport_batch)
        misreport_util = tiled_misreport_util(misreport_batch, batch, model)

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)
        total_regret += positive_regrets.sum().item() / args.n_agents
        total_regret_sq += (positive_regrets**2).sum().item() / args.n_agents
        for i in range(n_agents):
            total_regret_by_agt[i] += positive_regrets[:, i].sum().item()
            total_regret_sq_by_agt[i] += (positive_regrets[:, i]**2).sum().item()

        total_payment += torch.sum(payments).item()
        payment_list.append(total_payment/n_count)

    result = {"payment": total_payment / n_count,
              "regret_std": (total_regret_sq/n_count - (total_regret/n_count)**2)**.5,
              "regret_mean": total_regret/n_count,
              "regret_max": regret_max}
    for i in range(n_agents):
        result[f"regret_agt{i}_std"] = (total_regret_sq_by_agt[i]/n_count - (total_regret_by_agt[i]/n_count)**2)**.5
        result[f"regret_agt{i}_mean"] = total_regret_by_agt[i]/n_count
    return payment_tot, alloc_tot, payment_list, result


def train_loop(
    model,
    train_loader,
    test_loader,
    position_range,
    args,
    device="cpu",
    verbose=True,
    writer=None
):
    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    ir_lagr_mults = 20.0 * torch.ones((1, model.n_agents)).to(device)
    rp_lagr_mults = 40.0 * torch.ones((1, model.loc_n_dim)).to(device)
    payment_mult = 1.0

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    # reserved_position = args.reserved_price

    lagr_update_iter = args.lagr_update_iter
    lagr_update_iter_ir = args.lagr_update_iter_ir
    lagr_update_iter_rp = args.lagr_update_iter_rp
    rho = args.rho
    rho_ir = args.rho_ir

    loss_list = []

    for epoch in tqdm(range(args.num_epochs)):
        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = batch.clone().detach().to(device)
            # rp_limit = torch.full((batch.shape[0], args.n_agents), reserved_position).to(device)

            optimize_misreports(model, batch, misreport_batch, misreport_iter=args.misreport_iter, lr=args.misreport_lr)

            allocs, payments = model(batch)

            # rho_tensor = torch.full((batch.shape[0], args.n_agents), rho).to(device)
            rho_tensor = torch.full((batch.shape[0], args.n_items), rho).to(device)
            rp_lagr_mults_tensor = torch.full((batch.shape[0], args.n_items),
                                              rp_lagr_mults[0][0].item()).to(device)

            truthful_util = calc_agent_util(batch, allocs, payments)
            misreport_util = tiled_misreport_util(misreport_batch, batch, model)
            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)
            quadratic_regrets = positive_regrets ** 2
            regret_loss = (regret_mults * positive_regrets).mean()

            # ir_violation = -torch.clamp(truthful_util, max=0)
            ir_violation = torch.clamp(0.1 - payments, min=0)
            rp_violation = torch.clamp(allocs - position_range[1], min=0)
            rp_violation += torch.clamp(position_range[0]- allocs, min=0)

            # 计算losses
            ir_loss = (torch.abs(ir_violation)**args.ir_penalty_power).mean()
            rp_loss = (rp_lagr_mults *(torch.abs(rp_violation)**args.rp_penalty_power)).mean()
            # rp_loss = calc_rp_loss(model, payments, rp_limit, rp_lagr_mults, rho)

            # DONE(to calculate the investment partial)
            #  batch [-1, n_agents, loc_n_dim]
            # allocs [-1, k_position, loc_n_dim]
            # payment [-1, k_position]
            # https://pytorch.org/docs/stable/generated/torch.cdist.html#torch-cdist

            agent_postion_c_dist = torch.cdist(batch, allocs)
            # [-1, n_agents, k_positions]
            agent_choice = torch.argmin(agent_postion_c_dist, dim = 2)
            # [-1, n_agents] dtype int within k_position
            # payment_loss = payments.sum(dim=1).mean() * payment_mult
            payment_loss = 3 - torch.sqrt((torch.gather(payments, dim=1, index =agent_choice) * truthful_util).sum(dim=1) + 1).mean()  * payment_mult

            loss_func = regret_loss + rp_loss * (1.0 / (2.0 * rho)) +  payment_loss + ir_loss
                        # + (rho / 2.0) * torch.mean(quadratic_regrets) \
                        # + ir_loss\
                        #     + (1.0 / (2.0 * rho)) * rp_loss
                        # - payment_loss\
            loss_np = loss_func.cpu().detach().numpy()
            print(loss_np)
            loss_list.append(loss_np)
            # 更新网络参数
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            # 更新拉格朗日和增广拉格朗日参数
            if iter % lagr_update_iter == 0:
                with torch.no_grad():
                    regret_mults += rho * torch.mean(positive_regrets, dim=0)
            if iter % lagr_update_iter_ir == 0:
                with torch.no_grad():
                    ir_lagr_mults += rho_ir * torch.mean(torch.abs(ir_violation))
            if iter % lagr_update_iter_rp == 0:
                with torch.no_grad():
                    rp_lagr_mults += rho * torch.mean(torch.max(position_range[1] - allocs.max(dim=1)[0], -(rp_lagr_mults_tensor/rho_tensor)) + 
                    torch.max(allocs.min(dim=1)[0] - position_range[0], -(rp_lagr_mults_tensor/rho_tensor)), dim=0)
            if iter % args.rho_incr_iter == 0:
                rho += args.rho_incr_amount
            if iter % args.rho_incr_iter_ir == 0:
                rho_ir += args.rho_incr_amount_ir

        if epoch % 5 == 4:
            # TODO(sujinhua): make this run success
            _, _, _, test_result = test_loop(model, test_loader, args, device=device)
            # plot_payment(model, grid_width=0.01, name=f"{args.name}_{epoch}")
            # plot_12_model(model, grid_width=0.01, name=f"{args.name}_{epoch}")
            arch = {'n_agents': model.n_agents,
                    # 'n_items': model.n_,
                     'loc_n_dim' : model.loc_n_dim,
                    'k_locations' : model.k_locations,
                    'hidden_layer_size': model.hidden_layer_size,
                    'n_hidden_layers': model.n_hidden_layers,
                    'clamp_op': model.clamp_opt,
                    'activation': model.activation,
                    'p_activation': model.p_activation,
                    'a_activation': model.a_activation,
                    'separate': model.separate}
            torch.save({'name': args.name,
                        'arch': arch,
                        'state_dict': model.state_dict(),
                        'args': args}, f"result/{args.name}_{epoch}_checkpoint.pt")
    return loss_list
