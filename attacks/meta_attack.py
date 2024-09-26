import torch
import random
import numpy as np
import torch.nn.functional as F
import scipy.special as spec
import higher

EPS = torch.finfo(torch.float32).tiny
INF = np.finfo(np.float32).max
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def softtopk_forward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, 0, 0] = 0
    messages[:, 0, 1] = logits[:, 0]
    for i in range(1, n):
        for j in range(k + 1):
            logp_dont_use = messages[:, i - 1, j]
            logp_use = (
                messages[:, i - 1, j - 1] + logits[:, i] if j > 0 else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages

def softtopk_backward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, n - 1, k] = 0
    for i in range(n - 2, -1, -1):
        for j in range(k + 1):
            logp_dont_use = messages[:, i + 1, j]
            logp_use = (
                messages[:, i + 1, j + 1] + logits[:, i + 1] if j < k else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages

def softtopk_np(logits, k):
    batchsize = logits.shape[0]
    f = softtopk_forward_np(logits, k)
    b = softtopk_backward_np(logits, k)
    initial_f = -INF * np.ones((batchsize, 1, k + 1))
    initial_f[:, :, 0] = 0
    ff = np.concatenate([initial_f, f[:, :-1, :]], axis=1)
    lse0 = spec.logsumexp(ff + b, axis=2)
    lse1 = spec.logsumexp(ff[:, :, :-1] + b[:, :, 1:], axis=2) + logits
    return np.exp(lse1 - np.logaddexp(lse0, lse1))

class SoftTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k, eps):
        # ctx is a context object that can be used to stash information
        # for backward computation.
        ctx.save_for_backward(logits)
        ctx.k = k
        ctx.eps = eps
        dtype = logits.dtype
        device = logits.device
        print("hi")
        mu_np = softtopk_np(logits.cpu().detach().numpy(), k)
        mu = torch.from_numpy(mu_np).type(dtype).to(device)
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        r"""http://www.cs.toronto.edu/~kswersky/wp-content/uploads/carbm.pdf"""
        logits, = ctx.saved_tensors
        k = ctx.k
        eps= ctx.eps
        dtype = grad_output.dtype
        device = grad_output.device
        logits_np = logits.cpu().detach().numpy()
        grad_output_np = grad_output.cpu().detach().numpy()
        n1 = softtopk_np(logits_np + eps * grad_output_np, k)
        n2 = softtopk_np(logits_np - eps * grad_output_np, k)
        grad_np = (n1 - n2) / (2 * eps)
        grad = torch.from_numpy(grad_np).type(dtype).to(device)
        return grad, None, None

def k_subset_selection(soft_top_k, log_lab, BUDGET):
	"""
	k-hot vector selection from Stochastic Softmax Tricks
	"""
	_, topk_indices = torch.topk(log_lab, BUDGET)
	X = torch.zeros_like(log_lab).scatter(-1, topk_indices, 1.0)
	hard_X = X
	hard_topk = (hard_X - soft_top_k).detach() + soft_top_k
	budget_top_k = hard_topk.T
	
	return budget_top_k

def meta_attack(model, data, epsilon, seed, args):
	
    np.random.seed(seed)
    random.seed(seed)
    model = model.to(device)
    print(torch.sum(data.train_mask).item())
    # find the number of nodes in train set
    BUDGET = np.ceil(torch.sum(data.train_mask).item() * (args.df_size)).astype(int)

    num_classes = data.y.max() + 1
    eye = torch.eye(num_classes).to(device)
    inner_opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #args.lr #args.weight_decay
    Y = eye[data.y]
    y_gt = Y.argmax(1)

    """
    Setting 3: flip based on target model logits
    """

    ab_train_ids = range(data.train_mask.sum())
    train_preds = F.softmax(model(data.x, data.edge_index)[data.train_mask], dim=1)
    Y_false = 1 - Y[data.train_mask]
    Y_L_flipped = eye[torch.argmax(train_preds * Y_false, 1)]

    # reset model weights
    # model.apply(weight_reset)

    # top-k based log-lab
    top_k = SoftTopK() # SST based soft-top-k

    # if args.naive_log_lab:
        # Naive log-lab
    log_lab = torch.nn.Parameter(torch.log(eye[data.y[data.train_mask]]+0.01))
    log_lab_H = torch.nn.Parameter(torch.zeros(data.train_mask.sum(), 1))
    torch.nn.init.uniform_(log_lab_H)
        # torch.nn.init.uniform_(log_lab)
    # else:
        # log_lab = torch.nn.Parameter(torch.zeros(1, data.false_onehot.shape[0]).to(device)) # for SIMPLE top-k
        # torch.nn.init.uniform_(log_lab)

    # H = torch.nn.Parameter(torch.zeros(Y_L_flipped.shape[0], 1))
    H = torch.nn.Parameter(torch.zeros(len(ab_train_ids), 1))
    torch.nn.init.uniform_(H)
    meta_opt = torch.optim.Adam([H], lr=0.1, weight_decay=1e-5)

    # for tracking best poisoned labels w.r.t meta test accuracy
    best_test_acc = 200
    best_poisoned_labels = None
    PATIENCE = 20
    patience_ctr = 0

    for ep in range(100): #150
        #poison_model.train()
        print(ep)
        BUDGET = min(BUDGET, H.shape[0])
        soft_top_k = top_k.apply(H.T, BUDGET, 1e-2)
        budget_top_k = k_subset_selection(soft_top_k, H.T, BUDGET).cuda()

        zeros_vec = torch.zeros(Y_L_flipped.shape[0], 1).cuda()
        zeros_vec[ab_train_ids] = budget_top_k
        poisoned_ = (Y_L_flipped * zeros_vec)
        correct_ = Y[data.train_mask] * (1 - zeros_vec)

        poisoned_labels = poisoned_ + correct_

        print('hello')
        
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
            fmodel.train()
            print('hello2')
            for epoch in range(15):
                print(epoch)
                out = fmodel(data)
                if args.naive_log_lab:
                    loss = F.cross_entropy(out[data.train_mask], poisoned_labels)
                # else:
                #     # soft margin loss
                #     # loss = -margin_loss_soft(out[data.train_mask], poisoned_labels).tanh().mean()
                    # loss = F.cross_entropy(out[data.train_mask], poisoned_labels) #F.softmax(poisoned_labels, dim=1)
                diffopt.step(loss)

            fmodel.eval()
            meta_out = fmodel(data)

            # train_acc = (meta_out.argmax(1)[data.train_mask] == log_lab.argmax(1)).sum() / data.train_mask.sum() 
            train_acc = (meta_out.argmax(1)[data.train_mask] == poisoned_labels.argmax(1)).sum() / data.train_mask.sum()
            acc = (meta_out.argmax(1)[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
            # n_poisoned = (log_lab.argmax(1) != data.y[data.train_mask]).sum() 
            n_poisoned = (poisoned_labels.argmax(1) != data.y[data.train_mask]).sum()

            # early stopping based on meta-test acc
            if (acc < best_test_acc):
                best_test_acc = acc
                best_poisoned_labels = poisoned_labels.type(torch.float32).detach() #log_lab.detach()
                patience_ctr = 0 
            else:
                patience_ctr += 1

            if patience_ctr >= PATIENCE:
                break

            # convert meta out to binary using top-k
            meta_test_out = meta_out[data.test_mask]

            # 0-1 gumbel loss
            meta_out_bin = F.gumbel_softmax(meta_test_out, tau=100, hard=True, dim=1)
            meta_loss = (Y[data.test_mask] * meta_out_bin).sum()/len(meta_out_bin)

            # if verbose:
            #     # print('acc', acc.cpu().numpy(), 'n_poisoned', n_poisoned.cpu().numpy(), 'diff', diff.cpu().numpy())
            print("epoch: {:4d} \t Meta-Train Acc: {:.2f} \t Meta-Test Acc: {:.2f} \t N-poisoned: {:4d} \t patience ctr: {:2d} \t MetaLoss: {:.4f}".format(ep, \
                                                                                        train_acc.cpu().numpy(), acc.cpu().numpy()*100, \
                                                                                        n_poisoned.cpu().numpy(), patience_ctr,
                                                                                        meta_loss.detach().cpu().numpy())) #diff.cpu().numpy()

            meta_opt.zero_grad()
            meta_loss.backward()
            meta_opt.step()

    # if verbose:
    print("Best Test Acc (attacked): {:.2f}".format(best_test_acc * 100))

    #poison_model.apply(weight_reset)
    torch.cuda.empty_cache()

    # return log_lab.detach()
    # this is the train set modified labels
    print(len(best_poisoned_labels))

    data.y[data.train_mask] = best_poisoned_labels.argmax(1)

    # Identify poisoned indices (nodes whose labels were changed)
    poisoned_indices = (data.y[data.train_mask] != y_gt[data.train_mask]).nonzero(as_tuple=True)[0]
    data.poisoned_nodes = poisoned_indices.detach().cpu()

    torch.cuda.empty_cache()

    print(f"Number of poisoned nodes: {len(poisoned_indices)}")

    return data, poisoned_indices