import torch
from scipy import special
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace


# ==================================== AUTOBOT-EGO STUFF ====================================

def get_BVG_distributions(pred):
    B = pred.size(0)
    T = pred.size(1)
    mu_x = pred[:, :, 0].unsqueeze(2)
    mu_y = pred[:, :, 1].unsqueeze(2)
    sigma_x = pred[:, :, 2]
    sigma_y = pred[:, :, 3]
    rho = pred[:, :, 4]

    cov = torch.zeros((B, T, 2, 2)).to(pred.device)
    cov[:, :, 0, 0] = sigma_x ** 2
    cov[:, :, 1, 1] = sigma_y ** 2
    cov[:, :, 0, 1] = rho * sigma_x * sigma_y
    cov[:, :, 1, 0] = rho * sigma_x * sigma_y

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist(pred):
    return Laplace(pred[:, :, :2], pred[:, :, 2:4])


def nll_pytorch_dist(pred, data, rtn_loss=True):
    # biv_gauss_dist = get_BVG_distributions(pred)
    biv_gauss_dist = get_Laplace_dist(pred)
    if rtn_loss:
        # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(-1).sum(1)  # Laplace
    else:
        # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(dim=(1, 2))  # Laplace


def nll_loss_multimodes(pred, data, modes_pred, entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True):
    """NLL loss multimodes for training. MFP Loss function
    Args:
      pred: [K, T, B, 5]
      data: [B, T, 5]
      modes_pred: [B, K], prior prob over modes
      noise is optional
    """
    modes = len(pred)
    nSteps, batch_sz, dim = pred[0].shape

    # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=False)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = modes_pred.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(data.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=True) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions(pred[kk]).entropy())
    entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
    entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(modes_pred), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde(pred, data)
    else:
        adefde_loss = torch.tensor(0.0).to(data.device)

    return loss, kl_loss, post_entropy, adefde_loss


def l2_loss_fde(pred, data):
    fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    ade_loss = torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    return 100.0 * loss.mean()


# ==================================== AUTOBOT-JOINT STUFF ====================================


def get_BVG_distributions_joint(pred):
    B = pred.size(0)
    T = pred.size(1)
    N = pred.size(2)
    mu_x = pred[:, :, :, 0].unsqueeze(3)
    mu_y = pred[:, :, :, 1].unsqueeze(3)
    sigma_x = pred[:, :, :, 2]
    sigma_y = pred[:, :, :, 3]
    rho = pred[:, :, :, 4]

    cov = torch.zeros((B, T, N, 2, 2)).to(pred.device)
    cov[:, :, :, 0, 0] = sigma_x ** 2
    cov[:, :, :, 1, 1] = sigma_y ** 2
    cov_val = rho * sigma_x * sigma_y
    cov[:, :, :, 0, 1] = cov_val
    cov[:, :, :, 1, 0] = cov_val

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist_joint(pred):
    return Laplace(pred[:, :, :, :2], pred[:, :, :, 2:4])


def nll_pytorch_dist_joint(pred, data, agents_masks):
    # biv_gauss_dist = get_BVG_distributions_joint(pred)
    biv_gauss_dist = get_Laplace_dist_joint(pred)
    num_active_agents_per_timestep = agents_masks.sum(2)
    loss = (((-biv_gauss_dist.log_prob(data).sum(-1) * agents_masks).sum(2)) / num_active_agents_per_timestep).sum(1)
    return loss


def nll_loss_multimodes_joint(pred, gt_agents_and_ego, mode_probs, agents_masks, entropy_weight=1.0, kl_weight=1.0,
                              use_FDEADE_aux_loss=True, predict_yaw=False):
    """
    Args:
      pred: [c, T, B, M+1, 6]
      gt_agents_and_ego: [B, T, M+1, 3]
      mode_probs: [B, c], prior prob over modes
      agents_masks: [B, T, M+1]
    """

    # gt_agents_and_ego = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2)
    modes = len(pred)
    nSteps, batch_sz, N, dim = pred[0].shape
    # agents_masks = torch.cat((torch.ones(batch_sz, nSteps, 1).to(gt_agents_and_ego.device), agents_data[:, :, :, -1]), dim=-1)

    agents_masks = agents_masks

    # compute posterior probability based on predicted prior and likelihood of predicted scene.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents_and_ego[:, :, :, :2], agents_masks)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = mode_probs.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=1).reshape((batch_sz, 1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(gt_agents_and_ego.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents_and_ego[:, :, :, :2], agents_masks) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions_joint(pred[kk]).entropy())
    entropy_loss = torch.mean(torch.stack(entropy_vals).permute(2, 0, 3, 1).sum(3).mean(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(mode_probs), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss, var_ade_fde = l2_loss_fde_joint(pred, gt_agents_and_ego, agents_masks, predict_yaw)
    else:
        adefde_loss = torch.tensor(0.0).to(gt_agents_and_ego.device)
        var_ade_fde = torch.tensor(0.0).to(gt_agents_and_ego.device)

    return loss, kl_loss, post_entropy, adefde_loss, var_ade_fde


def l2_loss_fde_joint(pred, data, agent_masks, predict_yaw):
    fde_loss = (torch.norm((pred[:, -1, :, :, :2].transpose(0, 1) - data[:, -1, :, :2].unsqueeze(1)), 2, dim=-1) *
                agent_masks[:, -1:, :]).mean(-1)
    ade_loss = (torch.norm((pred[:, :, :, :, :2].transpose(1, 2) - data[:, :, :, :2].unsqueeze(0)), 2, dim=-1) *
                agent_masks.unsqueeze(0)).mean(-1).mean(dim=2).transpose(0, 1)

    yaw_loss = torch.tensor(0.0).to(pred.device)
    if predict_yaw:
        # yaw_loss = torch.norm(pred[:, :, :, :, 5:].transpose(1, 2) - data[:, :, :, 2:].unsqueeze(0), dim=-1).mean(2)  # across time
        angle = torch.atan2(pred[..., 6], pred[..., 5])
        yaw_loss = torch.norm(angle_diff(angle.transpose(1, 2).unsqueeze(-1), data[:, :, :, 2:].unsqueeze(0)), dim=-1).mean(2)  # across time
        yaw_loss = (yaw_loss).mean(-1).transpose(0, 1)

    losses = fde_loss + ade_loss + yaw_loss # [B, num_modes]
    
    loss_batch = (fde_loss + ade_loss).sum(-1)
    var = torch.var(loss_batch, dim=-1)

    loss, min_inds = losses.min(dim=1)
    return 100.0 * loss.mean(), var

def angle_diff(a, b):
    # unused
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))