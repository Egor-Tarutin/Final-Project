import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(self.device)
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).to(device)
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).to(device)
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.to(device)
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.to(device)
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).to(device)
        lbs = torch.randint(0, 19, [16, 20, 20]).to(device)
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
