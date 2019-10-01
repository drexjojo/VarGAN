import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *
from torch.autograd import Function, Variable

#Modified loss to include KL-Divergence
class ModifiedLoss(nn.Module):
	def __init__(self):
		super(ModifiedLoss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID,reduction="sum")
		self.kld_weight = 1

	def compute_batch_loss(self, outputs, targets, normalization,kld_loss):
		cont_outputs = outputs.contiguous().view(-1,outputs.shape[-1])
		cont_targets = targets.contiguous().view(-1)
		ce_loss = self.criterion(cont_outputs, cont_targets)

		loss = ce_loss + self.kld_weight * kld_loss
		loss = loss.div(normalization)
		loss_dict = {
			"CELoss": float(ce_loss)/normalization, 
			"KLDLoss":float(kld_loss)/normalization
			}
		del ce_loss, kld_loss
		accuracy = self.get_accuracy(cont_outputs, cont_targets)
		return loss, loss_dict, accuracy

	def get_accuracy(self, probs, golds):
		probs = F.log_softmax(probs,dim=-1)
		preds = probs.data.topk(1, dim=-1)[1]
		non_padding = golds.ne(PAD_ID) 
		correct = preds.squeeze().eq(golds).masked_select(non_padding)
		num_words = non_padding.long().sum()

		num_correct = correct.long().sum()
		accuracy = 100 * (float(num_correct.item())/num_words.item())
		return accuracy

#Helper class to track the gradient flow
class Check_Grad(Function):
	def forward(self, x):
		print("forwarding")
		return x.view_as(x)

	def backward(self, grad_output):
		print(grad_output)
		return grad_output.view_as(grad_output)

def check_grad(x):
	return Check_Grad()(x)