import torch
from torch.distributions.normal import Normal
from scipy.stats import norm


def expected_improvement(mean, var, reference):
	"""
	expected_improvement for minimization problems
	:param mean: 
	:param var: 
	:param reference: 
	:return: 
	"""
	predictive_normal = Normal(mean.new_zeros(mean.size()), mean.new_ones(mean.size()))
	std = torch.sqrt(var)
	standardized = (-mean + reference) / std
	return (std * torch.exp(predictive_normal.log_prob(standardized)) + (-mean + reference) * predictive_normal.cdf(standardized)).clamp(min=0)

def upper_confidence_bounds(mean, var, reference):
	kappa=2.576
	std = torch.sqrt(var)
	return mean+kappa*std

def probability_of_improvement(mean, var, reference):
	std=torch.sqrt(var)
	z=(mean-reference)/std
	# return norm.cdf(z)
	return z
