

def optimize_model_settings(model):
	model.config.use_cache = False
	model.gradient_checkpointing_enable()


def freeze_model(model):
	model.eval()
	for param in model.parameters():
		param.requires_grad = False