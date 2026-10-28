import torch


import torch.nn as nn


class BasePromptLearner(nn.Module):
    """
    Base class for prompt learners.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, txt_feats=None):
        """
        Forward method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        """
        String representation of the prompt learner.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PromptLearner(BasePromptLearner):
    """
    Learnable context prompt for text encoder.
    """
    def __init__(self, prompt_length=16, embed_dim=512, ctx_init=None):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        if ctx_init is not None:
            self.prompt = nn.Parameter(ctx_init.clone().detach())  # shape [prompt_length, embed_dim]
            self.prompt_length = self.prompt.shape[0]
        else:
            self.prompt = nn.Parameter(torch.randn(prompt_length, embed_dim))

    def forward(self, x, txt_feats):
        """
        Forward method to concatenate the prompt with the text features.
        """
        batch_size = x.shape[0]
        n_ctx = self.prompt_length
        n_cls = txt_feats.shape[0]
        prompt = self.prompt.unsqueeze(0).expand(n_cls, n_ctx, self.embed_dim)

        txt_feats = torch.cat((txt_feats[:, 0:1, :], prompt, txt_feats[:, n_ctx+1:, :]), dim=1)
        return txt_feats

    def __str__(self):
        """
        String representation of the prompt learner.
        """
        return f"PromptLearner(prompt_length={self.prompt_length}, embed_dim={self.embed_dim}, ctx_init.shape={self.prompt.shape})"


