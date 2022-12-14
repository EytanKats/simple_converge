import torch


default_settings = {
    'temperature': 1.0
}


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, settings):
        super(ContrastiveLoss, self).__init__()

        self.temperature = settings['loss']['temperature']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, query, key):

        # Normalize
        query = torch.nn.functional.normalize(query, dim=1)
        key = torch.nn.functional.normalize(key, dim=1)

        # Get logits by calculating cosine distance between each query and key
        logits = torch.einsum('nc,mc->nm', [query, key]) / self.temperature

        # Create labels
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long).to(self.device)

        # Calculate loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels) * (2 * self.temperature)

        return loss
