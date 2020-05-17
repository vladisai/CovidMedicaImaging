import torch
import torch.nn.functional as F

import torchxrayvision as xrv


class FeatureExtractor:
    def extract(self, dataset):
        pass


class NeuralNetFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.model = xrv.models.DenseNet(weights='all')
        self.model.eval()
        self.model.cuda()

    def get_features(self, batch):
        d = batch['img'].cuda()
        features = self.model.features(d)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out.cpu()

    def extract(self, dataset):
        """Returns numpy array with 1024 features for each example"""
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        results = []
        # I'm not sure why no grad is necessary here.
        # Calling model.eval() doesn't seem to work, model runs out
        # of memory after a few batches without torch.no_grad().
        with torch.no_grad():
            for i, batch in enumerate(loader):
                features = self.get_features(batch)
                results.append(features)
        results = torch.cat(results, axis=0).numpy()
        return results
