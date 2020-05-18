import torch
import torch.nn.functional as F

from feature_engineering import *
import torchxrayvision as xrv

class FeatureExtractor:
    def __init__(self,lbp=True,hog=True,fft=True):
        assert lbp or hog or fft, "no transformations specified"
        self.hog=hog
        self.lbp=lbp
        self.fft=fft

    def extract(self,dataset):
        results=[]
        for example in dataset:
            result=[]
            if self.hog:
                result=np.concatenate([result,get_hog(example['img']).reshape(-1)])
            if self.fft:
                result=np.concatenate([result,get_fft(example['img']).reshape(-1)])
            if self.lbp:
                result=np.concatenate([result,get_lbp(example['img']).reshape(-1)])
            results.append(result)
        return np.array(results)

class NeuralNetFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.model = xrv.models.DenseNet(weights='all')
        self.model.eval()
        self.model.cuda()

    def get_features(self, batch):
        # return batch['lab']
        d = batch['img'].cuda()
        features = self.model.features(d)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = out.cpu()
        # out = torch.cat([out, batch['lab']], axis=1)
        return out

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
