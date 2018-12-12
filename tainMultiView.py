from base_model2 import MultiViewModel
import torch
from torch.autograd import Variable
model = MultiViewModel(n_classe=750, n_view=4)
#print(model)
#for name, parameters in model.named_parameters():
#    print(name, ':', parameters.size())

if __name__ == '__main__':
    batch = Variable(torch.randn(4, 1, 3, 224, 224)) # n_views x b x c x h x w
    model = model.to('cuda:0')
    batch = batch.to('cuda:0')
    predict = model(batch)
    print(predict)