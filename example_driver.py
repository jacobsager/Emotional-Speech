import models
import loaders
import torch

#csv = loaders.csv_loader('/Users/jakesager/Desktop/NSA/Features.nosync/400 ms, smart test/test(SD).csv')
#
#print(csv.__getitem__(50))
#
#trainloader = torch.utils.data.DataLoader(csv, shuffle=True)
#print(trainloader)
#a,b,c = 0,0,0
#print(a)
#print(c)
#print(loaders)
#print(models)

test_nn = models.three_layer_net(input_layer=144, output_layer=2)
test_nn.train('/Volumes/mceh-nsalab$/Users/jsager2/Features/400 ms, smart test, binary/train(SD).csv', epochs=1, loader='ii')
test_nn.test('/Volumes/mceh-nsalab$/Users/jsager2/Features/400 ms, smart test, binary/test(SD).csv', loader='ii')

results = test_nn.get_results()
print(results)
