import models
import loaders
import torch

test_nn = models.three_layer_net(input_layer=144, output_layer=2)
test_nn.train('train(SD).csv', epochs=1, loader='ii')
test_nn.test('test(SD).csv', loader='ii')

results = test_nn.get_results()
print(results)
