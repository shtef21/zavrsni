
from src.models import *
from src.train import train as train_model

for train_epochs in range(1, 82, 10):

  # 1, 10, ..., 80
  if train_epochs >= 11:
    train_epochs -= 1

  for model, name in [
    ( MnistSimpleCNN(kernel_size=3, output_size=27), 'MnistSimpleCNN_3x3' ),
    ( MnistSimpleCNN(kernel_size=3, output_size=27, dropout=True), 'MnistSimpleCNN_3x3_drop' ),
    ( MnistSimpleCNN(kernel_size=5, output_size=27), 'MnistSimpleCNN_5x5' ),
    ( MnistSimpleCNN(kernel_size=5, output_size=27, dropout=True), 'MnistSimpleCNN_5x5_drop' ),

    ( MnistSimpleCNN(kernel_size=7, output_size=27, dropout=True), 'MnistSimpleCNN_7x7_drop' ),

    ( SimpleNetv1(output_size=27), 'SimpleNetv1' ),
    ( SimpleNetv1(output_size=27, dropout=True), 'SimpleNetv1_drop' ),

    ( Resnet18_EMNIST(output_size=27), 'Resnet18' ),
    ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1), 'Resnet18_drop_25' ),
    ( Resnet18_dd(output_size=27, residual_drop_p=0.5, in_channels=1), 'Resnet18_drop_50' ),
    ( Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1), 'Resnet18_drop_75' ),

    ( ResidualNet(output_size=27), 'ResidualNet' ),
    ( ConvNet(output_size=27), 'ConvNet' ),

    ( ProfNet(output_size=27), 'ProfNet' ),
  ]:
    print(name)
    root = './'
    custom_args = {
      # **kwargs
    }
    train_model(model, name, train_epochs, root, **custom_args)
