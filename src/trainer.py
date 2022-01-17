
from torch import optim
from torchvision import transforms 
from torchvision import datasets 
from src.models import *
from src.train import train as train_model

# Train all models with Adam (no lr_scheduler)
def train_all_adam():

    for train_epochs in range(1, 82, 10):

        # 1, 10, 20, ... 200 
        if train_epochs >= 11:
            train_epochs -= 1

        for model, name in [

            # MnistSimpleCNN
                                        
            ( MnistSimpleCNN(kernel_size=3, output_size=27), 'MnistSimpleCNN_3x3' ),
            ( MnistSimpleCNN(kernel_size=3, output_size=27, dropout=True), 'MnistSimpleCNN_3x3_drop' ),
            ( MnistSimpleCNN(kernel_size=5, output_size=27), 'MnistSimpleCNN_5x5' ),
            ( MnistSimpleCNN(kernel_size=5, output_size=27, dropout=True), 'MnistSimpleCNN_5x5_drop' ),
            ( MnistSimpleCNN(kernel_size=7, output_size=27, dropout=True), 'MnistSimpleCNN_7x7_drop' ),

            # SimpleNetv1

            ( SimpleNetv1(output_size=27), 'SimpleNetv1' ),
            ( SimpleNetv1(output_size=27, dropout=True), 'SimpleNetv1_drop' ),

            # Resnet18

            ( Resnet18_EMNIST(output_size=27), 'Resnet18' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1), 'Resnet18_drop_25' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.5, in_channels=1), 'Resnet18_drop_50' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1), 'Resnet18_drop_75' ),

            ( Resnet18_EMNIST(output_size=27), 'Resnet18_224px' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1), 'Resnet18_drop_25_224px' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.5, in_channels=1), 'Resnet18_drop_50_224px' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1), 'Resnet18_drop_75_224px' ),

            # ResidualNet
            ( ResidualNet(output_size=27), 'ResidualNet' ),

            # ConvNet
            ( ConvNet(output_size=27), 'ConvNet' ),

            # ProfNet
            ( ProfNet(output_size=27), 'ProfNet' ),
        ]:
            custom_args = {
                'batch_size': 128,
            }
            if '224px' in name:
                trsf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
                custom_args['trainset'] = datasets.EMNIST('data', split='letters', train=True, download=True, transform=trsf)
                custom_args['testset'] = datasets.EMNIST('data', split='letters', train=False, download=True, transform=trsf)

            print(name)
            root = './'
            train_model(model, name, train_epochs, root, **custom_args)


# Train best Adam model on SGD with lr_scheduler
def train_best_sgd_sched():

    for train_epochs in range(1, 82, 5):

        # 1, 5, 10, ..., 80
        if train_epochs >= 6:
            train_epochs -= 1

        for model, name in [
            ( ProfNet(output_size=27), 'ProfNet-lr_scheduler' ),
            ( MnistSimpleCNN(kernel_size=3, output_size=27), 'MnistSimpleCNN_3x3-lr_scheduler' ),
            ( ResidualNet(output_size=27), 'ResidualNet-lr_scheduler' ),
            ( ConvNet(output_size=27), 'ConvNet-lr_scheduler' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1), 'Resnet18_drop_25_224px-lr_scheduler' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1, out_drop=True), 'Resnet18_drop_25_d10_224px-lr_scheduler' ), # Depth dropout = .25, out drop = .1
            ( Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1), 'Resnet18_drop_75_224px-lr_scheduler' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1), 'Resnet18_drop_25-lr_scheduler' ),
            ( Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1), 'Resnet18_drop_75-lr_scheduler' ),
        ]:
            custom_args = {
                'batch_size': 128,
                'use_scheduler': True,
                'optimizer': optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
            }
            if '224px' in name:
                trsf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
                custom_args['trainset'] = datasets.EMNIST('data', split='letters', train=True, download=True, transform=trsf)
                custom_args['testset'] = datasets.EMNIST('data', split='letters', train=False, download=True)
            print(name)
            root = './'
            train_model(model, name, train_epochs, root, **custom_args)
