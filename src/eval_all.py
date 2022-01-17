from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision import transforms 

from models import *
import glob 
import pprint 
import tqdm 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoints = glob('./checkpoints/*')
print('Found', len(model_checkpoints), 'checkpoints.')

models = {
    'ConvNet-best.chk': ConvNet(output_size=27),
    'ConvNet.chk': ConvNet(output_size=27),
    'MnistSimpleCNN_3x3_drop-best.chk': MnistSimpleCNN(kernel_size=3, output_size=27, dropout=True),
    'MnistSimpleCNN_3x3_drop.chk': MnistSimpleCNN(kernel_size=3, output_size=27, dropout=True),
    'MnistSimpleCNN_3x3-best.chk': MnistSimpleCNN(kernel_size=3, output_size=27),
    'MnistSimpleCNN_3x3.chk': MnistSimpleCNN(kernel_size=3, output_size=27),
    'MnistSimpleCNN_5x5_drop-best.chk': MnistSimpleCNN(kernel_size=5, output_size=27, dropout=True),
    'MnistSimpleCNN_5x5_drop.chk': MnistSimpleCNN(kernel_size=5, output_size=27, dropout=True),
    'MnistSimpleCNN_5x5-best.chk': MnistSimpleCNN(kernel_size=5, output_size=27),
    'MnistSimpleCNN_5x5.chk': MnistSimpleCNN(kernel_size=5, output_size=27),
    'MnistSimpleCNN_7x7_drop-best.chk': MnistSimpleCNN(kernel_size=7, output_size=27, dropout=True),
    'MnistSimpleCNN_7x7_drop.chk': MnistSimpleCNN(kernel_size=7, output_size=27, dropout=True),
    'ProfNet-best.chk': ProfNet(output_size=27),
    'ProfNet.chk': ProfNet(output_size=27),
    'ResidualNet-best.chk': ResidualNet(output_size=27),
    'ResidualNet.chk': ResidualNet(output_size=27),
    'Resnet18_drop_25-best.chk': Resnet18_dd(27, 0.25, in_channels=3),
    'Resnet18_drop_25.chk': Resnet18_dd(27, 0.25, in_channels=3),
    'Resnet18_drop_50-best.chk': Resnet18_dd(27, 0.50, in_channels=3),
    'Resnet18_drop_50.chk': Resnet18_dd(27, 0.50, in_channels=3),
    'Resnet18_drop_75-best.chk': Resnet18_dd(27, 0.75, in_channels=3),
    'Resnet18_drop_75.chk': Resnet18_dd(27, 0.75, in_channels=3),
    'Resnet18-best.chk': Resnet18_EMNIST(output_size=27),
    'Resnet18.chk': Resnet18_EMNIST(output_size=27),
    'SimpleNetv1_drop-best.chk': SimpleNetv1(output_size=27, dropout=True),
    'SimpleNetv1_drop.chk': SimpleNetv1(output_size=27, dropout=True),
    'SimpleNetv1-best.chk': SimpleNetv1(output_size=27),
    'SimpleNetv1.chk': SimpleNetv1(output_size=27),
    
    # 'MnistSimpleCNN.chk': None,
}

acc_dict = {
    # 'model_name': acc,
}

for idx, model_filename in enumerate(models):
    model = models[model_filename][0].to(device)
    model_path = './checkpoints/' + model_filename

    chk = torch.load(model_path, map_location=device)
    model.load_state_dict(chk['model'])
    testset = datasets.EMNIST('data', split='letters', download=True, train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ]))
    test_loader = DataLoader(testset, batch_size=16, shuffle=True)

    with torch.no_grad():
      model.eval()
      correct, total = 0, 0
      loop = tqdm(enumerate(test_loader), total=len(test_loader))
      loop_name = model_filename.replace('.chk', '')
      loop.set_description(f'{loop_name} model={idx + 1}/{len(models)}, model_epoch={chk["epoch"]}')

      for idx, (images, labels) in loop:
        
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        preds = torch.argmax(out, -1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        acc_str = round(correct / total * 100, 4)

        loop.set_postfix(acc=str(acc_str) + '%')
        acc_dict[loop_name] = acc_str

print()
print('accuracies:')
pprint(sorted(list(acc_dict.items()), key=lambda e: e[1]))
