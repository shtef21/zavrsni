from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision import transforms 

from src.models import *
from glob import glob 
from pprint import pprint 
from tqdm import tqdm 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoints = glob('./checkpoints/*')
print('Found', len(model_checkpoints), 'checkpoints.')

# Top 10 models (best_models.png)
models = {
  'Resnet18_drop_25_224px-lr_scheduler-best.chk': Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1),

  'ProfNet-best.chk': ProfNet(output_size=27),

  'MnistSimpleCNN_3x3-best.chk': MnistSimpleCNN(kernel_size=3, output_size=27),

  'ResidualNet-best.chk': ResidualNet(output_size=27),

  'Resnet18_drop_25-lr_scheduler-best.chk': Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1),

  'MnistSimpleCNN_5x5-best.chk': MnistSimpleCNN(kernel_size=5, output_size=27),

  'Resnet18_drop_25_224px-best.chk': Resnet18_dd(output_size=27, residual_drop_p=0.25, in_channels=1),

  'ConvNet-best.chk': ConvNet(output_size=27),

  'Resnet18_drop_50_224px-best.chk': Resnet18_dd(output_size=27, residual_drop_p=0.5, in_channels=1),

  'Resnet18_drop_75_224px-best.chk': Resnet18_dd(output_size=27, residual_drop_p=0.75, in_channels=1),
}

acc_dict = {
    # 'model_name': acc,
}

for idx, model_filename in enumerate(models):
    model = models[model_filename].to(device)
    model_path = './checkpoints/' + model_filename

    if '224px' in model_filename:
      transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
      ])
    else:
      transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
      ])

    chk = torch.load(model_path, map_location=device)
    model.load_state_dict(chk['model'])
    testset = datasets.EMNIST('data', split='letters', download=True, train=False, transform=transform)
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
print('Accuracies:')
pprint(sorted(list(acc_dict.items()), key=lambda e: e[1]))
