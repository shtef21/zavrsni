import torch 
from torch import nn 
from torch import optim
from torchvision import transforms
import torchvision.datasets as torch_datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import test as test_model
import os 
from tqdm import tqdm 

def train(model, name, epochs, root, **kwargs):
  """
    Custom parameters in kwargs:
      trainset, testset (default: ExtendedEmnist)
      batch_size (default: 256)
      optimizer (default: Adam)
      use_scheduler (default: False) - if True, use StepLR
  """

  def argtry(arg, default=None):
    return kwargs[arg] if arg in kwargs else default

  # Setup parameters
  os.makedirs(f'{root}/checkpoints', exist_ok=True)
  os.makedirs(f'{root}/runs', exist_ok=True)
  cuda = torch.cuda.is_available()
  device = 'cuda' if cuda else 'cpu'
  batch_size = argtry('batch_size', 256)
  use_scheduler = argtry('use_scheduler', False)

  # Dataset
  trainset = argtry('trainset')
  testset = argtry('testset')
  if trainset == None or testset == None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = torch_datasets.EMNIST('data', split='letters', train=True, download=True, transform=transform)
    testset = torch_datasets.EMNIST('data', split='letters', train=False, download=True, transform=transform)
    
  # Dataloaders
  train_loader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True)
  test_loader = DataLoader(testset, batch_size=32, pin_memory=True)

  # Model
  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss().to(device)
  optimizer = argtry('optimizer', optim.Adam(model.parameters()))
  lr_scheduler = None 

  if use_scheduler:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10)

  # Current epoch stats
  curr_acc, epoch, batch_acc = 0, 0, 0
  # Best epoch stats
  best_acc, best_epoch = 0, 0
  chk_path = f'{root}/checkpoints/{name}.chk'
  chk_best_path = f'{root}/checkpoints/{name}-best.chk'

  # Load checkpoint
  if os.path.exists(chk_path):
    chk = torch.load(chk_path, map_location=device)
    epoch = chk['epoch'] + 1
    model.load_state_dict(chk['model'])
    loss_fn.load_state_dict(chk['loss_fn'])
    optimizer.load_state_dict(chk['optimizer'])
    curr_acc = chk['accuracy']
    print('Loaded model. ', end='')

  # Load best model
  if os.path.exists(chk_best_path):
    chk = torch.load(chk_best_path, map_location=device)
    best_epoch = chk['epoch'] + 1
    best_acc = chk['accuracy']
    print('Loaded best model. ', end='')

  # Print model stats
  if epoch > 0 and best_epoch > 0:
    print(f'epoch={epoch} best_acc={round(best_acc, 2)} best_acc_e={best_epoch}')

  # Ignore model
  if epoch - best_epoch > 15:
    print(' -> Best accuracy was more than 15 epochs ago. Leaving...')
    return

  # Initialize writer if training isn't done
  if epoch < epochs:
    writer = SummaryWriter(f'{root}/runs/{name}')


  for e in range(epoch, epochs):

    # Setup loop
    images_processed = e * len(trainset)
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    loop.set_description(f'e={e}/{epochs} best={round(best_acc, 2)} curr={round(curr_acc, 2)}')

    for idx, (images, labels) in loop:

      # Move to CUDA
      if cuda:
        images = images.to(device)
        labels = labels.to(device)

      # Main
      optimizer.zero_grad()
      out = model(images)
      loss = loss_fn(out, labels)
      loss.backward()
      optimizer.step()

      # Stats

      preds = torch.argmax(out, 1)
      correct = (preds == labels).sum().item()
      total = len(labels)
      
      batch_acc = round(correct / total * 100.0, 2)
      loop.set_postfix(loss=loss.item(), acc=batch_acc)
      global_step = images_processed + idx * batch_size + len(labels)

      writer.add_scalar('loss', loss.item(), global_step)
      writer.add_scalar('acc', batch_acc, global_step)

  # Scheduler step
  if use_scheduler:
    lr_scheduler.step()

  # Test model after each epoch
  test_model.test(model, test_loader, device, writer, e, loss_fn, optimizer, chk_path, chk_best_path, best_acc)
