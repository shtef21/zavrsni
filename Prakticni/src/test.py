import torch 

def test(model, loader, device, writer, e, loss_fn, optimizer, chk_path, chk_best_path, best_acc):

    # Don't calculate grads
    with torch.no_grad():

      # Evaluate model
      model.eval()
      total, correct = 0, 0
      
      for images, labels in loader: 
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        preds = torch.argmax(out, -1)
        correct += (preds == labels).sum().item()
        total += len(labels)

      # Save checkpoints

      curr_acc = correct / total * 100
      writer.add_scalar('model_acc', curr_acc, e)
      save_obj = {
          'epoch': e,
          'model': model.state_dict(),
          'loss_fn': loss_fn.state_dict(),
          'optimizer': optimizer.state_dict(),
          'accuracy': curr_acc,
      }
      torch.save(save_obj, chk_path)
      
      if curr_acc > best_acc:
        print(f'New best acc: {round(best_acc, 2)}% --> {round(curr_acc, 2)}%')
        torch.save(save_obj, chk_best_path)
        best_acc = curr_acc

      # Set model to train mode
      model.train()
