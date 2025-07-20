import wandb
import tqdm
import torch

def batch_log(step, loss, optimizer, log_freq):
    if step % log_freq == 0:
        wandb.log({
            "train/loss": loss,
            "train/lr": optimizer.param_groups[0]['lr'],
        })

def epoch_log(epoch, train_loss, val_loss, num_epochs):
    print(f"epoch [{epoch:2}/{num_epochs}] train_loss {train_loss:.4f}")
    print(f"epoch [{epoch:2}/{num_epochs}] val_loss {val_loss:.4f}")
    
    wandb.log({
        "train/epoch_loss": train_loss,
        "val/epoch_loss": val_loss,
        "epoch": epoch
    })

def train(epoch, model, loader, criterion, optimizer, scheduler, device, log_freq):
    model.train()
    
    total_loss = 0

    for step, (idx, rating) in enumerate(tqdm.tqdm(loader, desc=f"Train epoch {epoch}")):
        idx, rating = idx.to(device, non_blocking=True), rating.to(device, non_blocking=True)

        outputs = model(idx)
        loss = criterion(outputs, rating)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_log(step, loss.item(), optimizer, log_freq)

    if scheduler:
        scheduler.step()
    
    return total_loss / len(loader)

def validate(epoch, model, loader, criterion, device):
    model.eval()
    
    total_loss = 0

    with torch.no_grad():
        for idx, rating in tqdm.tqdm(loader, desc=f"Val epoch {epoch}"):
            idx, rating = idx.to(device, non_blocking=True), rating.to(device, non_blocking=True)

            outputs = model(idx)
            loss = criterion(outputs, rating)

            total_loss += loss.item()

    return total_loss / len(loader)

class EarlyStopper:
    def __init__(self, model, model_path, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.model = model
        self.model_path = model_path
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            print(f'best_val_loss {val_loss:.4f}, save model!')
            torch.save(self.model.module.state_dict(), self.model_path)
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def test(model, loader, criterion, device):
    model.eval()
    
    total_loss = 0

    with torch.no_grad():
        for idx, rating in tqdm.tqdm(loader):
            idx, rating = idx.to(device, non_blocking=True), rating.to(device, non_blocking=True)

            outputs = model(idx)
            loss = criterion(outputs, rating)

            total_loss += loss.item()

    return total_loss / len(loader)