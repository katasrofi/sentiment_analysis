import torch
from tqdm.auto import tqdm
import numpy as np

def custom_train_test_split(X, y):
    X = np.array(X)
    y = np.array(y)
    from sklearn.model_selection import train_test_split
    X_train, y_train, X_test, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=1)
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
    return X_train, y_train, X_test, y_test

torch.manual_seed(1)
def training_loop(model,
                  X_train, 
                  y_train,
                  loss_fn,
                  optimizer,
                  accuracy_fn,
                  EPOCHS,
                  device = torch.device("cpu")):
    for epochs in tqdm(range(EPOCHS)):
        # Start Training
        model.train()
        
        # Predict the result
        y_pred = model(X_train)
        
        # Calculate the Loss
        loss = loss_fn(y_pred, y_train)
        
        # Calculate the accuracy
        accuracy = accuracy_fn(y_train, y_pred)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epochs % 16 == 0:
            print(f"Epochs {epochs} | loss = {loss.item()} | accuracy = {accuracy}" )
            
    return {"Model_name ": model.__class__.__name__,
            "Model_loss ": loss.item(),
            "Model_accuracy ": accuracy}