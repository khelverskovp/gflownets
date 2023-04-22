import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

class Model(nn.Module):
    def __init__(self, num_input: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Linear(num_input,64),nn.ReLU(),nn.Linear(64,105))

    def forward(self, x):
        return self.conv(x)
    
    def mols2batch(self,mols):
        batch = Batch.from_data_list(
            mols, follow_batch=['stems', 'bonds'])
        return batch


if __name__ == "__main__":
    model = Model(100)

    x = torch.ones((64,100))

    mol = Data()

    s = model.mols2batch([mol])

    print(s)

    out = model(s)

    print(out)

    

    

