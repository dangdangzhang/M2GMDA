
import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(400, 300)
        self.drop1 = torch.nn.Dropout(0.6)
        self.fc2 = torch.nn.Linear(300, 200)
        self.drop2 = torch.nn.Dropout(0.6)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 50)
        self.fc5 = torch.nn.Linear(50, 1)

    def forward(self, din):

        dout = F.relu(self.drop1(self.fc1(din)))
        dout = F.relu(self.drop2(self.fc2(dout)))

        dout1 = F.relu(self.fc3(dout))
        dout2 = F.relu(self.fc4(dout1))
        dout2 = torch.sigmoid(self.fc5(dout2))
        return MLP.py,dout2

