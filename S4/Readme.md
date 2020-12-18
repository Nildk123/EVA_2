Assignment Objective
### ---------- Reach 99.44% Accuracy on MNIST dataset using less than 20k params-------------

Reached 99.44% Accuracy first at 18th epoch
Number of parameters in the model - 13,322
Receptive Field - 32


    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input -28 OUtput- 28 RF- 3
        self.BatchNorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) #input -28 OUtput- 28 RF- 5
        self.BatchNorm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) #input -28 OUtput- 14 RF- 10
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1) #input -14 OUtput- 14 RF- 12
        self.BatchNorm3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1) #input -14 OUtput- 14 RF- 14
        self.BatchNorm4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) #input -7 OUtput- 7 RF- 28
        self.conv5 = nn.Conv2d(16, 16, 3) #input -7 OUtput- 5 RF- 30
        self.BatchNorm5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, 3) #input -5 OUtput- 3 RF- 32
        self.conv7 = nn.Conv2d(32, 10, 1) #input -3 OUtput- 3 RF- 32

    def forward(self, x):
        x = self.pool1(F.relu(self.BatchNorm2(self.conv2(F.relu(self.BatchNorm1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.BatchNorm4(self.conv4(F.relu(self.BatchNorm3(self.conv3(x)))))))
        x = F.relu(self.conv6(F.relu(self.BatchNorm5(self.conv5(x)))))
        x = self.conv7(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 10)
        return F.log_softmax(x)



#### ---------------------------------------------------------------------------------------
####        Layer (type)     ->          Output Shape    ->     Param       -> Receptive Layer
#### ====================================================================================
####            Conv2d-1     ->       [-1, 8, 28, 28]      ->        80     -> 3
####       BatchNorm2d-2     ->       [-1, 8, 28, 28]       ->       16     -> 3
####            Conv2d-3     ->      [-1, 16, 28, 28]       ->    1,168     -> 5
####       BatchNorm2d-4     ->      [-1, 16, 28, 28]       ->       32     -> 5
####         MaxPool2d-5     ->      [-1, 16, 14, 14]       ->        0     -> 10
####            Conv2d-6     ->      [-1, 16, 14, 14]       ->    2,320     -> 12
####       BatchNorm2d-7     ->      [-1, 16, 14, 14]       ->       32     -> 12
####            Conv2d-8     ->      [-1, 16, 14, 14]       ->    2,320     -> 14
####       BatchNorm2d-9     ->      [-1, 16, 14, 14]       ->       32     -> 14
####        MaxPool2d-10     ->        [-1, 16, 7, 7]       ->        0     -> 28
####           Conv2d-11     ->        [-1, 16, 5, 5]       ->    2,320     -> 30
####      BatchNorm2d-12     ->        [-1, 16, 5, 5]       ->       32     -> 30
####           Conv2d-13     ->        [-1, 32, 3, 3]       ->    4,640     -> 32
####           Conv2d-14     ->        [-1, 10, 3, 3]       ->      330     -> 32
#### =================================================================================
#### Total params: 13,322
#### Trainable params: 13,322
#### Non-trainable params: 0
#### ----------------------------------------------------------------
#### Input size (MB): 0.00
#### Forward/backward pass size (MB): 0.42
#### Params size (MB): 0.05
#### Estimated Total Size (MB): 0.48
#### ----------------------------------------------------------------


#### LOGS

  0%|          | 0/1875 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:27: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.13960422575473785 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.70it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0468, Accuracy: 9861/10000 (98.6100%)

loss=0.06523478031158447 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 76.29it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0384, Accuracy: 9870/10000 (98.7000%)

loss=0.005232319235801697 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 76.12it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0375, Accuracy: 9886/10000 (98.8600%)

loss=0.0036078004632145166 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.58it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0266, Accuracy: 9916/10000 (99.1600%)

loss=0.08178359270095825 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.78it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0247, Accuracy: 9915/10000 (99.1500%)

loss=0.0009511785465292633 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.65it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0319, Accuracy: 9899/10000 (98.9900%)

loss=0.005627533886581659 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.39it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0279, Accuracy: 9912/10000 (99.1200%)

loss=0.004574770573526621 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.79it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0220, Accuracy: 9932/10000 (99.3200%)

loss=0.028514552861452103 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.98it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0190, Accuracy: 9939/10000 (99.3900%)

loss=0.03974150866270065 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.33it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0232, Accuracy: 9930/10000 (99.3000%)

loss=0.0005148382042534649 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.97it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0220, Accuracy: 9937/10000 (99.3700%)

loss=0.0006665097898803651 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.10it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0183, Accuracy: 9932/10000 (99.3200%)

loss=0.002070833696052432 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.14it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0242, Accuracy: 9920/10000 (99.2000%)

loss=0.00012275406334083527 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.55it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0194, Accuracy: 9930/10000 (99.3000%)

loss=0.0001359129964839667 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.50it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9932/10000 (99.3200%)

loss=0.0002975603274535388 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.68it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0185, Accuracy: 9936/10000 (99.3600%)

loss=0.0006295429193414748 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 75.06it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0181, Accuracy: 9944/10000 (99.4400%)

loss=8.715776493772864e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 75.00it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0220, Accuracy: 9931/10000 (99.3100%)

loss=7.31154577806592e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:25<00:00, 74.28it/s]

Test set: Average loss: 0.0200, Accuracy: 9938/10000 (99.3800%)


