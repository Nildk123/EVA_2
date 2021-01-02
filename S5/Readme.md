1. Target - Reach 99.4% accuracy in Validation within 15epochs using 10k params
2. Experimented on regularizer, augmentation, learning rate, and scheduler to reach the desired result
3. Experimented with Different architectures
4. Files are in order of experiments - EVA4S5F9_experiment1.ipynb < EVA4S5F9_experiment2.ipynb < EVA4S5F9_experiment3.ipynb
5. EVA4S5F9_Final.ipynb is the final notebook that can be used for evaluation

#Final Model:

```python
import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 26 ,Reseptive Field 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            #nn.Dropout(dropout_value)
        ) # output_size = 24 ,Reseptive Field 5

        # TRANSITION BLOCK 1

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12 ,Reseptive Field 6

        ## RES block 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        ) # output_size = 12 ,Reseptive Field 6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 10,Reseptive Field 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        ) # output_size = 10, ,Reseptive Field 10

        #RES Block 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 10 ,Reseptive Field 10

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 8,Reseptive Field 14

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(dropout_value)
        ) # output_size = 8 ,Reseptive Field 14

        #RES Block 3
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 8,Reseptive Field 14

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            #nn.Dropout(dropout_value)
        ) # output_size = 6,Reseptive Field 18

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            #nn.Dropout(dropout_value)
        ) # output_size = 4,Reseptive Field 22
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1, Reseptive Field 28

        self.convblock12 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 1, Reseptive Field 28


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        #x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.gap(x)        
        x = self.convblock12(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

```

#Final Parameter 

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
            Conv2d-4           [-1, 16, 24, 24]           1,440
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 10, 24, 24]             160
       BatchNorm2d-8           [-1, 10, 24, 24]              20
              ReLU-9           [-1, 10, 24, 24]               0
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
           Conv2d-14           [-1, 32, 10, 10]             320
      BatchNorm2d-15           [-1, 32, 10, 10]              64
             ReLU-16           [-1, 32, 10, 10]               0
           Conv2d-17           [-1, 10, 10, 10]             320
             ReLU-18           [-1, 10, 10, 10]               0
      BatchNorm2d-19           [-1, 10, 10, 10]              20
           Conv2d-20             [-1, 10, 8, 8]             900
             ReLU-21             [-1, 10, 8, 8]               0
      BatchNorm2d-22             [-1, 10, 8, 8]              20
           Conv2d-23             [-1, 32, 8, 8]             320
      BatchNorm2d-24             [-1, 32, 8, 8]              64
             ReLU-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 10, 8, 8]             320
             ReLU-27             [-1, 10, 8, 8]               0
      BatchNorm2d-28             [-1, 10, 8, 8]              20
           Conv2d-29             [-1, 14, 6, 6]           1,260
             ReLU-30             [-1, 14, 6, 6]               0
      BatchNorm2d-31             [-1, 14, 6, 6]              28
           Conv2d-32             [-1, 16, 4, 4]           2,016
             ReLU-33             [-1, 16, 4, 4]               0
      BatchNorm2d-34             [-1, 16, 4, 4]              32
        AvgPool2d-35             [-1, 16, 1, 1]               0
           Conv2d-36             [-1, 10, 1, 1]             160
================================================================
Total params: 8,546
Trainable params: 8,546
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.72
Params size (MB): 0.03
Estimated Total Size (MB): 0.76
----------------------------------------------------------------
```

#Model Logs

```python
  0%|          | 0/469 [00:00<?, ?it/s]EPOCH: 0
Loss=0.11214590072631836 Batch_id=468 Accuracy=88.32: 100%|██████████| 469/469 [00:43<00:00, 10.90it/s]
current Learing Rate:  0.01
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0665, Accuracy: 9835/10000 (98.35%)

EPOCH: 1
Loss=0.09433918446302414 Batch_id=468 Accuracy=97.74: 100%|██████████| 469/469 [00:42<00:00, 10.91it/s]current Learing Rate:  0.01

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0595, Accuracy: 9829/10000 (98.29%)

EPOCH: 2
Loss=0.05425375699996948 Batch_id=468 Accuracy=98.27: 100%|██████████| 469/469 [00:43<00:00, 10.85it/s]current Learing Rate:  0.01

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0402, Accuracy: 9877/10000 (98.77%)

EPOCH: 3
Loss=0.011051923967897892 Batch_id=468 Accuracy=98.46: 100%|██████████| 469/469 [00:42<00:00, 11.03it/s]current Learing Rate:  0.01

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0519, Accuracy: 9841/10000 (98.41%)

EPOCH: 4
Loss=0.00902416929602623 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:43<00:00, 10.85it/s]current Learing Rate:  0.01

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0290, Accuracy: 9920/10000 (99.20%)

EPOCH: 5
Loss=0.02549041621387005 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:42<00:00, 11.04it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0303, Accuracy: 9896/10000 (98.96%)

EPOCH: 6
Loss=0.03081030212342739 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:42<00:00, 10.98it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0238, Accuracy: 9926/10000 (99.26%)

EPOCH: 7
Loss=0.08570990711450577 Batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:43<00:00, 10.82it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0212, Accuracy: 9937/10000 (99.37%)

EPOCH: 8
Loss=0.008049164898693562 Batch_id=468 Accuracy=99.09: 100%|██████████| 469/469 [00:42<00:00, 10.91it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0248, Accuracy: 9924/10000 (99.24%)

EPOCH: 9
Loss=0.05069735646247864 Batch_id=468 Accuracy=99.06: 100%|██████████| 469/469 [00:42<00:00, 10.98it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0218, Accuracy: 9939/10000 (99.39%)

EPOCH: 10
Loss=0.04626016691327095 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:43<00:00, 10.90it/s]current Learing Rate:  0.005

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

EPOCH: 11
Loss=0.017562782391905785 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:43<00:00, 10.88it/s]current Learing Rate:  0.0025

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0202, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.00921888928860426 Batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:42<00:00, 11.04it/s]current Learing Rate:  0.0025

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.06074872240424156 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:43<00:00, 10.88it/s]current Learing Rate:  0.0025

  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0199, Accuracy: 9937/10000 (99.37%)

EPOCH: 14
Loss=0.025518519803881645 Batch_id=468 Accuracy=99.31: 100%|██████████| 469/469 [00:43<00:00, 10.73it/s]current Learing Rate:  0.0025


Test set: Average loss: 0.0182, Accuracy: 9940/10000 (99.40%)

```
