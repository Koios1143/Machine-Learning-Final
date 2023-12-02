#### 1. linear layer 0
```
self.lin0 = nn.Sequential(
    nn.Linear(in_dim, h, bias=False),
    nn.LayerNorm(h),
    nn.SiLU(inplace=True),
    nn.Dropout(0.5),
)
```
- nn.Sequential: applies layers in passed order  
- nn.Linear: linearly transform input (features)   
- nn.LayerNorm: normalize input for better convergence  
- nn.SiLU: activation function of choice  
- **nn.Dropout**: zeroes nodes with specified probability to 
prevent co-adaption (overfitting)  
  
---
#### 2. multilayer perceptron
similar to step 1., but multiple pass

---
#### 3. linear layer 1
once again reshape input, then pass through GroupNorm of 1 group (practically LayerNorm)

---
#### 4. upsampler
supposedly transforms to 4(output channels) * 64 * 64  
// TODO

---
#### 5. MyDataset
custom object to link fmri voxels to stimulus images  
images are resized to fit model requirement

---
#### 6. loading dataset
- concatenate lh, rh (dimension: (39548, ))  
- using pytorch lib:  
    - split dataset randomly into training and validation
    - use dataloader to optimize for parallel processing
    - use scheduler to dynamically adjust learning rate with specified weight decay to prevent overfitting

---
#### 7. training
