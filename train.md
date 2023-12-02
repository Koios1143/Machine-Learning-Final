1. linear layer 0
```
self.lin0 = nn.Sequential(
    nn.Linear(in_dim, h, bias=False),
    nn.LayerNorm(h),
    nn.SiLU(inplace=True),
    nn.Dropout(0.5),
)
```
a) nn.Sequential: applies layers in passed order  
b) nn.Linear: linearly transform input (features)   
c) nn.LayerNorm: normalize input for better convergence  
d) nn.SiLU: activation function of choice  
e) **nn.Dropout**: zeroes nodes with specified probability to 
prevent co-adaption (overfitting)  
  
---
2. multilayer perceptron
```
self.mlp = nn.ModuleList([
    nn.Sequential(
        nn.Linear(h, h, bias=False),
        nn.LayerNorm(h),
        nn.SiLU(inplace=True),
        nn.Dropout(0.25),
    ) for _ in range(n_blocks)
])
```
similar to step 1., but multiple pass

---
3. linear layer 1
```
self.lin1 = nn.Linear(h, 16384, bias=False)
self.norm = nn.GroupNorm(1, 64)
```
once again reshape input, then pass through GroupNorm of 1 group (practically LayerNorm)


