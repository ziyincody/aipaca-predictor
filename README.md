# Aipaca-predictor

A very fast library to predict Keras Neural Nets training time on GPU.

Currently it supports
```
1080Ti, K40, K80, M60, P100, T4, V100
```

This early prototype is able to achieve 20% margin of error for neural nets in Colab.

## Install

```
pip install aipaca_predictor==0.0.8
```

## Usage

```python
from aipaca_predictor.predictor import to_predict

feed_forward = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dropout(0.5),
  Dense(64, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax'),
])

to_predict(
    model=feed_forward,
    batch_size=2,
    iterations=60000//2,
    optimizer='sgd'
)
```

For a colab example feel free to check out this
[link](https://colab.research.google.com/drive/1bCl68Usp_yri9j3eIKun297PdAnns0th?usp=sharing).
