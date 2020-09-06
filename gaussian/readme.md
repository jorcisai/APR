# Gaussian classifier

A python implementation of the Gaussian classifier under the src directory.

![](exp/gaussian.jpg)

## Usage

Gaussian classifier
```python
from gaussian_classifier import gaussian_classifier

(x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
gauss = gaussian_classifier()
gauss.train(x_train,y_train,alpha=1.0) #Alpha value for smoothing
yhat = gauss.predict(x_test)
yhat!=y_test ## Error rate
```
## Examples
You can check the example on they work with the MNIST dataset on gaussian-exp.py under the scr directory.

Results from gaussian-exp.py

|             Unsmoothed              |              Smoothed               |
| :---------------------------------: | :---------------------------------: |
| ![](exp/unsmooth_gaussian.png) | ![](exp/smooth_gaussian.png) |

