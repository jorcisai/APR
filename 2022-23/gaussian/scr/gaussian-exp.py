import sys
sys.path.append('src')
from mnist_dataset import get_mnist
from gaussian import gaussian_classifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def main():
    (x_train, y_train), (x_test, y_test) = get_mnist("../dat/mnist").load_data()
    traindev_size = round(x_train.shape[0]*0.9)
    x_dev = x_train[traindev_size:]
    y_dev = y_train[traindev_size:]
    x_train = x_train[:traindev_size]
    y_train = y_train[:traindev_size]

    # Unsmooth gaussian classifier
    print("Gaussian classifier with mnist dataset reduced by PCA")
    pca_components = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    pca_results = []
    for pca_component in pca_components:
        pca = PCA(n_components=pca_component)
        pca.fit(x_train)
        x_train_proj = pca.transform(x_train)
        x_dev_proj = pca.transform(x_dev)
        gauss = gaussian_classifier()
        gauss.train(x_train_proj, y_train)
        yhat = gauss.predict(x_dev_proj)
        pca_results.append(np.mean(y_dev != yhat) * 100)
    ##Save plot
    fig, ax = plt.subplots()
    plt.title("pca components vs error rate in gaussian classifier")
    plt.plot(pca_components, pca_results, label="Error rate on dev set alpha=1.0")
    plt.legend(loc="upper left")
    plt.xticks(pca_components, rotation=70)
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Eror rate")
    plt.savefig("exp/unsmoothed_gaussian.png")
    # Smoothed gaussian classifier
    print("Gaussian classifier with MNIST dataset reduced by PCA with smoothing")
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.9]
    plt.title("pca components vs error rate in gaussian classifier with smoothing")
    plt.plot(pca_components, pca_results)
    for alpha in alphas:
        pca__smooth_results = []
        for pca_component in pca_components:
            pca = PCA(n_components=pca_component)
            pca.fit(x_train)
            x_train_proj = pca.transform(x_train)
            x_dev_proj = pca.transform(x_dev)
            gauss = gaussian_classifier()
            gauss.train(x_train_proj, y_train, alpha=alpha)
            yhat = gauss.predict(x_dev_proj)
            pca__smooth_results.append(np.mean(y_dev != yhat) * 100)
        plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on dev set alpha={alpha}")
    ##Save plot
    plt.legend(loc="upper left")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Error rate")
    plt.xticks(pca_components, rotation=70)
    plt.savefig("exp/smoothed_gaussian.png")


if __name__ == "__main__":
    main()
