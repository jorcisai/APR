import sys
sys.path.append('src')
from mnist_dataset import get_mnist
from mixgaussian import mixgaussian_classifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    classifier = mixgaussian_classifier()
    (x_train, y_train), (x_test, y_test) = get_mnist("../dat/mnist").load_data()
    traindev_size = round(x_train.shape[0]*0.9)
    x_dev = x_train[traindev_size:]
    y_dev = y_train[traindev_size:]
    x_train = x_train[:traindev_size]
    y_train = y_train[:traindev_size]

    # Mixture of gaussians with a single component for testing purposes 
    pca_components = [2, 5, 10, 20, 50, 100]
    pca_results = []
    print("Mixture of Gaussians with MNIST dataset reduced by PCA with smoothing")
    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        pca__smooth_results = []
        for pca_component in pca_components:
            pca = PCA(n_components=pca_component)
            pca.fit(x_train)
            x_train_proj = pca.transform(x_train)
            x_dev_proj = pca.transform(x_dev)
            gauss = mixgaussian_classifier()
            gauss.train(x_train_proj, y_train, x_dev_proj, y_dev, 1, alpha=alpha)
            yhat = gauss.predict(x_dev_proj)
            pca__smooth_results.append(np.mean(y_dev != yhat) * 100)
        plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on dev set with alpha={alpha} and k=1")
    ##Save plot
    plt.title("pca components vs error rate in GMM - K=1, Alpha=0.9")
    plt.legend(loc="upper left")
    plt.xticks(pca_components, rotation=70)
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Eror rate")
    plt.savefig("exp/mixgaussian_K1.png")
    # Mixture of gaussians
    plt.clf()
    pca_results = []
    print("Mixture of Gaussian with MNIST dataset reduced by PCA with smoothing")
    ks = [1, 2, 5, 10, 20, 50]
    pca__smooth_results = []
    for k in ks:
        pca = PCA(n_components=30)
        pca.fit(x_train)
        x_train_proj = pca.transform(x_train)
        x_dev_proj = pca.transform(x_dev)
        gauss = mixgaussian_classifier()
        gauss.train(x_train_proj, y_train, x_dev_proj, y_dev, k, alpha=0.9)
        yhat = gauss.predict(x_dev_proj)
        pca__smooth_results.append(np.mean(y_dev != yhat) * 100)
    plt.plot(ks, pca__smooth_results, marker="o", label=f"Error rate on dev set with alpha={alpha} and k={k}")
    ##Save plot
    plt.title("Number of mixtures vs error rate in mixture of gaussians - Alpha=0.9 - PCA to 30dim")
    plt.legend(loc="upper left")
    plt.xticks(ks, rotation=70)
    plt.xlabel("Number of mixtures")
    plt.ylabel("Eror rate")
    plt.savefig("exp/mixgaussian.png")


if __name__ == "__main__":
    main()

