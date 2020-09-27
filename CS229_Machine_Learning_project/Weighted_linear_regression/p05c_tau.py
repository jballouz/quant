import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = 0
    min_MSE = np.inf
    for i in range(len(tau_values)):
        LWR = LocallyWeightedLinearRegression(tau = tau_values[i], x_eval = x_eval)
        LWR.fit(x_train, y_train)
        MSE = np.mean((y_eval - LWR.predict())**2)
        print('For tau = ' + str(tau_values[i]) + ', the MSE value is ' + str(MSE))
        if MSE < min_MSE:
            best_tau = tau_values[i]
            min_MSE = MSE
        plt.scatter(x_eval[:,1], y_eval, marker='o', color='red', label='validation')
        plt.scatter(x_train[:,1], y_train, marker='x', color='blue', label='training')
        index=np.argsort(x_eval[:,-1])
        plt.plot(x_eval[index,-1], np.reshape(LWR.predict(),(x_eval.shape[0],1))[index], color="black", linestyle='--', label='fit')
        plt.xlabel('x1')
        plt.ylabel('y')
        plt.legend()
        plt.title('Locally Weighted Regression on validation split, tau = ' + str(tau_values[i]))
        plt.show()
    # Fit a LWR model with the best tau value
    print('For the validation set, the tau value which achieves the lowest MSE = ' + str(min_MSE) +  ', is tau = ' + str(best_tau))
    # Run on the test set to get the MSE value
    LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
    LWR.fit(x_train, y_train)
    MSE = np.mean((y_test - LWR.predict())**2)
    print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
    # Save test set predictions to pred_path
    np.savetxt(pred_path,LWR.predict())
    # Plot data
    plt.scatter(x_train[:,1], y_train, marker='x', color='blue', label='training')
    plt.scatter(x_test[:,1], y_test, marker='o', color='green', label='test')
    index=np.argsort(x_test[:,-1])
    plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", linestyle='--', label='fit')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Regression on test split, tau = ' + str(best_tau))
    plt.show()
    # *** END CODE HERE ***
