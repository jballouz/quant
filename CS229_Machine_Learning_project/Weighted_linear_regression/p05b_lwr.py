import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    # *** START CODE HERE ***
    # Fit a LWR model
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    LWR = LocallyWeightedLinearRegression(tau = 0.5, x_eval = x_eval)
    LWR.fit(x_train, y_train)
    # Get MSE value on the validation set
    MSE = np.mean((y_eval - LWR.predict())**2)
    print('the MSE value is ' + str(MSE))
    # Plot validation predictions on top of training set
    plt.scatter(x_eval[:,1], y_eval, marker='o', color='red', label='validation')
    plt.scatter(x_train[:,1], y_train, marker='x', color='blue', label='training')
    index=np.argsort(x_eval[:,-1])
    plt.plot(x_eval[index,-1], np.reshape(LWR.predict(),(x_eval.shape[0],1))[index], color="black", linestyle='--', label='fit')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Regression, tau = ' + str(tau))
    plt.show()
    # No need to save anything
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau, x_eval):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x_eval = x_eval
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        m_eval = self.x_eval.shape[0]
        n=self.x_eval.shape[1]
        m=x.shape[0]
        self.theta=np.zeros((n,m_eval))
        for i in range(m_eval):
            #print(1/2*np.exp(-np.linalg.norm(x-self.x_eval[[i],:],axis=1)**2/(2*self.tau**2)))
            w=np.diag(1/2*np.exp(-np.linalg.norm(x-self.x_eval[[i],:],axis=1)**2/(365**2*self.tau**2)))
            self.theta[:,[i]] = np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(np.reshape(y,(m,1)))
        return self.theta
        # *** END CODE HERE ***

    def predict(self):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.diagonal(np.dot(self.x_eval,self.theta))
        # *** END CODE HERE ***
