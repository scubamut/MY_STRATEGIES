from . import backtest, compute_weights_PMA, compute_weights_RS_DM, endpoints
from . import monthly_return_table, portfolio_helper_functions
import pandas as pd
from cvxopt import spdiag, solvers, matrix
import numpy as np

# Portfolio Helper Functions

# Functions:
#    1. compute_efficient_portfolio        compute minimum variance portfolio
#                                            subject to target return
#    2. compute_global_min_portfolio       compute global minimum variance portfolio
#    3. compute_tangency_portfolio         compute tangency portfolio
#    4. compute_efficient_frontier         compute Markowitz bullet
#    5. compute_portfolio_mu               compute portfolio expected return
#    6. compute_portfolio_sigma            compute portfolio standard deviation
#    7. compute_covariance_matrix          compute covariance matrix
#    8. compute_expected_returns           compute expected returns vector

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_covariance_matrix(prices):
    # calculates the cov matrix for the period defined by prices
    returns = np.log(1 + prices.pct_change())[1:]
    excess_returns_matrix = returns - returns.mean()
    return 1. / len(returns) * (excess_returns_matrix.T).dot(excess_returns_matrix)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_expected_returns(prices):
    mu_vec = np.log(1 + prices.pct_change(1))[1:].mean()
    return mu_vec


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_portfolio_mu(mu_vec, weights_vec):
    if len(mu_vec) != len(weights_vec):
        raise RuntimeError('mu_vec and weights_vec must have same length')
    return mu_vec.T.dot(weights_vec)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_portfolio_sigma(sigma_mat, weights_vec):
    if len(sigma_mat) != len(sigma_mat.columns):
        raise RuntimeError('sigma_mat must be square\nlen(sigma_mat) = {}\nlen(sigma_mat.columns) ={}'.
                           format(len(sigma_mat), len(sigma_mat.columns)))
    return np.sqrt(weights_vec.T.dot(sigma_mat).dot(weights_vec))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_efficient_portfolio(mu_vec, sigma_mat, target_return, shorts=True):
    # compute minimum variance portfolio subject to target return
    #
    # inputs:
    # mu_vec                  N x 1 DataFrame expected returns
    #                         with index = asset names
    # sigma_mat               N x N DataFrame covariance matrix of returns
    #                         with index = columns = asset names
    # target_return           scalar, target expected return
    # shorts                  logical, allow shorts is TRUE
    #
    # output is portfolio object with the following elements
    #
    # mu_p                   portfolio expected return
    # sig_p                  portfolio standard deviation
    # weights    Ü            with index = asset names

    # check for valid inputs
    #

    if len(mu_vec) != len(sigma_mat):
        print("dimensions of mu_vec and sigma_mat do not match")
        raise
    if np.matrix([sigma_mat.ix[i][i] for i in range(len(sigma_mat))]).any() <= 0:
        print('Covariance matrix not positive definite')
        raise

    #
    # compute efficient portfolio
    #

    solvers.options['show_progress'] = False
    P = 2 * matrix(sigma_mat.values)
    q = matrix(0., (len(sigma_mat), 1))
    G = spdiag([-1. for i in range(len(sigma_mat))])
    A = matrix(1., (1, len(sigma_mat)))
    A = matrix([A, matrix(mu_vec.T.values).T], (2, len(sigma_mat)))
    b = matrix([1.0, target_return], (2, 1))

    if shorts == True:
        h = matrix(1., (len(sigma_mat), 1))

    else:
        h = matrix(0., (len(sigma_mat), 1))

    # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
    #                                     sigma_mat.columns)
    weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)

    #
    # compute portfolio expected returns and variance
    #
    # print ('*** Debug ***\n_compute_efficient_portfolio:\nmu_vec:\n', self.mu_vec, '\nsigma_mat:\n',
    #        self.sigma_mat, '\nweights:\n', self.weights_vec )
    weights_vec.index = mu_vec.index
    mu_p = compute_portfolio_mu(mu_vec, weights_vec)
    sigma_p = compute_portfolio_sigma(sigma_mat, weights_vec)

    return weights_vec, mu_p, sigma_p


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_global_min_portfolio(mu_vec, sigma_mat, shorts=True):
    solvers.options['show_progress'] = False
    P = 2 * matrix(sigma_mat.values)
    q = matrix(0., (len(sigma_mat), 1))
    G = spdiag([-1. for i in range(len(sigma_mat))])
    A = matrix(1., (1, len(sigma_mat)))
    b = matrix(1.0)

    if shorts == True:
        h = matrix(1., (len(sigma_mat), 1))
    else:
        h = matrix(0., (len(sigma_mat), 1))

    # print ('\nP\n\n{}\n\nq\n\n{}\n\nG\n\n{}\n\nh\n\n{}\n\nA\n\n{}\n\nb\n\n{}\n\n'.format(P,q,G,h,A,b))
    # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
    #                                     index=sigma_mat.columns)
    weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)

    #
    # compute portfolio expected returns and variance
    #
    # print ('*** Debug ***\n_Global Min Portfolio:\nmu_vec:\n', mu_vec, '\nsigma_mat:\n',
    #        sigma_mat, '\nweights:\n', weights_vec)

    mu_p = compute_portfolio_mu(mu_vec, weights_vec)
    sigma_p = compute_portfolio_sigma(sigma_mat, weights_vec)

    return weights_vec, mu_p, sigma_p


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_efficient_frontier(mu_vec, sigma_mat, risk_free=0, points=100, shorts=True):
    efficient_frontier = pd.DataFrame(index=range(points), dtype=object, columns=['mu_p', 'sig_p', 'sr_p', 'wts_p'])

    gmin_wts, gmin_mu, gmin_sigma = compute_global_min_portfolio(mu_vec, sigma_mat, shorts=shorts)

    xmax = mu_vec.max()
    if shorts == True:
        xmax = 2 * mu_vec.max()
    for i, mu in enumerate(np.linspace(gmin_mu, xmax, points)):
        w_vec, portfolio_mu, portfolio_sigma = compute_efficient_portfolio(mu_vec, sigma_mat, mu, shorts=shorts)
        efficient_frontier.ix[i]['mu_p'] = w_vec.dot(mu_vec)
        efficient_frontier.ix[i]['sig_p'] = np.sqrt(w_vec.T.dot(sigma_mat.dot(w_vec)))
        efficient_frontier.ix[i]['sr_p'] = (efficient_frontier.ix[i]['mu_p'] - risk_free) / efficient_frontier.ix[i][
            'sig_p']
        efficient_frontier.ix[i]['wts_p'] = w_vec

    return efficient_frontier


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_tangency_portfolio(mu_vec, sigma_mat, risk_free=0, shorts=True):
    efficient_frontier = compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
    index = efficient_frontier.index[efficient_frontier['sr_p'] == efficient_frontier['sr_p'].max()]

    wts = efficient_frontier['wts_p'][index].values[0]
    mu_p = efficient_frontier['mu_p'][index].values[0]
    sigma_p = efficient_frontier['sig_p'][index].values[0]
    sharpe_p = efficient_frontier['sr_p'][index].values[0]

    return wts, mu_p, sigma_p, sharpe_p


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_target_risk_portfolio(mu_vec, sigma_mat, target_risk, risk_free=0, shorts=True):
    efficient_frontier = compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
    if efficient_frontier['sig_p'].max() <= target_risk:
        print('TARGET_RISK {} > EFFICIENT FRONTIER MAXIMUM {}; SETTING IT TO MAXIMUM'.
                 format(target_risk, efficient_frontier['sig_p'].max()))
        index = len(efficient_frontier) - 1
    elif efficient_frontier['sig_p'].min() >= target_risk:
        print('TARGET RISK {} < GLOBAL MINIMUM {}; SETTING IT TO GLOBAL MINIMUM'.
                 format(target_risk, efficient_frontier['sig_p'].max()))
        index = 1
    else:
        index = efficient_frontier.index[efficient_frontier['sig_p'] >= target_risk][0]

    wts = efficient_frontier['wts_p'][index]
    mu_p = efficient_frontier['mu_p'][index]
    sigma_p = efficient_frontier['sig_p'][index]
    sharpe_p = efficient_frontier['sr_p'][index]

    return wts, mu_p, sigma_p, sharpe_p