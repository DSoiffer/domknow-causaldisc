import numpy as np
import torch
from scipy.special import expit as sigmoid

def _as_lambda(lambda1, W):
    """lambda1 may be a scalar or a (p, p) per-edge weight matrix (negative
    entries encourage an edge). For torch, arrays are moved to W's device/dtype."""
    if isinstance(W, torch.Tensor) and not np.isscalar(lambda1):
        return torch.as_tensor(lambda1, dtype=W.dtype, device=W.device)
    return lambda1


class reg_fn:
    @staticmethod
    def user_reg(W, **kwargs):
        """
        This is a user-defined regularization function. Users can define their own regularization function by customizing this function.

        Parameters
        ----------
        W : numpy array or torch tensor
            Weight matrix.

        Returns
        -------
        reg_val : float
            Value of the regularization function.
        reg_grad : numpy array or torch tensor
            Gradient of the regularization function.

        Important
        ---------
        If users work with **numpy** as the computation library, the output of the ``user_reg`` function should be 
        (value of regularization, gradient of regularization) in numpy arrays.
        If users work with **torch** as the computation library, the output of the ``user_reg`` function should be 
        the value of regularization in a torch tensor.
        """
        
        if isinstance(W, np.ndarray):
            pass
        elif isinstance(W, torch.Tensor):
            pass
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        raise NotImplementedError("User-defined regularization is not implemented yet. User are free to define their own regularization function")


    @staticmethod
    def no_reg(W, **kwargs):
        
        if isinstance(W, np.ndarray):
            return 0, np.zeros_like(W)
        elif isinstance(W, torch.Tensor):
            return 0
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def l1_reg(W,**kwargs):
        
        lambda1 = _as_lambda(kwargs.get('lambda1',0.1), W)
        if isinstance(W, np.ndarray):
            return (lambda1*np.abs(W)).sum(), lambda1*np.sign(W)
        elif isinstance(W, torch.Tensor):
            return (lambda1*torch.abs(W)).sum()
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def l2_reg(W,**kwargs):
        
        lambda1 = _as_lambda(kwargs.get('lambda1',0.1), W)
        if isinstance(W, np.ndarray):
            return (lambda1/2 * W ** 2).sum(), lambda1*W
        elif isinstance(W, torch.Tensor):
            return torch.sum(lambda1/2 * W**2)
        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def mcp_reg(W, **kwargs):

        lambda1 = _as_lambda(kwargs.get('lambda1', 0.1), W)
        gamma = kwargs.get('gamma',1)
        if isinstance(W, np.ndarray):
            cond = np.abs(W) <= gamma 
            reg_val = lambda1* np.where(cond, np.abs(W) - W**2/(2*gamma) , gamma/2)
            reg_grad = lambda1 * np.where(cond, np.sign(W) -W / gamma, 0)
            return reg_val.sum(), reg_grad
        elif isinstance(W, torch.Tensor):
            cond = torch.abs(W) <= gamma
            reg_val = lambda1* torch.where(cond, torch.abs(W) - W**2/(2*gamma) , gamma/2)
            reg_grad = lambda1 * torch.where(cond, torch.sign(W) -W / gamma, 0)
            return reg_val.sum()
        else:
            raise ValueError("W must be either numpy array or torch tensor")

class loss_fn:
    @staticmethod
    def user_loss(W,X,**kwargs):
        """
        This is a user-defined loss function for **linear** model. Users can define their own loss function by customizing this function.

        Parameters
        ----------
        W : numpy array or torch tensor
            Weight matrix.
        X : numpy array or torch tensor
            Data matrix.

        Returns
        -------
        loss : float
            Value of the loss function.
        G_loss : numpy array or torch tensor
            Gradient of the loss function.

        Important
        ---------
        If users work with **numpy** as the computation library, the output of the ``user_loss`` function should be 
        (value of loss, gradient of loss) in numpy arrays.
        If users work with **torch** as the computation library, the output of the ``user_loss`` function should be 
        the value of the loss in a torch tensor.

        """
        
        if isinstance(W, np.ndarray):
            pass
        elif isinstance(W, torch.Tensor):
            pass
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        raise NotImplementedError("User-defined loss is not implemented yet. User are free to define their own loss function")



    @staticmethod
    def l2_loss(W, X, **kwargs):
        
        if isinstance(W, np.ndarray):
            M = X @ W
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            M = X.matmul(W)
            R = X - M
            loss = 0.5 / X.shape[0] * torch.sum(R ** 2)
            return loss
        else:
            raise ValueError("W must be either numpy array or torch tensor")
    @staticmethod
    def logistic_loss(W,X,**kwargs):
        r"""
        Calculate the logistic loss, and correpsonding gradient of the linear model
        
        Structural Equation Model: x_i = Bernoulli(sigmoid(\sum_j w_ji x_j))

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)

        Returns:
        loss: float
        gradient: ndarray, shape (n_nodes, n_nodes)
        """
        if isinstance(W, np.ndarray):
            M = X @ W
            loss = 1.0 / X.shape[0] * (np.logaddexp(0,M)-X * M).sum()
            G_loss =  1.0 / X.shape[0] * X.T @ (sigmoid(M)- X)
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            M = X.matmul(W)
            loss = 1.0 / X.shape[0] * torch.sum(torch.logaddexp(torch.zeros_like(M), M) - X * M)
            return loss
        else:
            raise ValueError("W must be either numpy array or torch tensor")
    
    @staticmethod
    def logll_loss(W,X,**kwargs): 
        r"""
        Calculate the loglikelihood loss, and correpsonding gradient of the linear model
        
        Structural Equation Model: x_i = (\sum_j w_ji x_j) + \epsilon where \epsilon is normal distribution

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            mean = np.mean(X, axis=0)
            X = X - mean
            n, d = X.shape
            Sigma = 1/n * X.T@X
            I = np.eye(d)
            diag_elements = np.diag((W-I).transpose()@Sigma@ (W-I))
            loss = 1/2* np.sum(np.log(diag_elements))
            G_loss = Sigma@(W-I)@np.diag(1/diag_elements)
            return loss, G_loss
        elif isinstance(W, torch.Tensor):
            n, d = X.shape
            X = X - torch.mean(X, dim=0)
            Sigma = 1/n * X.T@X
            I = torch.eye(d, device=W.device, dtype=W.dtype)
            diag_elements = torch.diag((W-I).transpose(0,1)@Sigma@ (W-I))
            loss = 1/2* torch.sum(torch.log(diag_elements))
            return loss
            
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        
    @staticmethod
    def logdetll_ev_loss(W, X, **kwargs):
        """
        Calculate the logdet loglikelihood loss assuming equal noise variance

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            raise ValueError("logdetll_ev_loss is not implemented for numpy")
        elif isinstance(W, torch.Tensor):
            # Floor the pooled residual variance so a (near-)deterministic node
            # can't drive the total RSS -> 0 and log(RSS) -> -inf. var_floor is a
            # minimum noise variance; scaled by n*p to match the pooled (summed
            # over all entries) RSS scale used here. Set via
            # general_options["var_floor"] (allow-listed in configure.py); 0
            # (default) leaves the loss unchanged.
            var_floor = kwargs.get("var_floor", 0)
            rss = torch.square(torch.linalg.matrix_norm(X - X @ W)) + var_floor * X.shape[0] * X.shape[1]
            return (
                0.5
                * W.shape[0]
                * torch.log(rss)
                - torch.linalg.slogdet(torch.eye(W.shape[1], device=W.device, dtype=W.dtype) - W)[1]
            )

        else:
            raise ValueError("W must be either numpy array or torch tensor")

    @staticmethod
    def logdetll_nv_loss(W, X, **kwargs):
        """
        Calculate the logdet loglikelihood loss assuming non-equal noise variance

        Args:
        W: ndarray, shape (n_nodes,n_nodes)
        X: ndarray, shape (n_samples, n_nodes)
        """
        if isinstance(W, np.ndarray):
            raise ValueError("logdetll_nv_loss is not implemented for numpy")
        elif isinstance(W, torch.Tensor):
            # Floor each node's residual variance so a (near-)deterministic node
            # can't drive its RSS -> 0 and log(RSS) -> -inf (which sends the
            # weights to +/-inf and overflows the acyclicity term). var_floor is
            # a minimum noise variance; scaled by n to match the RSS (summed,
            # not averaged) scale used here. Set via general_options["var_floor"]
            # (allow-listed in configure.py); 0 (default) leaves the loss unchanged.
            var_floor = kwargs.get("var_floor", 0)
            rss = torch.sum(torch.square(X - X @ W), axis=0) + var_floor * X.shape[0]
            return (
                0.5 * torch.sum(torch.log(rss))
                - torch.linalg.slogdet(torch.eye(W.shape[1], device=W.device, dtype=W.dtype) - W)[1]
            )

        else:
            raise ValueError("W must be either numpy array or torch tensor")


def given_hinge_requested(**kwargs) -> bool:
    """Whether the given-edge magnitude-floor hinge is configured/active: a mask
    plus a nonzero margin and weight (the single 'enabled' check, shared by the
    penalty and by the numpy-path guards that raise NotImplementedError)."""
    return (kwargs.get('given_hinge_mask') is not None
            and bool(kwargs.get('given_hinge_margin', 0))
            and bool(kwargs.get('given_hinge_weight', 0)))


def given_hinge_penalty(W, X, **kwargs):
    """Magnitude-floor hinge encouraging the *given* (domain-knowledge) edges to
    exist:  ``rho * sum_{(u,v) given} max(0, m - |W[u,v]|)**2``.

    Pushes each given edge's magnitude up toward the margin ``m`` (~ the
    detection threshold) and then stops: it contributes nothing once
    ``|W[u,v]| >= m`` (the data sets the magnitude above the floor), and it does
    not prescribe a value or a sign -- it acts on ``|W|``, so it encourages the
    edge to *exist* without forcing whether the effect is +/-. The edge
    *direction* is fixed purely by which entries the mask marks (use it with
    ``lambda1 = 0`` on those entries so the sparsity penalty doesn't fight it).

    This is the soft penalty-method counterpart of the inequality constraint
    ``|W[u,v]| >= m`` (with finite ``rho`` it only approaches the floor). When a
    ``GivenHingeAL`` state is supplied via ``general_options['_given_al']`` (set
    by ``make_given_hinge_al`` on the torch paths when ``given_hinge_al=True``),
    this dispatches to the augmented-Lagrangian form instead, which reaches the
    floor exactly via a per-edge dual variable updated once per outer iteration.

    Controlled through ``general_options`` (no-op unless enabled):
      - ``given_hinge_mask``   : bool (p, p), True on the given edges.
      - ``given_hinge_margin`` : m, the magnitude floor (default 0 = off).
      - ``given_hinge_weight`` : rho, the strength (default 0 = off).
    Returns a torch scalar (torch-only)."""
    W = torch.as_tensor(W)
    al = kwargs.get('_given_al')
    if al is not None:
        return al.penalty(W)
    if not given_hinge_requested(**kwargs):
        return W.new_zeros(())
    mask = kwargs['given_hinge_mask']
    m = kwargs['given_hinge_margin']
    rho = kwargs['given_hinge_weight']
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=W.device)
    shortfall = torch.clamp(m - torch.abs(W), min=0.0)          # m - |W|, floored at 0
    shortfall = torch.where(mask_t, shortfall, torch.zeros_like(shortfall))
    return rho * torch.sum(shortfall ** 2)


class GivenHingeAL:
    """Augmented-Lagrangian state for the given-edge magnitude floor ``|W| >= m``.

    The fixed-weight hinge is the ``lambda = 0`` special case of the PHR
    (Powell-Hestenes-Rockafellar) augmented Lagrangian for the inequality
    ``g(W) = m - |W| <= 0``. Here a per-edge dual variable ``lam`` is accumulated
    across outer (dual-ascent) iterations so the floor is reached exactly with a
    finite penalty ``rho``, instead of only being approached.

    - :meth:`penalty` is the term added to the inner objective (differentiable).
    - :meth:`update` performs the dual ascent ``lam <- max(0, lam + rho*(m-|W|))``
      and is called once per outer iteration with the current adjacency.

    ``lam``/``mask`` are built lazily on first use to match the device/dtype of
    the adjacency tensor."""

    def __init__(self, mask, margin, rho):
        self._mask_src = mask
        self.m = float(margin)
        self.rho = float(rho)
        self.mask = None
        self.lam = None

    def _ensure(self, W):
        if self.lam is None:
            self.mask = torch.as_tensor(self._mask_src, dtype=torch.bool, device=W.device)
            self.lam = torch.zeros(self.mask.shape, dtype=W.dtype, device=W.device)

    def penalty(self, W):
        self._ensure(W)
        t = torch.clamp(self.lam + self.rho * (self.m - torch.abs(W)), min=0.0)
        t = torch.where(self.mask, t, torch.zeros_like(t))
        return torch.sum(t ** 2) / (2.0 * self.rho)

    @torch.no_grad()
    def update(self, W):
        W = torch.as_tensor(W)
        self._ensure(W)
        t = torch.clamp(self.lam + self.rho * (self.m - torch.abs(W)), min=0.0)
        self.lam = torch.where(self.mask, t, torch.zeros_like(t))


def make_given_hinge_al(**kwargs):
    """Return a :class:`GivenHingeAL` when the augmented-Lagrangian given-edge
    floor is requested (hinge configured *and* ``given_hinge_al`` truthy), else
    ``None`` so callers fall back to the fixed-weight hinge / no-op."""
    if not (given_hinge_requested(**kwargs) and kwargs.get('given_hinge_al', False)):
        return None
    return GivenHingeAL(kwargs['given_hinge_mask'],
                        kwargs['given_hinge_margin'],
                        kwargs['given_hinge_weight'])


class nl_loss_fn:
    @staticmethod
    def user_nl_loss(output, target, **kwargs):
        """
        This is a user-defined loss function for **nonlinear** model. Users can define their own loss function by customizing this function.

        Parameters
        ----------
        output : torch tensor
            Output of the model.
        target : torch tensor
            Target value.

        Returns
        -------
        loss : torch tensor
            Value of the loss function.

        Important
        ---------
        The output of the ``user_nl_loss`` function should be the value of the loss in a torch tensor.
        """


        raise NotImplementedError("User defined loss is not implemented yet. User are free to define their own loss function.")
    
    @staticmethod
    def nl_l2_loss(output, target, **kwargs):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    @staticmethod
    def nl_logistic_loss(output,target, **kwargs):
        import torch.nn as nn
        bce_loss = nn.BCELoss()
        return bce_loss(output, target)
        
    
    @staticmethod
    def nl_logll_loss(output, target, **kwargs):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1/n* torch.sum((output - target)**2))
        return loss
