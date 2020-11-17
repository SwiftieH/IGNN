import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Function
from utils import sparse_mx_to_torch_sparse_tensor



class ImplicitFunction(Function):
    #ImplicitFunction.apply(input, A, U, self.X_0, self.W, self.Omega_1, self.Omega_2)
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):

        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        grad_z, err, status, _ = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, trasposed_A=True)
        #grad_z.clamp_(-1,1)

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        #X = B if X is None else X

        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi