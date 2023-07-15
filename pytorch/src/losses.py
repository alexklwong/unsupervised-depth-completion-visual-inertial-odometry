'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf

@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
import torch


def color_consistency_loss_func(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the color consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        reduce_loss : bool
            Whether to reduce loss over height and weight dimensions
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean absolute difference between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : absolute difference between source and target images, N x 1 x H x W
    '''

    if w is None:
        w = torch.ones_like(src)

    loss = torch.sum(w * torch.abs(tgt - src), dim=1)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss.unsqueeze(1)

def structural_consistency_loss_func(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the structural consistency loss using SSIM

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 3 x H x W weights
        reduce_loss : bool
            if set then return mean over loss
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean 1 - SSIM scores between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : N x 1 x H x W tensor of 1 - SSIM scores between source and target images
    '''

    if w is None:
        w = torch.ones_like(src)

    refl = torch.nn.ReflectionPad2d(1)

    src = refl(src)
    tgt = refl(tgt)
    scores = ssim(src, tgt)

    loss = torch.sum(w * scores, dim=1)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss.unsqueeze(1)

def sparse_depth_consistency_loss_func(src, tgt, w=None):
    '''
    Computes the sparse depth consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 1 x H x W source depth
        tgt : torch.Tensor[float32]
            N x 1 x H x W target depth
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target depth
    '''

    if w is None:
        w = torch.ones_like(src)
    delta = torch.abs(tgt - src)
    loss = torch.sum(w * delta, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def smoothness_loss_func(predict, image, w=None):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    if w is not None:
        weights_x = weights_x * w[:, :, :, :-1]
        weights_y = weights_y * w[:, :, :-1, :]

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y

def pose_consistency_loss_func(pose0, pose1, use_pytorch_impl=False):
    '''
    Computes the pose consistency loss

    Arg(s):
        pose0 : torch.Tensor[float32]
            N x 4 x 4 pose matrix
        pose1 : torch.Tensor[float32]
            N x 4 x 4 pose matrix
    Returns:
        float : L2 distance from identity
    '''

    n_batch, _, _ = pose0.shape
    eye = torch.unsqueeze(torch.eye(4, 4, device=pose0.device), dim=0) \
        .repeat(n_batch, 1, 1)

    pose = torch.matmul(pose0, pose1)

    loss_func = torch.nn.MSELoss(reduction='mean')

    return loss_func(pose.view(n_batch, -1), eye.view(n_batch, -1))


'''
Helper functions for constructing loss functions
'''
'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes Structural Similarity Index distance between two images

    Arg(s):
        x : torch.Tensor[float32]
            N x 3 x H x W RGB image
        y : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : SSIM distance between two images
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1)*(2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)
