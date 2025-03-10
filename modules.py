import torch
import torch.nn as nn


def get_psnr(pred, gt, max_value=1.0):
    """
    Compute PSNR between signals
    """
    mse = torch.mean(torch.pow(pred - gt, 2))
    return 20 * torch.log10(max_value / torch.sqrt(mse))


class STFTLoss(nn.Module):
    """
    STFT loss
    """
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=512, 
                 window='hann',
                 win_length=None,
                 mag_weight=1.0,
                 ph_weight=1.0,
                 use_log=False):
        super(STFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        if window == 'hann':
            self.window = torch.hann_window(n_fft)
        elif window == 'hamm':
            self.window = torch.hamming_window(n_fft)
        else:
            self.window = torch.hann_window(n_fft)

        self.window = self.window.to('cuda')
        
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length

        self.mw = mag_weight
        self.phw = ph_weight

        self.use_log = use_log

        self.loss_fn = nn.L1Loss()

    def forward(self, pred, gt):
        
        pred = pred.squeeze(1)
        gt = gt.squeeze(1)

        pred_stft = torch.stft(pred,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length, 
                            window=self.window,
                            win_length=self.win_length,
                            center=True,
                            return_complex=True)
        gt_stft = torch.stft(gt,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length, 
                            window=self.window,
                            win_length=self.win_length,
                            center=True,
                            return_complex=True)

        pred_mag, pred_phase = pred_stft[..., 0], pred_stft[..., 1]
        gt_mag, gt_phase = gt_stft[..., 0], gt_stft[..., 1]

        mag_loss = self.loss_fn(pred_mag, gt_mag)
        ph_loss = self.loss_fn(pred_phase, gt_phase)

        if self.use_log:
            mag_loss = torch.log(mag_loss + 1)
            ph_loss = torch.log(ph_loss + 1)

        return self.mw * mag_loss + self.phw * ph_loss
    

class L1STFTLoss(nn.Module):
    """
    Combination of L1 loss and STFT losss
    """
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=512, 
                 window='hann',
                 win_length=None,
                 mag_weight=1.0,
                 ph_weight=1.0,
                 use_log=False):
        super(L1STFTLoss, self).__init__()

        self.l1_loss_fn = nn.L1Loss()
        self.stft_loss_fn = STFTLoss(n_fft=n_fft,
                                     hop_length=hop_length,
                                     window=window,
                                     mag_weight=mag_weight,
                                     ph_weight=ph_weight,
                                     win_length=win_length,
                                     use_log=use_log)

    def forward(self, pred, gt):
        l1_loss = self.l1_loss_fn(pred, gt)
        stft_loss = self.stft_loss_fn(pred, gt)

        return l1_loss + stft_loss