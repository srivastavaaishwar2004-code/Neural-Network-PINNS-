import torch 
from surrogate import SurrogateNet

def load_surrogate(path='inversion_model/saved_weights/surrogate.pt'):

    net = SurrogateNet()
    net.load_Sate_dict(torch.load(path))

    for param in net.parameters():
        param.requires_grad = False

    net .eval()
    return net

def data_loss(predicted_norm, true_norm):
    return torch.mean((predicted_norm - true_norm)**2)

def monotonic_loss(predicted_norm, true_norm):
    # penalize non-monotonicity
    vs1_norm = predicted_norm[:,0] 
    vs2_norm = predicted_norm[:,1] 
    violation = torch.relu(vs1_norm - vs2_norm) 
    return torch.mean(violation**2) 

def physics_loss(predicted_norm, curves_norm, surrogate_net):

    resdiduals = []
    for i in range (curves_norm.shape[1]):

        freq_n   = curves_norm[:,i,0] 
        Vr_obs_n = curves_norm[:,i,1] 

        surrogate_input = torch.cat([predicted_norm, freq_n.unsqueeze(1)], dim=1)

        Vr_pred_n = surrogate_net(surrogate_input)

        residual = (Vr_pred_n.squeeze(1) - Vr_obs_n)**2 
        residuals.append(residual) 

    residuals = torch.stack(residuals, dim=1)
    return torch.mean(residuals)


def total_loss(predicted_norm, true_norm, curves_norm, 
               surrogate_net, lambda_physics=0.0):
      
      d_loss = data_loss(predicted_norm, true_norm)
      m_loss = monotonic_loss(predicted_norm)
      p_loss = physics_loss(predicted_norm, curves_norm, surrogate_net)
      total  = d_loss + 0.1 * m_loss + lambda_physics * p_loss 
      return total, d_loss, m_loss, p_loss 
