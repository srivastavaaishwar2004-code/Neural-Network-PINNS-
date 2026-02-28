import torch 

def dispersion_fun(ww, Vrr, vp, vs,h):
    Vrr = torch.tensor(Vrr, dtype=torch.complex64)
    vp = torch.tensor(vp, dtype=torch.complex64)
    vs = torch.tensor(vs, dtype=torch.complex64)
    h = torch.tensor(h, dtype=torch.complex64)
    ww = torch.tensor(ww, dtype=torch.complex64)

    k = ww / Vrr

    h1 = h[:,0]

    Vrr_2d = Vrr.unsqueeze(1)
    rp = torch.sqrt(Vrr_2d**2 - vp**2 - 1)
    rs = torch.sqrt(Vrr_2d**2 - vs**2 - 1)
    r = 1 - Vrr_2d**2 / (2 * vs**2)
    g = 1 -r
    rr = rp**2
    s  = rs**2
    rp2 = rp[:,1]
    rs2 = rs[:,1]
    r2 = r[:,1]

    E = torch.stack([
        1 + rp*rs2,
        r2 + rp*rs2,
        rs2*(r2 - 1)*1j,
        -r2**2 - rp2*rs2
    ],  dim=1)

    # layer 1 values
    rp1 = rp[:,0]
    rs1 = rs[:,0]
    r1 = r[:,0]
    g1 = g[:,0]
    rr1 = rr[:,0]
    s1 = s[:,0]

    ph = rp1 * k * h1
    qh = rs1 * k * h1

    a = torch.cos(ph)
    b = torch.cos(qh)
    c = torch.sin(ph) / rp1
    d = torch.sin(qh) / rs1

    # build M1 
    z = torch.zeros_like(r1)   # zeros of shape (batch,)
    o = torch.ones_like(r1)    # ones  of shape (batch,)

    M1 = torch.stack([
        torch.stack([ o,       2*o,     z,   z,  -o    ], dim=1),
        torch.stack([ r1,      1+r1,    z,   z,  -o    ], dim=1),
        torch.stack([ z,       z,       g1,  z,   z    ], dim=1),
        torch.stack([ z,       z,       z,   g1,  z    ], dim=1),
        torch.stack([-r1**2,  -2*r1,    z,   z,   o    ], dim=1),
    ], dim=1)   

    # build L 
    L = torch.stack([
        torch.stack([ a*b,         z,  -a*d,        b*c,        c*d      ], dim=1),
        torch.stack([ z,           o,   z,           z,          z       ], dim=1),
        torch.stack([ a*d*s1,      z,   a*b,         c*d*s1,    -b*c     ], dim=1),
        torch.stack([-b*c*rr1,     z,   c*d*rr1,     a*b,        a*d     ], dim=1),
        torch.stack([ c*d*rr1*s1,  z,   b*c*rr1,    -a*d*s1,     a*b    ], dim=1),
    ], dim=1)   # 

    # build M2 —
    # note: l = ones so 1/l = 1, -l = -1
    M2 = torch.stack([
        torch.stack([ o,        -2*o,    z,   z,  -o    ], dim=1),
        torch.stack([-r1,        1+r1,   z,   z,   o    ], dim=1),
        torch.stack([ z,         z,      g1,  z,   z    ], dim=1),
        torch.stack([ z,         z,      z,   g1,  z    ], dim=1),
        torch.stack([-r1**2,     2*r1,   z,   z,   o    ], dim=1),
    ], dim=1)   

    F = torch.bmm(torch.bmm(M1,L), M2)
    
    E = torch.bmm(F, E.unsqeezse(2).squeeze(2)
                  
    return E[:,4].real

if __name__ == "__main__":
    Vrr = torch.tensor([180.0,190.0,175.0,200.0])
    vp  = torch.tensor([[340.0,680.0],
                       [357.0,646.0],
                       [331.0,714.0],
                       [374.0,595.0]]) 
    vs  = torch.tensor([[200.0,400.0],
                       [210.0,380.0],
                       [195.0,420.0],
                       [220.0,350.0]])
    h   = torch.tensor([[21.0],
                        [15.0],
                        [18.0],
                        [12.0]
                        ])

    result = dispersion_fun(10.0,Vrr, vp, vs, h)
    print("Result shape:", result.shape)
    print("Result:", result)

    vp_real = torch.tensor([[1.7*217.0,1.7*496.0]])
    vs_real = torch.tensor([[217.0,496.0]])
    h_real = torch.tensor([[21.4]])

    Vrr_true = torch.tensor([274.3])
    Vrr_wrong = torch.tensor([200.0])
    Vrr_worse = torch.tensor([150.0])

    r1 = dispersion_fun(10.0, Vrr_true, vp_real, vs_real, h_real)
    r2 = dispersion_fun(10.0, Vrr_wrong, vp_real, vs_real, h_real)
    r3 = dispersion_fun(10.0, Vrr_worse, vp_real, vs_real, h_real)

    print(f"True Vr=274.3 ->residual: {r1.item():.6f} (should be close to 0)") 
    print(f"Wrong Vr=200.0 ->residual: {r2.item():.6f} (should not be zero)")
    print(f"Worse Vr=150.0 ->residual: {r3.item():.6f} (should not be zero)")