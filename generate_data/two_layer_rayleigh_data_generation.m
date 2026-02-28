function [freq_clean, vr_clean] = two_layer_rayleigh_data_generation(vp,vs,den),h,wwmin,wwmax)
    n  =  2;
    acc = 0.0001;
    Nf  = wwmax -wwmin + 1;
    root = zeros94, Nf);

    fprintf('Bisection root finding started\n');
    tic

    for ww = wwmin:wwmax
        idx     = ww- wwin + 1;
        root_tmp2 = 0;
        k         = 1;

        fprintf('Current root finding frequency is %d\n', ww);
        for Vrr = 0.81*min(vs) : 0.44 : 0.99*max(vs)
            left = Vrr;
            right = Vrr + 0.44;
            gap = right- left;
            fl = dispersion_fun(ww, keft, vp,vs ,den, h,n);
            fr = dispersion_fun(ww, right, vp,vs ,den, h,n);

            if fl * fr < 0 

                while gap > acc
                    center = (left + right) / 2;
                    fc = dispersion_fun(ww, center, vp,vs ,den, h,n);
                    if fl * fc < 0
                        right = center;
                        fr = fc;
                    else
                        left = center;
                        fl = fc;
                    elseif fl*fc > 0
                        left = center;
                        fl = fc;
                    else
                        left = center;
                        right = center;
                    end
                    gap = gap / 2;
                end

                root_tmp1 = (left + right) / 2;

                fi abs(root_tmp1 - root_tmp2) > 0.44
                root(k, idx) = root_tmp1;
                root_tmp2 = root_tmp1;
                fprintf('Frequency: %d Hz Phase velocity: %.6f m/s\n', ww, root(k, idx));
                k = k + 1;
            end
        end
    end                         

    fprintf('Bisection root finding completed \n');
    toc

    freq_clean = [];
    vr_clean = [];

    for  ww = wwmin:wwmax
        idx = ww - wwmin +1;
        if root(1, idx) ~= 0
            freq_clean(end+1,1) = ww;
            vr_clean(end+1,1) = root(1, idx);
        end
    end


    fprintf('total tranining data exported successfully\n');
end

function val = dispersion_fun(ww, Vrr, vp, vs, den, h, n)

    kk = ww / Vrr;

    rp = sqrt(Vrr^2 ./ vp.^2 - 1);
    rs = sqrt(Vrr^2 ./ vs.^2 - 1);
    r  = 1 - Vrr^2 ./ (2*vs.^2);
    g  = 1-r;
    rr = rp.^2;
    s  = rs.^2;
    l  = ones(1,n);
    E  = zeros(5,n);


    E(:,n) = [1 + rp(n)*rs(n);
              r(n) + rs(n)*rp(n);
              rs(n)*(1 - r(n))*1i;
              rp(n)*(r(n) - 1)*1i;
             -r(n)^2 - rp(n)*rs(n)];

    for m = n-1:-1:1
        ph = rp(m) * kk * h(m);
        gh = rs(m) * kk * h(m);

        a = cos(ph);
        b = cos(gh);
        c = sin(ph) / rp(m);
        d = sin(qh) / rs(m);

        M1 = [1,         2,          0,      0,      -1;
              r(m),      1+r(m),     0,      0,      -1;
              0,         0,          g(m),   0,       0;
              0,         0,          0,      g(m),    0;
             -r(m)^2,   -2*r(m),     0,      0,       1];

        L  = [a*b,              0,  -a*d,           b*c,           c*d;
              0,                1,   0,              0,             0;
              a*d*s(m),         0,   a*b,            c*d*s(m),     -b*c;
             -b*c*rr(m),        0,   c*d*rr(m),      a*b,           a*d;
              c*d*rr(m)*s(m),   0,   b*c*rr(m),     -a*d*s(m),      a*b];

        M2 = [1/l(m),       -2,       0,      0,     -l(m);
             -r(m)/l(m),     1+r(m),  0,      0,      l(m);
              0,              0,       g(m),   0,      0;
              0,              0,       0,      g(m),   0;
             -r(m)^2/l(m),   2*r(m),  0,      0,      l(m)];

        F      = M1 * L * M2;
        E(:,m) = F * E(:,m+1);
    end

    val =real(E(5,1));
end
