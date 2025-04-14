% NMF function
function [W,H] = NMF(V,r)
    % given V, find W, H such that V = WH
    % V = WH <=> || V - WH|| = small
    % Define my objective function f
    [m,n] = size(V);
    % Use r from function parameter instead of hardcoding
    W_ini = rand(m,r);
    H_ini = rand(r,n);
    f_ini = 0.5 * norm( V - W_ini * H_ini, 'fro')^2;
    W = W_ini;
    H = H_ini;
    for ii = 1 : 500
     gradW = (W*H-V)*H';
     f_new = inf;
     stepsize = 1;
    while f_new > f_ini
     stepsize = stepsize/2;
     W_temp = max( W - stepsize * gradW, 0);
     f_new = 0.5 * norm( V - W_temp*H, 'fro')^2;
    end
     W = W_temp;
     f_ini = f_new;
     gradH = W'*(W*H-V);
     f_new = inf;
     stepsize = 1;
    while f_new > f_ini
     stepsize = stepsize/2;
     H_temp = max( H - stepsize * gradH, 0);
     f_new = 0.5 * norm( V - W*H_temp, 'fro')^2;
    end
     H = H_temp;
    end
end