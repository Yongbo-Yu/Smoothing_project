% Implementation of the Aguilera-Perez Algorithm.
% Aguilera, Antonio, and Ricardo P?rez-Aguila. "General n-dimensional rotations." (2004).
function M = rotmnd(v,theta)
    n = size(v,1);
    M = eye(n);
    for c = 1:(n-2)
        for r = n:-1:(c+1)
            t = atan2(v(r,c),v(r-1,c));
            R = eye(n);
            R([r r-1],[r r-1]) = [cos(t) -sin(t); sin(t) cos(t)];
            R([r r-1],[r r-1])
            v = R*v;
            M = R*M;
            
        end
    end
    R = eye(n);
    R([n-1 n],[n-1 n]) = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    M = M\R*M;