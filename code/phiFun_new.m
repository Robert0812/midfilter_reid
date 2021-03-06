function phi = phiFun_new(x, y, m1, s1, s2, m2, par, x_all)

% global par;
p1 = s1;
A = (exp(-double(x).^2/par.sigma1^2).*m1).^par.nor;

if nargin >= 8
    [ny, nx] = size(s1);
    A_all = (exp(-double(x_all).^2/par.sigma1^2).*repmat(m1, [1, 1, nx])).*par.nor;
    p2 = s2;
    phi = zeros(8, nx, nx, ny);
    for r = 1:ny
        pwsim = squeeze(A_all(r, :, :));
        for i = 1:nx
            phi(:, :, i, r) = ...
                cat(1, p1(r, i).*pwsim(i, :).*p2(r, :), ...
                p1(r, i).*pwsim(i, :).*(1 - p2(r, :)), ...
                (1-p1(r, i)).*pwsim(i, :).*p2(r, :), ...
                (1-p1(r, i)).*pwsim(i, :).*(1-p2(r, :)), ...
                p1(r, i).* p2(r, :), ...
                p1(r, i).*(1 - p2(r, :)), ...
                (1-p1(r, i)).*p2(r, :), ...
                (1-p1(r, i)).*(1-p2(r, :)));
        end
    end
else
    
    % phi = cat(3, (p1.*A).^par.nor);
    
    
    % phi = cat(3, p1.*p2.*A, ...
    %             p1.*(1-p2).*A, ...
    %             (1-p1).*p2.*A, ...
    %             (1-p1).*(1-p2).*A, ...
    %             0.35*((p1.*p2).*m1), ...
    %             0.35*(p1.*(1-p2).*m1), ...
    %             0.35*((1-p1).*p2.*m1), ...
    %             0.35*((1-p1).*(1-p2).*m1));
    
    p2 = get_saj(y, s2);
    
%     phi = cat(3, p1.*p2.*A, ...
%         p1.*(1-p2).*A, ...
%         (1-p1).*p2.*A, ...
%         (1-p1).*(1-p2).*A, ...
%         0.35*((p1.*p2).*m1), ...
%         0.35*(p1.*(1-p2).*m1), ...
%         0.35*((1-p1).*p2.*m1), ...
%         0.35*((1-p1).*(1-p2).*m1));
    scale = 0.49; % VIPeR: 0.35 CAMPUS: 0.49
%       scale = 0.45;
%     phi = cat(3, p1.*p2.*A, ...
%             p1.*(1-p2).*A, ...
%             (1-p1).*p2.*A, ...
%             (1-p1).*(1-p2).*A, ...
%             scale*((p1.*p2).*m1), ...
%             -scale*(p1.*(1-p2).*m1), ...
%             -scale*((1-p1).*p2.*m1), ...
%             scale*((1-p1).*(1-p2).*m1));
        
    phi = double(cat(3, p1.*p2.*A, ...
            p1.*(1-p2).*A, ...
            (1-p1).*p2.*A, ...
            (1-p1).*(1-p2).*A, ...
            scale*((p1.*p2)), ...
            -scale*(p1.*(1-p2)), ...
            -scale*((1-p1).*p2), ...
            scale*((1-p1).*(1-p2))));
end

