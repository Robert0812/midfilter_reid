function salience = get_saj(disp1, sa2)
% compute matched salience for gallery image
% 
% INPUT
%   disp1: query to gallery matching displacement
%   sa2:    original salience map of gallery image
%  
% OUTPUT
%   salience: query matched salience 
%

[ny, nx] = size(disp1);
salience = sa2(sub2ind([ny, nx], repmat((1:ny)', 1, nx), double(disp1)));
