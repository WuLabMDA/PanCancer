function [fvec, deg, Z] = create_SPHARM_des_LSF2(vertices, sph_verts ,maxDeg)

vertnum = size(sph_verts,1);

max_d = maxDeg;
% Note that degree 'd' we want to use depends on the vertnum 
% The total number of unknowns is (d+1)*(d+1)
% The total number of equations is vertnum
% We want equ_num >= unk_num
deg = max(1, floor(sqrt(vertnum)*1/2));
deg = min(deg, max_d);

Z = calculate_SPHARM_basis(sph_verts, deg); 

[x,y] = size(Z);
disp(sprintf('Least square for %d equations and %d unknowns',x,y));

% Least square fitting
fvec = Z\vertices;   %This does not work as it is expected to work in certain environment
% for i=1:size(vertices,2)
%     fvec(:,i) = Z\vertices(:,i);
% end

return;
