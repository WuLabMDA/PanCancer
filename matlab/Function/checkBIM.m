function [bim, vxsize] = checkBIM(bim,origin,vxsize)

bim = double(bim);
bdr_num = countBDRY(bim,origin,vxsize);

while bdr_num < 10000
    bim = imresize3(bim, 2, 'nearest');
    vxsize = vxsize./2;
    bdr_num = countBDRY(bim,origin,vxsize);
end

end

function bdr_num = countBDRY(bim,origin,vxsize)
[vertices, faces] =  gen_surf_data(bim,origin,vxsize);

bdr_num = size(vertices,1);

end
