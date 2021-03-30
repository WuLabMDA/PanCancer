function [names,rad,rad_norm,shape_scale,shape_scale_norm,energy_dgr,energy_dgr_norm,energy_skewness,volume, spharm_coeff] = spharm_coeff(objs, binaryDir, degree)

numSbj = length(objs);

names = {};
spharm_coeff = zeros(numSbj, (degree+1)^2);
% do processing of each file
for i = 1:numSbj
    % read in file.
    file = objs{i};
    disp(file);
    load(file);
    [path, name, ext] = fileparts(file);
    fvec = abs(fvec);
    newA = zeros(256,3);
    newA(1:size(fvec,1),1:size(fvec,2)) = fvec;
    fvec = newA;
    [nrows ncols] = size(fvec);
    spharm_coeff(i,:) = sqrt(sum(fvec.^2, 2));
    
    % check the size of aligned spharm coefficients
    if nrows ~= 256 | ncols ~=3
        error('coefficient size is wrong!!');
    end
    
    if size(vertices,1) < 5000
        warning('tumor vertices number is wrong!!');
    end
    
    tmp2 = vertices - mean(vertices);
    rad(i) = sum(sqrt(sum(tmp2.^2, 2)))/size(tmp2,1);  % the scale for each shape
    tmp2 = fvec(2:4,:); 
    shape_scale(i,:) = [sqrt(sum(tmp2(:,1).^2)/3), sqrt(sum(tmp2(:,2).^2)/3), sqrt(sum(tmp2(:,3).^2)/3)];
    
    if ~issorted(shape_scale(i,:))
        warning(file);
        warning('***********not sorted!!!************');
    end
    
    fvec_norm = fvec./rad(i);
    vertices_norm = vertices./rad(i);
    tmp2 = vertices_norm - mean(vertices_norm);
    rad_norm(i) = sum(sqrt(sum(tmp2.^2, 2)))/size(tmp2,1);  % the scale for each normalized shape
    tmp2 = fvec_norm(2:4,:);
    shape_scale_norm(i,:) = [sqrt(sum(tmp2(:,1).^2)/3), sqrt(sum(tmp2(:,2).^2)/3), sqrt(sum(tmp2(:,3).^2)/3)];
    
    for j = 1:degree  % start from degree of 1, exclude first 3 elements for translation
        idx1 = j^2 + 1;
        idx2 = (j+1)^2;
        tmp = fvec(idx1:idx2,:);
        energy_dgr(i,j) = sqrt(sumsqr(tmp)); % the absolute energe at each degree
        
        tmp = fvec_norm(idx1:idx2,:);
        energy_dgr_norm(i,j) = sqrt(sumsqr(tmp));  % the normalized coefficient
        
        tmp = sum(tmp.^2,2);
        tmp = tmp./sum(tmp);
        tmp1 = log(tmp+1e-5);
        energy_skewness(i,j) = -1*sum(tmp.*tmp1); % the skewness of energy distribution by shannon entropy
    end
    
    names{i} = extractBefore(name,'_CALD');
    cd(binaryDir)
    load([names{i} '_fix.mat'])
    volume(i) = sum(bim(:))*prod(vxsize);
    
    clear('faces', 'vertices', 'sph_verts','fvec');
end
 
end