%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spherical Harmonic Modeling and Analysis Toolkit (SPHARM-MAT) is a 3D 
% shape modeling and analysis toolkit. 
% It is a software package developed at Shenlab in Center for Neuroimaging, 
% Indiana University (SpharmMat@gmail.com, http://www.iupui.edu/~shenlab/)
% It is available to the scientific community as copyright freeware 
% under the terms of the GNU General Public Licence.
% 
% Copyright 2009, 2010, ShenLab, Center for Neuroimaging, Indiana University
% 
% This file is part of SPHARM-MAT.
% 
% SPHARM-MAT is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% SPHARM-MAT is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with SPHARM-MAT. If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function saveOptions(inputFile, confs, inObjs1, inObjs2)

fout = fopen(inputFile, 'wt');

for k = 1:length(confs.vars)
    if confs.args(k) < 10
        varSTR = sprintf('confs.%s', confs.vars{k});
        fprintf(fout, '%s %s\n', confs.vars{k}, num2str(eval(varSTR)));
    else
        varSTR = sprintf('confs.%s', confs.vars{k});
        fprintf(fout, '%s %s\n', confs.vars{k}, char(eval(varSTR)));
    end
end

if ~isempty(inObjs1)
    fprintf(fout, 'inputs\n');
    for l = 1:length(inObjs1)
        fprintf(fout, '%s\n', inObjs1{l});
    end
end

if ~isempty(inObjs2)
    fprintf(fout, 'inputs2\n');
    for l = 1:length(inObjs2)
        fprintf(fout, '%s\n', inObjs2{l});
    end
end

fclose(fout);


return;