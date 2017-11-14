function TN=matrix2mpo(A,n,varargin)
% TN=matrix2mpo(A,n)
% ------------------
% Converts a given (sparse) matrix A into the Tensor Network MPO-form
% with given MPO-tensor dimensions as specified by each row of the matrix
% n. 
%
% TN        =   Tensor Network, MPO that represents a given matrix A,
%
% A         =   matrix, (sparse) matrix,
%
% n         =   matrix, n(i,:) contains the dimensions of the ith MPO-tensor.
%
% Reference
% ---------
%
% Computing low-rank approximations of large-scale matrices with the tensor network randomized SVD.
%
% 06-07-2016, Kim Batselier, Wenjian Yu, Luca Daniel, Ngai Wong

addpath(genpath(pwd));

if isscalar(n)
    d=varargin{1};
else
    d=size(n,1);
end

tileX=n(1,1);
tileY=n(1,2);

% first keep track how many nonzero block matrices there are and where they
% are
R=0; % the total maximal MPO-rank
rows=cell(1,prod(n(2:end,2)));
for i=1:prod(n(2:end,2)) % i counts the columns
   % determine which blocks on this "column" are nonzero
   I=find(sum(abs(A(:,(i-1)*tileY+1:i*tileY)),2));
   if ~isempty(I)
       I2=find(rem(I,tileX)==0);
       I=fix(I/tileX);
       I=I+1;
       I(I2)=I(I2)-1;
       rows{i}=[I(1);I(find(diff(I))+1)]';
       R=R+length(rows{i});
   end
end   

% initialize TN
TN.core=cell(1,d);
TN.n=[1 n(1,:) R;R*ones(d-2,1) n(2:end-1,:) R*ones(d-2,1);R n(end,:) 1];
nze=cell(1,d);

r=1;
for i=1:prod(n(2:end,2)) 
   if ~isempty(rows{i})
       for j=rows{i} %j iterates over the nonzero row blocks
           % determine the nonzero entries in each of the Kronecker product
           % matrices
           indices=[lin2ten(j,n(2:end,1))' lin2ten(i,n(2:end,2))'];
           [ii,jj,vv]=find(A((j-1)*tileX+1:j*tileX,(i-1)*tileY+1:i*tileY));
           temp=ones(numel(ii),1);
           nze{1}=[nze{1};temp ii jj r*temp vv];
           for k=2:d-1
               nze{k}=[nze{k};r indices(k-1,1) indices(k-1,2) r 1];
           end
           nze{d}=[nze{d};r indices(d-1,1) indices(d-1,2) 1 1];
           r=r+1;
       end
   end
end

for i=1:d
    TN.core{i}=ndSparse.build(nze{i}(:,1:4),nze{i}(:,5),TN.n(i,:));
end

end
