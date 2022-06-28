load tmp.dat;
A = spconvert(tmp);
B = [];

NG=2;
alpha=1;%0.3;
beta=50;

% Run enumKOCG.m
cd KOCG;
[X_enumKOCG_cell, time] = enumKOCG(A, B, NG, alpha, beta);
cd ..;

fid = fopen('FOCG.out','w');
for i=1:length(X_enumKOCG_cell)
	X = X_enumKOCG_cell{i};

	% write the non-zero entries of X(:,1) and X(:,2) separately
	for j=1:2
		v = X(:,j);
		indices = find(v);

		fprintf(fid,'%d ',indices);
		fprintf(fid,'\n');
	end
end
fclose(fid);

