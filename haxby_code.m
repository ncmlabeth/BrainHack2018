clear cfg

% for n=1:12
%     for i=(n-1)*16+1:n*16
%         regressor_names{2,i}=n;
%     end
% end
% 
% cfg.files.chunk=[1:16 1:16];
% cfg.files.label=[-ones(16,1) ones(16,1)];

decoding_type = 'roi';
labelname1= 'face';
labelname2= 'house';
beta_dir='C:\Machine Learning\subj1\beta images';
output_dir = 'C:\Machine Learning\Results6';

[results, cfg] = decoding_example2(decoding_type,labelname1,labelname2,beta_dir,output_dir);