function [] = eig_landmark()
[train,test,m_l] = load_disp();

% Center train and test
for i = 1:177
   if i>150
       test_m(i-150,:) = test(i-150,:)-m_l;
   else
       train_m(i,:) = train(i,:)-m_l;
   end
end
% Calculate eigen vectors
[v,~] = eigs(train_m'*train_m,149);

% Plot the first 5 eigen warppings
% for i = 1:5
%    subplot(1,5,i)
%    eig_land = m2l(v(:,i)'*15^2+m_l);
%    plot(eig_land(:,1),eig_land(:,2),'.');
% end

% Reconstruct test landmarks by first 5 landmarks
% for i = 1:5
%    test_l = m2l(test(i,:));
%    subplot(2,5,i+5)
%    plot(test_l(:,1),test_l(:,2),'.');
%    eig_l = m2l(test_m(1,:)*v(:,1:5)*v(:,1:5)'+m_l);
%    subplot(2,5,i)
%    plot(eig_l(:,1),eig_l(:,2),'.');
% end

% Test error
test_err = [];
rec_l = zeros(27,174);
for i = 1:149
    rec_l = rec_l+test_m*v(:,i)*v(:,i)';
    test_err = [test_err,sum(sum(abs(test_m-rec_l)))/(27*174)];
end
plot([1:149],test_err);
xlabel('Num of eigen vectors');
ylabel('Test error L1 norm');
title('Reconstruction error of landmarks');
end

%Help function
function land = m2l(m)
land = [m(1,1:87)',m(1,88:174)'];
end