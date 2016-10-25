function fisher_face = fisher_face()
[m_face_train,m_face_test,f_face_train,f_face_test,m_mx,f_mx] = load_fisher();

% Step1: C: 256^2*d
C = [double(m_face_train)-repmat(m_mx,1,78),double(f_face_train)-repmat(f_mx,1,75)];
[~,d] = size(C);
% Step2: B: d*d
B = C'*C;
% Step3: Solve eigen values (lambda), eigen vectors (V) of B.
[V,lambda] = eig(B);
% Step4: Define A: 256^2*d
A = [];
for i = 1:d
    temp = C*V(:,i);
    A =  [A,(sqrt(lambda(i,i))/norm(temp))*temp];
end
% Step5: Compute y = A'(f_mx-m_mx), where y: d*1
y = A'*(f_mx-m_mx);
% Step6: Get z:
z = (lambda^2*V')\y;
% Step7: Compute W:
W = C*z;
W = W/norm(W);
% Rescale
fisher_face = W;
subplot(2,1,1);
p_m = W'*(double(m_face_train));
p_f = W'*(double(f_face_train));
p1 = plot(p_m,zeros(1,78),'*',p_f,zeros(1,75),'o');
xlim([.6,2.4]*10^-5);
legend({'Male train','Female train'});
title('Projection on train set');
subplot(2,1,2);
p_m = W'*(double(m_face_test));
p_f = W'*(double(f_face_test));
plot(p_m,zeros(1,10),'*',p_f,zeros(1,10),'o');
xlim([.6,2.4]*10^-5);
legend({'Male test','Female test'});
title('Projection on test set');
end

% Help function 256^2*1 -> 256*256
function f = m2f(m)
f = [];
for i = 1:256
    f = [f,m(((i-1)*256+1):(i*256),1)];
end
end


