function [] = fisher_face_align()
[m_face_train,m_face_test,m_land_train,m_land_test,f_face_train,f_face_test,f_land_train,f_land_test,m_mx,f_mx,m_lx,f_lx]=load_fisher_align();
% Fisher linear on apperance, all faces are aligned.
% Step1: C: 256^2*d
[m_face_train,m_face_test,f_face_train,f_face_test,m_mx,f_mx] = load_fisher();
C = [double(m_face_train)-repmat(m_mx,1,78),double(f_face_train)-repmat(f_mx,1,75)];
% C = [double(m_face_train),double(f_face_train)];
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
p_m =(W'*(double(m_face_train)));
p_f = (W'*(double(f_face_train)));
p_m_test = (W'*(double(m_face_test)));
p_f_test = (W'*(double(f_face_test)));
% plot(p_m_test,zeros(1,10),'*',p_f_test,zeros(1,10),'o');
% legend({'Male test','Female test'});
% xlabel('Appearance');
% title('Fisher faces with warpping face');
%========================================================
% Fisher face for geometric shape
% Step1: C: 153*d
C = [m_land_train-repmat(m_lx,1,78),f_land_train-repmat(f_lx,1,75)];
% C = [m_land_train,f_land_train];
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
y = A'*(f_lx-m_lx);
% Step6: Get z:
z = lambda^2*V'\y;
% Step7: Compute W:
W = C*z;
W = W/norm(W);
% subplot(1,2,1);
p_m2 = (W'*(m_land_train));
p_f2 = (W'*(f_land_train));
n = norm([p_m2,p_f2]);
p_m2_test = (W'*(m_land_test));
p_f2_test = (W'*(f_land_test));
% plot(p_m2_test,zeros(1,10),'*',p_f2_test,zeros(1,10),'o');

% subplot(2,1,1)
plot(p_m,p_m2/n,'*',p_f,p_f2/n,'o');
legend({'Male train','Female train'})
xlabel('Projection on appearance');
ylabel('Projection on geometric shape');
title('2D feature space on training set')

% subplot(2,1,2);
% plot(p_m_test,p_m2_test,'*',p_f_test,p_f2_test,'o');
% legend({'Male test','Female test'})
% xlabel('Projection on appearance');
% ylabel('Projection on geometric shape');
% title('2D feature space on testing set')
% plot(p_m2,zeros(1,78),'^',p_f2,zeros(1,75),'o');

% Plot training and testing together
% subplot(3,1,3);
% plot(p_m_n,p_m2_n,'^',p_f_n,p_f2_n,'o',p_m_test,p_m2_test,'^',p_f_test,p_f2_test,'o');
% legend({'Male train','Female train','Male test','Female test'})
% xlabel('Projection on appearance');
% ylabel('Projection on geometric shape');
% title('2D feature space')
%==================================================
% p1 = plot(p_m,zeros(1,78),'^','Color',[65,105,225]/256);
% % xlim([.6,2.4]*10^-5);
% hold on;
% p2 = plot(p_f,zeros(1,75),'s','Color',[255,99,71]/256);
% % xlim([.6,2.4]*10^-5);
% legend({'Male','Female'});
% title('Projection on train set');
% hold off;
% subplot(2,1,2);
% p_m = W'*(double(m_face_test));
% p_f = W'*(double(f_face_test));
% plot(p_m,zeros(1,10),'^','Color',[65,105,225]/256);
% % xlim([.6,2.4]*10^-5);
% hold on;
% plot(p_f,zeros(1,10),'s','Color',[255,99,71]/256);
% % xlim([.6,2.4]*10^-5);
% legend({'Male','Female'});
% title('Projection on test set');
end

% Help function 256^2*1 -> 256*256
function f = m2f(m)
f = [];
for i = 1:256
    f = [f,m(((i-1)*256+1):(i*256),1)];
end
end