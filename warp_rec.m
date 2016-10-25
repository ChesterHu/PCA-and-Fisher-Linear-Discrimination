function [] = warp_rec()
%Load images and landmarks, img_train&test are uint8.
[img_train,img_test,img_mx] = load_data;
[l_train,l_test,l_mx] = load_disp;

% Center data
for i = 1:177
    if i>150
       img_test_m(i-150,:) = double(img_test(i-150,:))-img_mx; 
       l_test_m(i-150,:) = l_test(i-150,:)-l_mx;
    else
       img_train_m(i,:) = double(img_train(i,:))-img_mx;
       l_train_m(i,:) = l_train(i,:)-l_mx;
    end
end

% Eigen vectors
[l_v,l_d] = eigs(l_train_m'*l_train_m,10);
[img_u,img_d] = eigs(img_train_m*img_train_m',149);
img_v = img_train_m'*img_u;
% Normalize v
for i = 1:149
   img_v(:,i) = img_v(:,i)/norm(img_v(:,i)); 
end

img_u = [];

% Step1: project landmarks on top 10 eigen vectors
rec_l = l_test_m*l_v(:,1:10)*l_v(:,1:10)';
% 
% % Step2:
% Warp test images to mean postion;
for i = 1:27
   temp = m2f(img_test(i,:));
   test_m_w(i,:) = f2m(double(warpImage_kent(temp,[l_test(i,1:87)',l_test(i,88:174)'],[l_mx(1,1:87)',l_mx(1,88:174)'])))-img_mx;
end
% % Project&Reconstruct test images on k=10 eigen-faces
% rec_f = test_m_w*img_v(:,1:10)*img_v(:,1:10)';
% 
% 
% % Step3: warp the reconstructed face to the reconstructed landmarks
% for i = 1:27
%    temp = m2f(rec_f(i,:)+img_mx);
%    res(i,:) = f2m(double(warpImage_kent(temp,[l_mx(1,1:87)',l_mx(1,88:174)'],[rec_l(i,1:87)'+l_mx(1,1:87)',rec_l(i,88:174)'+l_mx(1,88:174)'])));
% end
% 
% % Plot the oringinal faces and the reconstructed faces
% for i = 1:5
%    subplot(2,5,i)
%    imshow(uint8(floor(m2f(res(i,:)))));
%    subplot(2,5,i+5)
%    imshow(uint8(m2f(img_test(i,:))));
% end

% Plot test error k = 1:100
rec_f = zeros(27,256^2);
test_err = [];
for k = 1:100
    % Project&Reconstruct test images on k=10 eigen-faces
    rec_f = rec_f+test_m_w*img_v(:,k)*img_v(:,k)';
    % Step3: warp the reconstructed face to the reconstructed landmarks
    for i = 1:27
       temp = m2f(rec_f(i,:)+img_mx);
       res(i,:) = f2m(double(warpImage_kent(temp,[l_mx(1,1:87)',l_mx(1,88:174)'],[rec_l(i,1:87)'+l_mx(1,1:87)',rec_l(i,88:174)'+l_mx(1,88:174)'])));
    end
    err = sum(sum(abs(rec_f-double(img_test_m))))/(27*256^2);
    disp(sprintf('%d eigen-faces error is %d ',k,err));
    test_err = [test_err,err];
end
plot([1:100],test_err)
xlabel('Num of eigen-faces')
ylabel('Test error L1 norm')
title('Reconstruction error in warped face');


% Synthesize 20 random faces
% for i = 1:20
%     % Sampling on the appearance
%     r_x = img_mx+ (rand(1,10))*19^-4.*diag(img_d(1:10,1:10))'*img_v(:,1:10)';
%     r_l = l_mx+ (rand(1,10))*10^-3.*diag(l_d(1:10,1:10))'*l_v(:,1:10)';
%     r_face = warpImage_kent(uint8(m2f(r_x)),[l_mx(1,1:87)',l_mx(1,88:174)'],[r_l(1,1:87)',r_l(1,88:174)']);
%     subplot(4,5,i)
%     imshow(uint8(r_face));
% end

end

% Help functions
% Get face of [256,256]
function face = m2f(m)
face = zeros(256,256);
for i = 1:256
     face(i,:)= m(1,((i-1)*256+1):(i*256));
end
end

function m = f2m(face)
m = [];
for i = 1:256
    m = [m,face(i,:)];
end
end