function [] = eig_face(k)
% Will return mean face and first 20 eigen faces.
[train,test,m_x] = load_data();

% W*u = lambda*u, where W = train*train'. dim(W) = [150,150].
% train*train'*u = lambda*u => train'train*train'*u = lambda*train'*u
% train'train is the desired cov matrix
for i = 1:177
    if i>150
       test_m(i-150,:) = double(test(i-150,:))-m_x; 
    else
       train_m(i,:) = double(train(i,:))-m_x;
    end
end

[u,~] = eigs(train_m*train_m',149);

% Is the eigen face
% dim(v) = [256^2,150]
v = train_m'*u; 
% need rescale v.
for i = 1:149
   v(:,i) = v(:,i)/norm(v(:,i)); 
end

% Show eigen faces
% Variance decrease as the eigen value decrease.
% alpha = 2.6*10^4;
% for i = 1:20
%    subplot(4,5,i);
%    imshow(uint8(m2f(m_x+v(:,i)'*alpha)));
%    alpha = alpha*0.9;
% end

% Projection & Reconstruction
% a = test_m*v(:,1:k);
% r_x = decode(m_x,a,v);
% 
% for i = 1:5
%    subplot(2,5,i)
%    imshow(uint8(m2f(r_x(i,:))))
%    subplot(2,5,i+5)
%    imshow(uint8(m2f(test(i,:))));
% end


% Compute test error
test_err = [];
r_x = zeros(27,256^2);
for i = 1:148
    % Decoding
    r_x = r_x + test_m*v(:,i)*v(:,i)';
    test_err = [test_err,sum(sum(double(abs(floor(test_m)-floor(r_x)))))/(256^2*27)];
end
plot(1:148,test_err);
xlabel('Num of eigen vectors');
ylabel('Test error L1 norm');
title('Reconstruction error of unalignment face');
end

% Help functions
% Get face of [256,256]
function face = m2f(m)
face = zeros(256,256);
for i = 1:256
     face(i,:)= m(1,((i-1)*256+1):(i*256));
end
end
% Get matrix from face.
function m = f2m(face)
m = [];
for i = 1:256
    m = [m,face(i,:)];
end
end

