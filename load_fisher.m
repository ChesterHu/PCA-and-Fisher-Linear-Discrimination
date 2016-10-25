function [m_face_train,m_face_test,f_face_train,f_face_test,m_mx,f_mx] = load_fisher()
% Load data that are centerd by male and female mean face.
m_face_train = [];
m_face_test = [];
f_face_train = [];
f_face_test = [];
m_mx = zeros(256^2,1);
f_mx = zeros(256^2,1);

% Male face
for i = 1:89
    if i == 58
        continue
    end
    path = sprintf('./face_data/male_face/face%03d.bmp',i-1);
    temp = f2m(imread(path));
    if i<=79
       m_face_train = [m_face_train,temp];
       m_mx = m_mx+double(temp);
    else
       m_face_test = [m_face_test,temp];
    end
end

m_mx = m_mx/78;

% Female face
for i = 1:85
   path = sprintf('./face_data/female_face/face%03d.bmp',i-1);
   temp = f2m(imread(path));
   if i <= 75
      f_face_train = [f_face_train,temp];
      f_mx = f_mx+double(temp);
   else
       f_face_test = [f_face_test,temp];
   end
end
f_mx = f_mx/75;
end

% Problem 6, all faces are aligned, 
function [m_face_train,m_face_test,m_land_train,m_land_test,f_face_train,f_face_test,f_land_train,f_land_test,m_mx,f_mx]=load_fisher_align()
m_face_train = [];
m_face_test = [];
f_face_train = [];
f_face_test = [];
m_land_train = [];
m_land_test = [];
f_land_train = [];
f_land_test = [];
m_mx = zeros(256^2,1);
f_mx = zeros(256^2,1);

% Male landmark
for i = 1:89
    if i == 58
        continue
    end
    land_path = sprintf('./face_data/male_landmark_87/face%03_87pt.txt',i-1);
    land_temp = textread(land_path);
    land_temp = [land_temp(:,1);land_temp(:,2)];
    if i<=79
       m_land_train = [m_land_train,land_temp];
    else
       m_land_test = [m_land_test,land_temp];
    end
end
% Female landmark
for i = 1:85
   land_path = sprintf('./face_data/female_landmark_87/face%03_87pt.txt',i-1);
   land_temp = textread(land_path);
   land_temp = [land_temp(:,1);land_temp(:,2)];
   if i <= 75
      f_land_train = [f_land_train,land_temp];
   else
      f_land_test = [f_land_test,land_temp];
   end
end
% average landmark
m_lx = (sum(m_land_train,2)+sum(f_land_train,2))/153;

% Male face
for i = 1:89
    if i == 58
        continue
    end
    path = sprintf('./face_data/male_face/face%03d.bmp',i-1);
    if i<=79
        % Warpping face
        j = i;
        if j == 59; 
            j = i-1;
        end
       temp = f2m(warpImage_kent(imread(path),[m_land_train(1:87,j),m_land_train(88:174,j)],[m_lx(1:87,1),m_lx(88:174,1)]));
       m_face_train = [m_face_train,temp];
       m_mx = m_mx+double(temp);
    else
       temp = f2m(warpImage_kent(imread(path),[m_land_test(1:87,i-79),m_land_test(88:174,i-79)],[m_lx(1:87,1),m_lx(88:174,1)]));
       m_face_test = [m_face_test,temp];
    end
end
m_mx = m_mx/78;

% Female face
for i = 1:85
   path = sprintf('./face_data/female_face/face%03d.bmp',i-1);
   if i <= 75
      temp = f2m(warpImage_kent(imread(path),[f_land_train(1:87,i),f_land_train(88:174,i)],[m_lx(1:87,1),m_lx(88:174,1)]));
      f_face_train = [f_face_train,temp];
      f_mx = f_mx+double(temp);
   else
      temp = f2m(warpImage_kent(imread(path),[f_land_test(1:87,i-75),f_land_test(88:174,i-75)],[m_lx(1:87,1),m_lx(88:174,1)]));
      f_face_test = [f_face_test,temp];
   end
end
f_mx = f_mx/75;
end

% Help function, 256*256->256^2*1
function m = f2m(f)
m = [];
for i = 1:256
   m = [m;f(:,i)]; 
end
end

