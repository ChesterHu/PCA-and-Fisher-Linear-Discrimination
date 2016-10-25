function [train, test, m_x] = load_data()
% Use 150 images as training set, 27 images as testing set.
m_x = zeros(1,256^2);
for i = 1:178
    % The 103 obs is missing.
    if i==104
        impath = ['./face_data/face/face',sprintf('%03d.bmp',104)];
    else
        impath = ['./face_data/face/face',sprintf('%03d.bmp',i-1)];
    end
    face = imread(impath);
    
    % Input image matrix as an array of obs (by column).
    obs = f2m(face);
    % Split obs into train and test.
    if i>151
        test((i-151),:)=obs;
    else
        train(i,:)=obs;
        m_x = m_x+double(obs);
    end
end
train(104,:)=[];
m_x = m_x/150;
end



%Help fuctions

% Matrix to face
% Get matrix from face
% Read the face by column.
function m = f2m(face)
m = [];
for i = 1:256
    m = [m,face(i,:)];
end
end
