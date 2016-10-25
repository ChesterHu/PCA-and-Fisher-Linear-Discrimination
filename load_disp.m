function [train,test,m_l] = load_disp()
m_l = zeros(1,174);
for i = 1:178
    if i == 104
        path = sprintf('./face_data/landmark_87/face%03d_87pt.dat',104);
    else
        path = sprintf('./face_data/landmark_87/face%03d_87pt.dat',i-1);
    end
    data = textread(path);
    
    if i>151
        test(i-151,:) = [data(2:88,1)',data(2:88,2)'];
    else
        train(i,:) = [data(2:88,1)',data(2:88,2)'];
        m_l = m_l+train(i,:);
    end
end

train(104,:)=[];
m_l = m_l/150;
end