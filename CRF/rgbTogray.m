filename = dir('C:\Users\25162\Desktop\CRF 调参\sil\*.jpg');
len = length(filename);
for i = 1: len
    rgb_img = imread(strcat('C:\Users\25162\Desktop\CRF 调参\sil\',filename(i).name));
    gray_img = rgb2gray(rgb_img);
    gray_img(gray_img<128)=0;
    gray_img(gray_img>=128)=1;
    name = filename(i).name;
    name = strrep(name,'jpg','png');
    imwrite(gray_img,strcat('C:\Users\25162\Desktop\CRF 调参\sli1\',name));
end
