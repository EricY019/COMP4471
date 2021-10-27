clear;
%% 利用MD5去重 Use MD5 to remove repetition
% str,C 第一列为文件名，第二列为MD5校验码
index = 0;
str = strings(1000,2);
while true
    filename = strcat( '.\gifs\', num2str(index), '.gif' );
    % 如果不存在此文件，则退出
    if ~isfile( filename )
        break;
    end
    % 动态申请空间
    % Dynamically allocate the space
    if mod( index - 1000, 1000 ) == 0 && index > 0
        str = [str; strings(1000,2)];
    end
    str{index+1,1} = filename;
    % 计算文件校验码MD5
    % Calculate Checksum MD5
    str{index+1,2} = Simulink.getFileChecksum( str{index+1,1} );
    index = index + 1;
end
[C, ia, ic] = unique( str(1:index,2) );
C = [str(ia, 1) C];
if size(C,1) == size(str,1)
    disp("没有发现重复数据 No repetition is detected");
end

%% 转换为 PNG 图片 Transform to PNG image
% 每一千张图片大约需要1.5分钟
% 如果文件夹不存在，则创建文件夹
myfolder = ".\pngs";
if size(C,1) > 0 && exist(myfolder,'dir')==0
    mkdir(myfolder);
end
for image_idx = 1:size(C,1)
    [img,map] = imread( C{image_idx,1}, 'frames', 'all' );
    for frame_idx = 1:size(img,4)
        filename = ['.\pngs\', num2str(image_idx), '_', num2str(frame_idx), '.png'];
        imwrite(img(:, :, 1, frame_idx), map, filename);
    end
end

% %% 转成9*3的图片方便标注
% % 如果文件夹不存在，则创建文件夹
% myfolder = ".\labels";
% if size(C,1) > 0 && exist(myfolder,'dir')==0
%     mkdir(myfolder);
% end
% for image_idx = 1:size(C,1)
%     [img,map] = imread( C{image_idx,1}, 'frames', 'all' );
%     bigimage = imtile( img, map, 'GridSize', [9 3]);
%     filename = ['.\labels\', num2str(image_idx), '.png'];
%     imwrite( bigimage, filename );
% end