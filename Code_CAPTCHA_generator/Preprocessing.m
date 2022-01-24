clear;
% 需要手动新建文件夹
%% 利用MD5去重 Use MD5 to remove repetition
% str,C 第一列为文件名，第二列为MD5校验码
str = strings(0,4);
index = 1;
data = readtable('.\dataset-2000\labels-2000.csv', 'ReadVariableNames', false);
label_2000 = string( data{:,2} );
data = readtable('MATLAB动态验证码实现\labels_test.csv', 'ReadVariableNames', false);
label = string( data{:,:} );
while index <=4000
    if index <= 2000
        filename = strcat( '.\gifs\', num2str(index), '.gif' );
    else
        filename = strcat( '.\MATLAB动态验证码实现\dataset\', num2str(index), '.gif' );
    end
    % 如果不存在此文件，则退出
    if ~isfile( filename )
        break;
    end
    % 动态申请空间
    % Dynamically allocate the space
    if mod( index - 1000, 1000 ) == 1 && index > 0
        str = [str; strings(1000,4)];
    end
    str{index,1} = filename;
    % 计算文件校验码MD5
    % Calculate Checksum MD5
    str{index,2} = Simulink.getFileChecksum( str{index,1} );
    str{index,3} = num2str(index);
    if index <= 2000
        str{index,4} = label_2000{index, 1};
    else
        str{index,4} = label{index-2000, 1};
    end
    index = index + 1;
end
[C, ia, ic] = unique( str(1:index-1,2) );
% C = [str(ia, 1) C];
if size(C,1) == index-1
    disp("没有发现重复数据 No repetition is detected");
    C = str(1:index-1, :);
else
    return;
end

%% 转换为 PNG 图片 Transform to PNG image
% 每一千张图片大约需要1.5分钟
% 如果文件夹不存在，则创建文件夹
myfolder = ".\pngs";
if size(C,1) > 0 && exist(myfolder,'dir')==0
    mkdir(myfolder);
end
% 这里五个为一组进行重排，所以要求数据集的大小必须是5的倍数
rearranged_str = strings(size(C));
for image_idx = 1:2000%size(C,1)
    image_idx = image_idx
    rearranged_idx = image_idx+2000
%     rearranged_idx = size(C,1)/2 * mod(image_idx-1,2) + ceil(image_idx/2);
    [img,map] = imread( C{rearranged_idx,1}, 'frames', 'all' );
    rearranged_str(image_idx,3) = num2str(image_idx);
    rearranged_str(image_idx,4) = C(rearranged_idx,4);
    filename_gif = strcat('.\dataset-2000\gifs\', num2str(image_idx),  '.gif');
    for frame_idx = 1:size(img,4)
        filename = ['.\dataset-2000\pngs\', num2str(image_idx), '_', num2str(frame_idx), '.png'];
        %% 存储pngs
        imwrite(img(:, :, 1, frame_idx), map, filename);
%         %% 存储gif
%         if frame_idx == 1
%             imwrite(img(:, :, 1, frame_idx), map, filename_gif, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
%         else
%             imwrite(img(:, :, 1, frame_idx), map, filename_gif, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%         end
    end
end
%%
data = [table(rearranged_str(:,3), rearranged_str(:,4))];
writetable( data, '.\dataset-2000\labels.csv', 'WriteVariableNames', false );

% %% 转成9*3的图片方便标注
% % 如果文件夹不存在，则创建文件夹
% myfolder = ".\labels";
% if size(C,1) > 0 && exist(myfolder,'dir')==0
%     mkdir(myfolder);
% end
% for image_idx = 1:size(C,1)
%     [img,map] = imread( C{image_idx,1}, 'frames', 'all' );
%     bigimage = imtile( img, map, 'GridSize', [5 5]);
%     filename = ['.\labels\', num2str(image_idx), '.png'];
%     imwrite( bigimage, filename );
% end