clear all;
fontslist = string(listTrueTypeFonts);
fontslist_exclude = ["MT Extra" "HGSXT_CNKI" "Bookshelf Symbol 7" "HGOCR_CNKI" "Holo MDL2 Assets" "MS Outlook" "MS Reference Specialty" "Segoe MDL2 Assets" "Symbol" "TeamViewer15" "Webdings" "Wingdings" "Wingdings 3" "Wingdings 2"];
fontslist = setdiff( fontslist, fontslist_exclude );
captcha_num = 20000;
frame_num = 25;
height = 70;
width = 250;
characters_list = strings(captcha_num,1);
for idx = 2001:captcha_num
    font_id = randsample(size(fontslist,1),1, true);
    frames = uint8(zeros(height,width,3,frame_num));
    char_num = 4; %randsample(4:6, 1);
    colors = reshape(randsample(100/255:1/255:1,3*char_num, true), char_num, 3);
    % 数字48:57 大写字母65:90 小写字母97:122 0:48 O:79 1:49 I:73 l:108
    characters_list(idx,1) = char(randsample([48:57 65:72 74:78 80:90], char_num, true ));
    characters = char( characters_list(idx,1) )
    rndMtn = reshape( randsample(-0.2:0.01:0.2, 6*char_num, true), char_num, 6);
    rndMtn(:, [1 4]) = rndMtn(:, [1 4]) + 1;
    rndMtn(:, [5 6]) = round(rndMtn(:, [5 6])*25);
    line_num = randsample(10:20, 1);
    color_line = reshape( randsample(1:255, line_num*3, true), line_num, 3 );
    line_position = zeros(line_num, 4 );
    line_position(:, [1 3]) = reshape( randsample([0:width], line_num*2, true), line_num, 2 );
    line_position(:, [2 4]) = reshape( randsample([-height/4:height*1.5], line_num*2, true), line_num, 2 );
    % Add Circles
    circle_num = randsample(20:40, 1);
    circle_position = zeros(circle_num, 3 );
    circle_position(:, 1) = reshape( randsample(width, circle_num, true), circle_num, 1 );
    circle_position(:, 2) = reshape( randsample(height, circle_num, true), circle_num, 1 );
    circle_position(:, 3) = reshape( randsample([zeros(1,20) 1:10], circle_num, true), circle_num, 1 );
    circle_color_ori = reshape( randsample(1:255, circle_num*3, true), circle_num, 3 );
    %% Add Masking
    mask = ones(frame_num, char_num);
    mask_idx = randsample(frame_num, 1, true);
    mask_sequence = repmat( [0.9 0.5 0.1 zeros(1,3) 0.1 0.5 0.9 ones(1,8)], 1, 3);
    if mask_idx > frame_num/2
        mask_step = randsample(-4:-2, 1);
    else
        mask_step = randsample(2:4, 1);
    end
    for char_idx = 1:char_num
        left = max(mask_idx-25,1);
        right = min(mask_idx+25,frame_num);
        mask( left:right, char_idx ) = mask_sequence( 26-(mask_idx-left):26-mask_idx + right ).';
        mask_idx = mask_idx + mask_step;
    end
    %%
    for frame_idx = 1:frame_num
        img = uint8( zeros(height, 1, 3) );
        % Add lines
        line_position = line_position + reshape( randsample(-6:6, line_num*4, true), line_num, 4 );
        frames(:, : ,: ,frame_idx) = insertShape( frames(:, : ,: ,frame_idx), 'Line', line_position, 'Color', color_line, 'LineWidth', 3);
        % Add Characters
        rndMtn(:,1:4) = rndMtn(:,1:4) + reshape( randsample(-0.02:0.01:0.02, 4*char_num, true), char_num, 4);
        rndMtn(:,5:6) = rndMtn(:,5:6) + reshape( randsample(-2:2, 2*char_num, true), char_num, 2);
        rndMtn(:,5:6) = max( min(rndMtn(:,5:6),15), -15);
        for char_idx = 1:char_num
            character = uint8(zeros(160,160,3));
            character = insertText(character, [80,80], characters(char_idx), 'FontSize', 40, 'Font', fontslist(font_id) ,'BoxOpacity', 0, 'TextColor', 'white', 'AnchorPoint', 'Center');
            tform = affine2d( [rndMtn(char_idx, 1) 2*rndMtn(char_idx, 2) 0; 2*rndMtn(char_idx, 3) rndMtn(char_idx, 4) 0; 0 0 1  ] );
%                     imshow(character);
            sameAsInput = affineOutputView(size(character),tform,'BoundsStyle','CenterOutput');
            character = imwarp( character, tform,'OutputView',sameAsInput );
%                     imshow(character);
            xrange = floor( size(character,1)/2 ) - ceil(height/2)+1: floor( size(character,1)/2 ) + floor(height/2);
            xrange = xrange +  rndMtn(char_idx,6);
            yrange = floor( find( character(:,:,1), 1)/size(character,1) - 1 ): ceil( find( character(:,:,1), 1, 'last' )/size(character,1) + 1 );
            character = character( xrange, yrange, :  );
            %%
            colormap = repmat( reshape(colors(char_idx, :)* mask(frame_idx,char_idx) , [1 1 3]), height, size(character,2), 1);
            character = uint8( double(character) .* colormap );
            img = [img character];
        end
        img = [img uint8( zeros(height, 1, 3) )];
        % 如果拼接后尺寸过大，就缩小尺寸
        if size(img,2) > width
            img = imresize(img, [size(img,1) width]);
        end
        if size(img,1) > height
            img = imresize(img, [height size(img,2)]);
        end
        temp = frames(:, max(1,floor((width-size(img,2))/2)): min(width ,floor((width-size(img,2))/2 + size(img,2))),: ,frame_idx);
        img_idx = find(img);
        temp(img_idx) = img(img_idx);
        frames(:, max(1,floor((width-size(img,2))/2)): min(width ,floor((width-size(img,2))/2 + size(img,2))),: ,frame_idx) = temp;
        % Add Circles
        circle_position(:, [1, 2]) = circle_position(:, [1, 2]) + reshape( randsample(-10:10, circle_num*2, true), circle_num, 2 );
        circle_position(:, 3) = max( circle_position(:, 3) + reshape( randsample([-1:1 0 0 0], circle_num, true), circle_num, 1 ), 0);
        circle_color = circle_color_ori.* repmat( randsample([0 0 0 1 1 1 1], circle_num, true),3,1).';
        frames(:, : ,: ,frame_idx) = insertShape( frames(:, : ,: ,frame_idx), 'Circle', circle_position, 'Color', circle_color, 'LineWidth', 2);
        %     imshow(img);
%                 imshow(frames(:, : ,: ,frame_idx));
%                 pause(0.1);
    end
    [frames_ind, map] = rgb2ind(imtile(frames, 'GridSize', [frame_num,1]), 256);
    filename = strcat('.\dataset\', num2str(idx),  '.gif');
    for frame_idx = 1:frame_num
        if frame_idx == 1
            imwrite(frames_ind((frame_idx-1)*height+1:frame_idx*height, :, :), map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
        else
            imwrite(frames_ind((frame_idx-1)*height+1:frame_idx*height, :, :), map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
    if( mod(idx,500)==0)
        writetable( table(characters_list), 'labels_test.csv', 'WriteVariableNames', false );
    end
end
writetable( table(characters_list), 'labels_test.csv', 'WriteVariableNames', false );


