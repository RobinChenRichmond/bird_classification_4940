function autocrop

%This function crop the images in the dataset automatically by first removing
%their backgrounds and then locating the largest part in the image

%choose the path of the folder
folders = dir('/Users/guanyuchen/Desktop/Github/bird_classification_4940/bird_dataset');

nfolders = length(folders);

%browse all the folders in the directory
for aa = 3:nfolders
    currentfoldername = folders(aa).name;
    fprintf('Now begin the folder %d.   \n' , ...
            currentfoldername);
    
    %browse all the images in each folder
    files = dir(fullfile('/Users/guanyuchen/Desktop/Github/bird_classification_4940/bird_dataset',currentfoldername,'*.jpg'));
    nfiles = length(files);

    %browse all the images in each folder
    for ii = 1:nfiles
        currentfilename = files(ii).name;
        A = imread(fullfile('/Users/guanyuchen/Desktop/Github/bird_classification_4940/bird_dataset', currentfoldername, currentfilename)) ;
        
        
        if strcmp(currentfoldername,'EmptyFeeder') == 0
            %set some parameters
            [p,q,r] = size(A);

            %cut off the green pixels
            for i = 1:p
                for j = 1:q
                    if A(i,j,2)>A(i,j,1)*1.5 & A(i,j,2)>A(i,j,3)*0.95
                        A(i,j,:) = 255;
                    end

                    if A(i,j,2)>A(i,j,1)*1.3 & A(i,j,2)>A(i,j,3)*1
                        A(i,j,:) = 255;
                    end

                    if A(i,j,2)>A(i,j,1)*1.1 & A(i,j,2)>A(i,j,3)*1.1
                        A(i,j,:) = 255;
                    end

                    %delete nearly white and light green
                    if A(i,j,2) > 240 & A(i,j,3) > 240
                        if A(i,j,1) > 160 & A(i,j,1)<240
                            A(i,j,:) = 255;
                        end
                    end
                    
                end
            end

            %find the largest area with binary image
            B = uint8(255) - A;
            B = im2bw(B,0.1);
            B = bwareafilt(B,1);

            %initialize the value of the boundary
            xmin = 0;
            xmax = 0;
            ymin = 0;
            ymax = 0;

            for i = 1:p
                for j = 1:q
                    if B(i,j) == 1
                        ymin = i;
                        break
                    end
                end
                if ymin~= 0
                    break
                end
            end

            for j = 1:q
                for i = 1:p
                    if B(i,j) == 1
                        xmin = j;
                        break
                    end
                end
                if xmin~= 0
                    break
                end
            end

            i = p;
            while i > 1
                for j = 1:q
                    if B(i,j) == 1
                        ymax = i;
                        break
                    end
                end
                if ymax~= 0
                    break
                end
                i = i - 1;
            end

            j = q;
            while j > 1
                for i = 1:p
                    if B(i,j) == 1
                        xmax = j;
                        break
                    end
                end
                if xmax~= 0
                    break
                end
                j = j - 1;
            end

            wid = xmax - xmin;
            leng = ymax - ymin;

            A = imcrop(A,[xmin ymin wid leng]);
            A = imresize(A,[64,64]);
        end
        
        figure(1);imshow(A);
        [xvalue,yvalue,zvalue] = size(A);
        
        filepath = fullfile('/Users/guanyuchen/Desktop/Github/bird_classification_4940/Result/',currentfoldername);
        
        %if you want to save the images in 900 by 1600 pixels
        %saveas(gca,fullfile(filepath,num2str(ii)),'jpg');
        
        %if you want to save the images in their original pixels
        set(gcf,'resize','off');
        set(gcf,'position',[0,0,yvalue,xvalue]);
        frm = getframe(gcf);
        imwrite(frm.cdata,fullfile(filepath,strcat(num2str(ii),'.jpg'))); 
        
        
        fprintf('image %d cropped.   \n' , ...
            ii);

    end
end


end
