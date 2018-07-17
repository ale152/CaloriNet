clear

% Input folder
folder_allcases = 'F:\SPHERE\SPHERE_Calorie';
folder_boxes = 'F:\SPHERE\SPHERE_Calorie';
folder_rgb = 'rgb';
folder_dep = 'depth';
folder_skel = 'skeleton';
folder_silhouette = 'silhouette';
box_filename = 'userBB*.txt';
dep_filename_dir = 'depth_*.png';
dep_filename = 'depth_%d.png';
rgb_filename_dir = 'img_*.png';
rgb_filename = 'img_%d.png';
sil_filename = 'sil_%d.png';
skel_filename = '%s_keypoints.json';

% Settings
Set.save_silhouettes = true;
Set.debug.video = true;
Set.debug.video_folder = 'C:\Data\calories_test';
Set.debug.onscreen = true;
% List all the cases
cases = dir(fullfile(folder_allcases,'Subject*'));
Ncases = numel(cases);
% Skeleton pairs
pairs = 1+[1 2; 1 5; 2 3; 3 4; 5 6; 6 7; 1 8; 8 9; 9 10; 1 11;11 12; 12 13;1 0];

if Set.debug.onscreen
    figure,him = image; axis ij equal
end

warning('off','stats:kmeans:FailedToConverge')
% Find the silhouettes for all the cases
parfor ci = 1:Ncases
    disp(['Processing: ',fullfile(folder_allcases,cases(ci).name,folder_rgb,rgb_filename_dir),'...'])
    case_name = cases(ci).name;
    % List all rgb images. Depth will be read afterwards
    rgb_list = dir(fullfile(folder_allcases,cases(ci).name,folder_rgb,rgb_filename_dir));
    Nfiles = numel(rgb_list);
    
    % Order the rgb list
    ord = zeros(size(rgb_list));
    for ii = 1:numel(rgb_list)
        ord(ii) = sscanf(rgb_list(ii).name,rgb_filename);
    end
    [~,ord] = sort(ord);
    rgb_list = rgb_list(ord);
    
    % Read the bounding box
    try
        box_list = dir(fullfile(folder_boxes,cases(ci).name,box_filename));
        boxes = [];
        for bli = 1:numel(box_list)
            boxes = cat(1,boxes,dlmread(fullfile(folder_boxes,cases(ci).name,box_list(bli).name)));
        end
        % Keep one box only
        [~,boxid] = unique(boxes(:,1));
        boxes = boxes(boxid,:);
    catch
        disp(ci)
        disp 'Bounding box not found!'
        continue
    end
    
    kmeans_exist = false;
    
    % Loop over the sequence of images
    tic
    initialise = true;
    for fi = 1:Nfiles
        % Find image id from the file name
        img_id = sscanf(rgb_list(fi).name,rgb_filename);
        
        % Find the specific bounding box
        box = boxes(boxes(:,1) == img_id,[10 11 12 13]);
        if isempty(box)
            % No bounding box found, no silhouette
            continue
        else
            % Rearrange bb into [x y w h] format
            hw = box(1:2)-box(3:4);
            box(1:2) = box(1:2)-hw(1:2);
            box(3:4) = hw;
        end
        
        % Read aligned rgb and depth images
        try
            img = imread(fullfile(rgb_list(fi).folder,rgb_list(fi).name));
            dep = imread(fullfile(folder_allcases,case_name,folder_dep,sprintf(dep_filename,img_id)));
        catch msgerr
            fprintf('Unable to read image %d (%d) because of %s\n',fi,Nfiles,msgerr.message)
            continue
        end
            
        dep = double(dep);
        rnd = rand(size(dep))*max(dep(:));
        dep(dep == 0) = rnd(dep == 0);
        
        % Read skeleton
        sk_path = fullfile(folder_allcases,case_name,folder_skel, ...
            sprintf(skel_filename,rgb_list(fi).name(1:end-4)));
        json = jsondecode(fileread(sk_path));
        % Find the right skeleton (the one within the bbox)
        bbcen = [box(1)+box(3)/2, box(2)+box(4)/2];
        dist_bb = zeros(1,numel(json.people));
        if numel(json.people) == 0
            % Openpose didn't find any person in the frame, skip it
            continue
        end
        for ppi = 1:numel(json.people)
            joints = reshape(json.people(ppi).pose_keypoints,3,[])';
            dist_bb(ppi) = mean(sum((bsxfun(@minus,joints(:,1:2),bbcen)).^2,2));
        end
        [~,ppi] = min(dist_bb);
        joints = reshape(json.people(ppi).pose_keypoints,3,[])';
        % Remove broken joints
        pairs_subj = pairs;
        missing_joints = find(all(joints==0,2));
        % If a joint is missing, all the bones connected to that joint must
        % be deleted
        pairs_to_remove = any(ismember(pairs_subj,missing_joints),2);
        pairs_subj(pairs_to_remove,:) = [];
        joints_subj = joints;
        joints_subj(all(joints==0,2),:) = [];
        % Pick the depth of the joints
        jx = max(1,min(size(dep,2),round(joints_subj(:,2))));
        jy = max(1,min(size(dep,1),round(joints_subj(:,1))));
        ind = sub2ind(size(dep),jx,jy);
        [zed,inds] = sort(dep(round(ind)));
        
        %%%%%
        % Use kmeans on the RGB-D data and keep the clusters that intersect
        % with the skeleton
        
        % Skeleton
        skel = zeros(size(dep),'uint8');
        points = [joints(pairs_subj(:,1),1) joints(pairs_subj(:,1),2) joints(pairs_subj(:,2),1) joints(pairs_subj(:,2),2)];
        skel = insertShape(skel,'line',points,'Opacity',1,'SmoothEdges',false,'Color','w','LineWidth',1);
        skel = skel(:,:,1);
        [skel_y,skel_x] = ind2sub(size(skel),find(skel));
        skel_x = skel_x - box(1) + 1;
        skel_y = skel_y - box(2) + 1;
        skel_box = skel(box(2):box(2)+box(4),box(1):box(1)+box(3));
        if ~any(skel_box(:))
            % The skeleton was not in the bounding box, skip frame
            continue
        end
        skel_dist = bwdist(skel_box);
        skel_dist = skel_dist(:)./max(skel_dist(:));
        
        img_box = double(img(box(2):box(2)+box(4),box(1):box(1)+box(3),:))/255;
        img_box = cell2mat(cellfun(@histeq,mat2cell(img_box,size(img_box,1),size(img_box,2),[1 1 1]),'uniformoutput',false));
        img_box = imgaussfilt(img_box,1);
        dep_box = histeq(imgaussfilt(normal(dep(box(2):box(2)+box(4),box(1):box(1)+box(3))),1));
        %img_lab = rgb2lab(img_box);
        %img_lab = img_lab(:,:,1);
        img_ft = reshape(double(img_box),[],3);
        [x,y] = meshgrid(1:size(img_box,2),1:size(img_box,1));
        x = x(:)/size(img_box,2)-0.5;
        y = y(:)/size(img_box,1)-0.5;
        ft = cat(2,img_ft,dep_box(:),skel_dist);
        
        Nclass = 6; %round(size(joints,1));
        if ~kmeans_exist
            [classif,kcen] = kmeans(ft,Nclass,'maxiter',10000,'OnlinePhase','off','options',struct('Display','off'));
            kmeans_exist = true;
        else
            [classif,kcen] = kmeans(ft,Nclass,'maxiter',10000,'OnlinePhase','off','options',struct('Display','off'),'Start',kcen);
        end
        classif = reshape(classif,size(dep_box,1),size(dep_box,2));
        
        % Select clusters with skeleton
        mask = zeros(size(dep_box));
        for cli = 1:Nclass
            BW = classif == cli;
            mask = max(mask,bwselect(BW,skel_x,skel_y,4));
        end
        
        % Cut areas too far
        THR = round(0.5*mean(sqrt((points(:,3)-points(:,1)).^2 + (points(:,4)-points(:,2)).^2)));
        skel_exp = imdilate(skel_box,strel('disk',THR,0));
        mask = mask & double(skel_exp);
        mask = imclose(mask,strel('disk',2,0));
        %mask = imfill(mask,'holes');
        
        img_vid = img;
        mask_vid = false(size(dep));
        mask_vid(box(2):box(2)+box(4),box(1):box(1)+box(3)) = mask;
        img_vid(repmat(~mask_vid,1,1,3)) = 0;
        bff = repmat(~mask_vid,1,1,3);
        bff(:,:,1:2) = 0;
        img_vid(bff) = 255;
        %%%%%
        
%         if Set.debug.onscreen
%             him.CData = img_vid;
%             drawnow
%         end
        
        % Initialise variables for the whole sequence
        if initialise
            initialise = false;
            if Set.save_silhouettes
                if ~isdir(fullfile(folder_allcases,cases(ci).name,folder_silhouette))
                    mkdir(fullfile(folder_allcases,cases(ci).name,folder_silhouette))
                end
            end
            if Set.debug.video
                if ~isdir(fullfile(Set.debug.video_folder))
                    mkdir(fullfile(Set.debug.video_folder))
                end
                Video = VideoWriter(fullfile(Set.debug.video_folder,case_name),'MPEG-4'); %#ok<TNMLP>
                Video.open();
            end
        end
        
        % Save the video
        if Set.save_silhouettes
            filename = fullfile(folder_allcases,cases(ci).name,folder_silhouette, ...
                sprintf(sil_filename,img_id));
            imwrite(mask_vid,filename)
        end
        if Set.debug.video
            Video.writeVideo(img_vid);
        end
    end
end
    
if Set.debug.video
    Video.close();
    disp(['Done with: ',fullfile(folder_allcases,cases(ci).name,folder_rgb,rgb_filename_dir),'...'])
end
    
