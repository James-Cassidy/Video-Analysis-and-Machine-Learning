function D = NMS(B,threshold)

%% nms version 2
% initialise empty array D
D = [];

% loop through all the bounding boxes B
for i = 1:size(B)
    % Set discard flag to false to indicate if boundingBox(i) should be
    % discarded
    discard = 0;

    % loop through again to compare with boundingBox(i)
    for j = 1:size(B)
        % Calculate the intersection over union IoU
        intersection = rectint(B(i,1:4), B(j,1:4));
        area_i = B(i,3) * B(i,4);
        area_j = B(j,3) * B(j,4);
        union = area_i + area_j - intersection;
        IoU = intersection / union;

        % if IoU of bounding boxes i and j is greater than the threshold
        if IoU > threshold
            % then discard boundingBox(i) if its score is lower than
            % boundingBox(j)
            if B(j,5) > B(i,5)
                discard = 1;
            end
        end
    end

    % if discard flag is set to false i.e. boundingBox(i) is not to be
    % discarded then add it to the output array D
    if discard == 0
        D = [D; B(i,:)];
    end
end


%% nms version 1
% % for every detected_bounding_box:
% for i = 1:size(Objects,1)
% %     if i > (size(Objects,1))
% %             break;
% %     end
%     % calculate the intersection_area with any other detected_bounding_box
%     for j = 1:(size(Objects,1)-1)
%     
% %         i
% %         j
% %         if j+1 > (size(Objects,1)-1)
% %             break;
% %         end
% 
%         intersection_area = rectint(Objects(i,1:4), Objects(j+1,1:4));
%         bounding_box_area = Objects(j+1,3) * Objects(j+1,4); % Multiply bounding box j's width and height
% 
%         % if (intersection_area / bounding_box_area ) is > threshold
%         if intersection_area / bounding_box_area > threshold
% 
%             % remove one of the intersecting bounding box (the one with smaller confidence)
%             if Objects(i,5) < Objects(j+1,5)
%                 Objects(i,:) = [];
%             else
%                 Objects(j+1,:) = [];
%             end
%         end
%         if j+1 > (size(Objects,1)-1)
%             break;
%         end
%     end
%      if i == (size(Objects,1))
%             break;
%     end
% 
% end 
    

end