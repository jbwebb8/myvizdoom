% ----------------------
% Plot agent positions
% ----------------------

function [] = plot_agent(varargin)
    
    p = inputParser;
    addOptional(p, 'plotPosition', true);
    addOptional(p, 'plotAction', true);
    parse(p, varargin{:});
    plotPosition = p.Results.plotPosition;
    plotAction = p.Results.plotAction;
    
    if (plotPosition && ~plotAction)
        [posFileName, posPathName] = uigetfile('', 'Choose position file');
        positions = load(posPathName + posFileName);
        plot(positions([2,3]));
    
    elseif (~plotPosition && plotAction)
        [actFileName, actPathName] = uigetfile('', 'Choose action file');
        actions = load(strcat(actPathName,actFileName));
    
    elseif (plotPosition && plotAction)
        [posFileName, posPathName] = uigetfile('', 'Choose position file');
        positions = load(strcat(posPathName,posFileName));
        [actFileName, actPathName] = uigetfile('', 'Choose action file');
        actions = load(strcat(actPathName,actFileName));
    end
    
    %positions = load('~/DRL/ViZDoom/myvizdoom/experiments/radial_maze_2/trial_4/positions_trial1.txt');
    %actions = load('~/DRL/ViZDoom/myvizdoom/experiments/radial_maze_2/trial_4/actions_trial1.txt');
    action_pos = zeros(length(actions), 8);
    actions_pos(:,1:2) = actions(:,2) .* positions(1:4:length(positions),2:3);
    actions_pos(:,3:4) = actions(:,3) .* positions(1:4:length(positions),2:3);
    actions_pos(:,5:6) = actions(:,4) .* positions(1:4:length(positions),2:3);
    actions_pos(:,7:8) = actions(:,5) .* positions(1:4:length(positions),2:3);
    size(actions_pos);
    axis equal;
    axis([-600 600 -600 600]);
    hold on;
    plot(positions(:,2), positions(:,3));
    plot(actions_pos(:,1),actions_pos(:,2),'r<');
    plot(actions_pos(:,3),actions_pos(:,4),'g>');
    plot(actions_pos(:,5),actions_pos(:,6),'b^');
    plot(actions_pos(:,7),actions_pos(:,8),'kx');
    hold off;
end