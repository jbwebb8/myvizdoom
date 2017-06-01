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
        positions = load(strcat(posPathName,posFileName))
        [actFileName, actPathName] = uigetfile('', 'Choose action file');
        actions = load(strcat(actPathName,actFileName))
    end
    
    plot(positions);
    plot(actions);
end