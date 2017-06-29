positions = load('/home/james/DRL/ViZDoom/myvizdoom/experiments/linear_track_cues_2/test_1/positions_trial1.txt');
actions = load('/home/james/DRL/ViZDoom/myvizdoom/experiments/linear_track_cues_2/test_1/actions_trial1.txt');
action_indices = load('/home/james/DRL/ViZDoom/myvizdoom/experiments/linear_track_cues_2/test_1/action_indices.txt');
action_symbols = ['d' 's' 'o'];
lgd = legend('TURN_LEFT', 'TURN_RIGHT', 'MOVE_FORWARD');
plot(positions(:,2), positions(:,3));
daspect([1 1 1])
hold on;
j = 1;
for i = 1:size(actions,1)
    while (actions(i,1) > positions(j,1))
        j = j + 1;
    end
    for k = 2:size(actions,2)
        if (actions(i,k) == 1)
            plot(positions(j,2),positions(j,3),action_symbols(k-1));
        end
    end
end
