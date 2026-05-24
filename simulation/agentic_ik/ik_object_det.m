robot = loadrobot("puma560", DataFormat="row");
endEffector = 'link7';
goal_url = "http://127.0.0.1:8000/goal";


q_current = homeConfiguration(robot);
alpha = 0.3; 

while true

    try
        res = webread(goal_url);
        if isfield(res,'goal') && ~isempty(res.goal)
            goal = res.goal;
        else
            pause(0.2);
            continue;
        end
    catch
        pause(0.2);
        continue;
    end

    goal = reshape(goal, 1, []);
    pos = goal(1:3);
    orient = goal(4:6);


    scale = 0.001;
    offset = [0.3, 0.4, 0.2];
    pos_robot = scale * pos + offset;  


    ik = robotics.InverseKinematics('RigidBodyTree', robot);
    weights = [0.5 0.5 0.5 1 1 1];
    tform = trvec2tform(pos_robot) * eul2tform(orient);
    [q_sol, solInfo] = ik(endEffector, tform, weights, q_current);

    if solInfo.ExitFlag > 0 && ~any(isnan(q_sol))

        q_current = alpha * q_sol + (1 - alpha) * q_current;

        clf;
        axis([-0.5 0.6 -0.5 0.6 0 0.6]);
        view(135, 25);
        grid on;
        hold on;
        axis manual;
        camlight('headlight');
        lighting gouraud;
        material metal;


        [Xs, Ys, Zs] = sphere(25);
        radius = 0.03;
        surf(radius*Xs + pos_robot(1), ...
            radius*Ys + pos_robot(2), ...
            radius*Zs + pos_robot(3), ...
            'FaceColor', 'red', 'EdgeColor', 'none');

        show(robot, q_current, 'PreservePlot', false, 'Frames', 'off');

        tform_curr = getTransform(robot, q_current, endEffector);
        pos_curr = tform2trvec(tform_curr);
        plot3(pos_curr(1), pos_curr(2), pos_curr(3), ...
            'g.', 'MarkerSize', 25);

        err = norm(pos_robot - pos_curr);
        title(sprintf("Chasing Target | Error: %.4f m", err));
    end

    pause(0.1); 
end