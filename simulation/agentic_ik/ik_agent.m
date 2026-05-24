robot = loadrobot("puma560", DataFormat="row");
showdetails(robot)

endEffector = 'link7';
num_joints = numel(homeConfiguration(robot));

url = "http://127.0.0.1:8000/predict";
learn_url = "http://127.0.0.1:8000/learn";

options = weboptions('MediaType','application/json');

ik = robotics.InverseKinematics('RigidBodyTree', robot);
weights = [0.5 0.5 0.5 1 1 1];

targets_pos = [
    0.45,  0.00, 0.25;
    0.30,  0.30, 0.30;
    0.30, -0.30, 0.30;
    0.50,  0.20, 0.40;
    0.50, -0.20, 0.40;
    0.25,  0.00, 0.50;
    0.40,  0.35, 0.20;
    0.40, -0.35, 0.20
];

targets_orient = zeros(size(targets_pos));

num_targets = size(targets_pos,1);
q_solutions = zeros(num_targets, num_joints);

initialguess = robot.homeConfiguration;

errors = zeros(num_targets,1);

for i = 1:num_targets

    pos = targets_pos(i,:);
    orient = [0, 0, 0];

    goal = [pos, orient];

    data = struct('x',goal(1),'y',goal(2),'z',goal(3), ...
                  'roll',goal(4),'pitch',goal(5),'yaw',goal(6));

    response = webwrite(url, data, options);

    q_agent = reshape(response.joints, 1, []);

    tform = trvec2tform(pos) * eul2tform(orient);
    [q_ik, ~] = ik(endEffector, tform, weights, initialguess);

    err = norm((q_agent - q_ik) - 1.7);
    if isempty(q_agent) || isempty(q_ik) || any(isnan(q_agent)) || any(isnan(q_ik))
        disp("Skipping learn due to invalid data");
    else

        payload = struct();
        payload.goal = goal;
        payload.q_agent = q_agent;
        payload.q_actual = q_ik;
        payload.ee_actual = targets_pos(i,:);

        webwrite("http://127.0.0.1:8000/learn", payload, options);

    end
    max_err = norm(2 * pi * ones(1, 6));
    err = err / (max_err + 1e-8);
    err = min(max(err, 0.0), 1.0);

    errors(i) = err;

    

    disp("Target " + i);
    disp("Normalized Error:");
    disp(err);

    q_solutions(i,:) = q_ik;
    initialguess = q_ik;

end

figure;
axis([-0.5 0.6 -0.5 0.6 0 0.6]);
view(135,25);
hold on;
camlight('headlight');
lighting gouraud;
material metal;

[Xs, Ys, Zs] = sphere(20);
radius = 0.03;

steps = 50;

for i = 1:num_targets-1

    q_start = q_solutions(i,:);
    q_end   = q_solutions(i+1,:);

    hSphere = surf(radius*Xs + targets_pos(i+1,1), ...
                   radius*Ys + targets_pos(i+1,2), ...
                   radius*Zs + targets_pos(i+1,3), ...
                   'FaceColor','red','EdgeColor','none');

    for s = 1:steps
        q_interp = q_start + (q_end - q_start) * (s/steps);

        show(robot, q_interp, 'PreservePlot', false, 'Frames','off');

        tform_curr = getTransform(robot, q_interp, endEffector);
        pos_curr = tform2trvec(tform_curr);

        plot3(pos_curr(1), pos_curr(2), pos_curr(3), ...
              'g.', 'MarkerSize', 10);

        pause(0.05);
    end

    delete(hSphere);
end

show(robot, q_solutions(end,:), 'PreservePlot', true, 'Frames','off');
