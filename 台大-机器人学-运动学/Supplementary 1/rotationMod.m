%order: [1 pi/3; 2 pi/6;...] n*2 matrix; each row indicates a rotation
%step; first col = axis; 2nd col = angle;
%fixedEuler: 0: Fixed angle, 1: Euler angle
%20161206 Modified: can input syms
function outM = rotationMod(order,fixedEuler)      
axisCode = 'xyz';
outM = eye(3);
for i = 1:size(order,1)
    axis = axisCode(order(i,1));
    angle = order(i,2);
    switch axis
        case 'x'
            rotatM = [1 0 0; 0 cos(angle) -sin(angle); 0 sin(angle) cos(angle);];
        case 'y'
            rotatM = [cos(angle) 0 sin(angle);0 1 0;-sin(angle) 0 cos(angle);];
        case 'z'
            rotatM = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1;];
    end
    if fixedEuler == 0
        outM = rotatM*outM; %Fixed
    else
        outM = outM*rotatM; %Euler
    end
end


