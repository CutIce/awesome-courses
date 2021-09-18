%% Instructions and Settings
% This program is to demostrate the frame rotation about x/y/z axis via
% animation.
% Only two inputs shall be required.
% 1. RotatOrder: n*2 Matrix, where n = no of steps of rotations to do with
%                the frame, each row consists of two inputs (axis of
%                rotation and degrees of rotation)
%
%    -  1st column: Axis of rotation, 1/2/3 is corresponding to x/y/z axis
%                   of rotation (values other than 1/2/3 are invalid inputs)
%    -  2nd column: Degrees of rotation about the axis. (Unit: degree)
%
%    Example: [ 1 60;
%               2 30;]; => rotation 60 deg about x-axis, then 30 deg about
%                          y-axis
%
% 2. FixedOrEuler: Determine whether the rotation is performed as fixed
%                  angles or euler angles. (Values other than 0/1 are
%                  invalid)
%                  Fixed Angles: 0       
%                  Euler Angles: 1

%% Input here
% Try this: Using fixed angles, to rotate the frame 60 degs about x-axis and
% then 30 degs about y-axis.
%
% You are free to modify the inputs or even add more steps to RotatOrder to
% see how the rotation goes
RotatOrder = [  1 60;
                2 30;];
FixedOrEuler = 0;

RotatMatArr = rotationMGen(RotatOrder,FixedOrEuler);

%% Display the rotation matrix after the transformation
disp('Rotation Matrix:');
disp(RotatMatArr(:,:,end));

%% Start the animation
rotationPlot(RotatMatArr,FixedOrEuler);


