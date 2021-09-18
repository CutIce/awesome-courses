%This is to generate the baby steps for all rotation matrix from a rotation
%order
%Input: RO (Rotation Order, e.g. [3 30; 2 -60; 1 50;] 
% -> about z 30 deg, y -60 deg, x 50 deg
%Output
% RO = [  3 30;
%         1 -30;
%         2 -60;];                %Rotation Order
% FoE = 0;                        %Fixed or Euler angle? 0 = Fixed 1 = Euler
function RA = rotationMGen(RO,FoE)

%Check:
if ((FoE ~= 0) && (FoE ~= 1))
    disp('Wrong Index');
end
Resol = 50;                     %n frames between each rotation (Default: 50)
SR = eye(3);                    %SR: Starting Rotation matrix (Default: eye(3))
% SR = [1 0 0; 0 -1 0; 0 0 -1];   %Alternative starting frame rotation matrix
% End of inputs

RO(:,2) = RO(:,2)*pi/180;       %change to radian
SpN = size(RO,1);               %Steps of rotation
RMN = SpN*Resol;                %R.M. total No
RA = zeros(3,3,RMN);            %creating Rotation matrix Array;

for i = 1:SpN
    Rang = linspace(0,RO(i,2),Resol);   %stores each rotation angle of each RM
    ROmod = RO(1:i,:);
    for j = 1:Resol
        ROmod(i,2) = Rang(j);
        if FoE == 0     %if fixed angle
            RA(:,:,Resol*(i-1)+j) = rotationMod(ROmod,FoE) * SR;
        else            %if euler angle
            RA(:,:,Resol*(i-1)+j) = SR * rotationMod(ROmod,FoE);
        end
    end
end
end