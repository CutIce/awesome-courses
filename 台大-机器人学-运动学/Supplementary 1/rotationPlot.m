%plot Rotation matrix just for checking
% R = RR(:,:,181);
% R = ans;
% R = eye(3);
% R = TESTR;
% RR = [0.1222 0.8536 0.5064; 0.2003 -0.5209 0.8298; 0.9721 0 -0.2346;];
function rotationPlot(RR,FoE)

N = size(RR,3);
Resol = 50;
videoSave = 0;
if videoSave == 1
    AVI1 = VideoWriter('DH1','MPEG-4');
    open(AVI1);
end

%Initial the plots
fig = figure(1);
plot3(0,0,0,'k.','MarkerSize',15);
set(fig,'Color',[1 1 1]);hold on;
AZ = -70;      %iso view angle Default: -75,30
EL = 35;        %iso view angle
grid on;grid minor;axis equal;axis([-1 1 -1 1 -1 1]);xlabel('x');ylabel('y');zlabel('z');
set(gca,'XDir','Reverse');set(gca,'YDir','Reverse');
set(gca,'FontSize',25);
view(AZ,EL);

%get the color order
CO = get(gca,'ColorOrder');
%label Texts
LB = ['x"'; 'y"'; 'z"';];
LW = ['x' ; 'y' ; 'z' ;];
%changing labels:

%Texts Scales
WTSc = 1.2;
BTSc = 1.3;

for j = 1:N
    %ploting world frame and texts
    if j < N
        Ind = floor(j/Resol);
        if FoE == 0
            Count = 1;
        else
            Count = Ind*Resol+1;
        end
    end
    
    WR = RR(:,:,Count);
    WF = quiver3([0 0 0],[0 0 0],[0 0 0],WR(1,:),WR(2,:),WR(3,:),'LineWidth',3);
    WT = text(WR(1,:)*WTSc,WR(2,:)*WTSc,WR(3,:)*WTSc,LW,'FontSize',15,'Color',CO(1,:));
    %ploting body frame and texts
    BR = RR(:,:,j);
    BF = quiver3([0 0 0],[0 0 0],[0 0 0],BR(1,:),BR(2,:),BR(3,:),'LineWidth',3);
    BT = text(BR(1,:)*BTSc,BR(2,:)*BTSc,BR(3,:)*BTSc,LB,'FontSize',15,'Color',CO(2,:));
    plotSetting(BF, WF);
    %to pause
    %     if mod(j+1,Resol) == 0
    %         pause();
    %     end
    pause(0.05);
%     if mod(j+1,Resol)==0 && j+1<N
%         pause(2);
%     end
    %Video Save:
    if videoSave == 1
        if mod(j+1,Resol)==0 && j+1<N
            no = Resol;
        else
            no = 1;
        end
        for k = 1:no
            vidF = getframe(fig);
            writeVideo(AVI1, vidF);
        end
    end
    
    if (j<N)
        delete(BF);
        delete(BT);
        delete(WF);
        delete(WT);
    end
end
hold off;
if videoSave == 1
    close(AVI1);
end

end

%plot setting for Bframe and Wframe
function plotSetting(BF, WF)
CO = get(gca,'ColorOrder');
%daspect([1 1 1]);
set(BF,'MaxHeadSize',0.4);
set(WF,'MaxHeadSize',0.4);

BF.AutoScaleFactor = 1;
WF.AutoScaleFactor = 1;

BF.Color = CO(2,:);
WF.Color = CO(1,:);
WF.LineStyle = '-.';
end

%%


