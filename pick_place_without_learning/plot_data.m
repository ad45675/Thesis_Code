clc ; clear ; close all ;

% >>>> ITRI_parameter
[ ITRI_parameter , DH_table , SamplingTime ] = ITRI_Parameter;
DOF = size( DH_table , 1 ) ;
% >>>> ITRI_constraint
ITRI_Limitation = ITRI_Constraint( ITRI_parameter.GearRatio );

EstimatedData1 = load('C:\Users\user\Desktop\pick_place_without_learning\Trajectory\Joint_input1.txt' ) ;  % 實際資料
EstimatedData2 = load('C:\Users\user\Desktop\pick_place_without_learning\Trajectory\Joint_input2.txt' ) ;  % 實際資料
EstimatedData3 = load('C:\Users\user\Desktop\pick_place_without_learning\Trajectory\Joint_input3.txt' ) ;  % 實際資料
EstimatedData4 = load('C:\Users\user\Desktop\pick_place_without_learning\Trajectory\Joint_input4.txt' ) ;  % 實際資料
Trajectory1 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/joint_out1.txt' ) ;  % 軌跡命令
Trajectory2 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/joint_out2.txt' ) ;  % 軌跡命令
Trajectory3 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/joint_out3.txt' ) ;  % 軌跡命令
Trajectory4 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/joint_out4.txt' ) ;  % 軌跡命令
% 
encorder1 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/vrep_joint1.txt' ) ;  % 軌跡命令
encorder2 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/vrep_joint2.txt' ) ;  % 軌跡命令
encorder3 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/vrep_joint3.txt' ) ;  % 軌跡命令
encorder4 = load( 'C:/Users/user/Desktop/pick_place_without_learning/Trajectory/vrep_joint4.txt' ) ;  % 軌跡命令

test_tra = load('C:/Users/user/Desktop/rl/vrep/SAC_camera_version2/Trajectory/joint.txt');

%% vrep 傳回來的data
% pos= load( 'Data/pos.txt' ) ;  % 回授位置
% vel= load('Data/vel.txt' ); % 回授速度
% vel=vel/0.5;
% torque = load('Data/torque.txt' ); % 回授速度

%% 送去vrep的命令
P_out=[Trajectory1(: , (1:6)) ; Trajectory2(: , (1:6)) ;Trajectory3(: , (1:6));  Trajectory4(: , (1:6)) ];
P_in=[EstimatedData1(: , (1:6));EstimatedData2(: , (1:6));EstimatedData3(: , (1:6));EstimatedData4(: , (1:6))];
vrep_joint=[encorder1(: , (1:6));encorder2(: , (1:6));encorder3(: , (1:6));encorder4(: , (1:6))]

% Vs=Trajectory(: , (7:12));
% Ta = EstimatedData( : , ( 19 : 24 ) ) ;  % 實際轉矩 (Nm)

n = 6 ;  % 關節數
tf = 10 ;  % 結束時間 (sec)
wm = 1000 ;  % 量測取樣頻率 (Hz)
tm = 1 / wm ;  % 量測取樣時間 (sec)
Tm = tm : tm : tf ;  % 量測時間 (sec)
Nm = tf / tm ;  % 量測資料數 (samples)
tm=length(EstimatedData1(: , (1)))+length(EstimatedData2(: , (1)))+length(EstimatedData3(: , (1)))+length(EstimatedData4(: , (1)));

% plot(Trajectory(,(1:6)))
% grid on;
% plot(pos)

% figure( 1 )
% for i = 1 : 6
% 
%     subplot( 3 , 2 , i ) ;
%     plot(linspace(1,tm-1,tm) /wm,  P_out( : , i ) , '-' , linspace(1,tm-1,tm) /wm,P_in( : , i ) , '--' , linspace(1,tm-1,tm) /wm,vrep_joint( : , i ) ,'c-. ' , 'LineWidth' , 2 ) ;
%     title( [ 'Direct comparison , joint ' , num2str( i ) ] , 'FontWeight' , 'bold' , 'FontSize' , 12 ) ;
%     xlabel( 'Time (sec)' ) ; ylabel( 'position (rad)' ) ;
%     legend( ' P_out' , 'P_in' ,'vep' ) ;
%     grid on ;
%  
% end



figure( 1 )
for i = 1 : 6

    subplot( 3 , 2 , i ) ;
    plot(linspace(1,tm-1,tm) /wm,  P_out( : , i ) , '-' , linspace(1,tm-1,tm) /wm,P_in( : , i ) , '--'  , 'LineWidth' , 2 ) ;
    title( [ 'pos , joint ' , num2str( i ) ] , 'FontWeight' , 'bold' , 'FontSize' , 12 ) ;
    xlabel( 'Time (sec)' ) ; ylabel( 'position (rad)' ) ;
%     legend( ' P_out' , 'vrep_joint'  ) ;
    grid on ;
    hold on;
    if min (P_out(:,i)) < ITRI_Limitation.Joint.Pos(i,2)
         mini = min (P_out(: , i)) * 1.1;
    else
         mini = ITRI_Limitation.Joint.Pos(i,2) * 1.1;
    end
    if max (P_out(:,i)) > ITRI_Limitation.Joint.Pos(i,1)
        maxi = max (P_out(: ,i) )* 1.1;
    else
         maxi = ITRI_Limitation.Joint.Pos(i,1) * 1.1;
    end
        
    plot( ones(1, 100) .* (length(encorder1(: , (1)))-1)*0.001,linspace(mini, maxi, 100), 'b--');
    plot( ones(1, 100) .* (length(encorder1(: , (1)))+length(encorder2(: , (1)))-1)*0.001 ,linspace(mini, maxi, 100), 'b--');
    plot( ones(1, 100) .* (length(encorder1(: , (1)))+length(encorder2(: , (1)))+length(encorder3(: , (1)))-1)*0.001 ,linspace(mini, maxi, 100), 'b--');
    plot( ones(1, 100) .* (length(encorder1(: , (1)))+length(encorder2(: , (1)))+length(encorder3(: , (1)))+length(encorder4(: , (1)))-1)*0.001 ,linspace(mini, maxi, 100), 'b--');
    hold on;
   
 
end


% figure( 2 )
% 
% for i = 1 : 6
% 
%     subplot( 3 , 2 , i ) ;
%     plot( Tm ,Vs( : , i ) , '-' , Tm ,vel( : , i ) , '--'  , 'LineWidth' , 2 ) ;
%     title( [ 'Direct comparison , joint ' , num2str( i ) ] , 'FontWeight' , 'bold' , 'FontSize' , 12 ) ;
%     xlabel( 'Time (sec)' ) ; ylabel( 'Velocity (rad/s)' ) ;
%     legend(  'Vs' , 'vel' ) ;
%     grid on ;
%  
% end
% 
% figure( 3 )
% 
% for i = 1 : 6
% 
%     subplot( 3 , 2 , i ) ;
%     plot( Tm , Ta( : , i ) , '-' , Tm , torque( : , i ) , '--' , 'LineWidth' , 2 ) ;
%     title( [ 'Direct comparison , joint ' , num2str( i ) ] , 'FontWeight' , 'bold' , 'FontSize' , 12 ) ;
%     xlabel( 'Time (sec)' ) ; ylabel( 'Torque (Nm)' ) ;
%     legend( 'Ta' , 'torque' ) ;
%     grid on ;
%  
% end