function ITRI_Limitation = ITRI_Constraint( GearRatio )

      PosLimit = [  170   -170  ;
                    130   -130  ;
                    170   -170  ;
                    190   -190  ;
                    125   -125  ;
                    360   -360  ] * pi / 180 * 0.9 ;

        
    VelLimit    =  (( 2000 / 60 ) / 0.7) ./ GearRatio * 2 * pi ;   % [rad/s]

    AccLimit    =  [  200  200  200  200  200  200  ] * pi / 180 ; % [rad/(s^2)]

    JerkLimit   =  AccLimit * 3 ; % [rad/(s^3)]

                                        
    PosCLimit   =  [ -40    80  ;    % X
                     -40    40  ;    % Y
                       2   120  ] ;  % Z
    
    VelCLimit   =  [  100  ;  100  ;  100  ] ;
    
    AccCLimit   =  [  1000   ;  1000   ;  1000  ] ;
    
    JerkCLimit  =  [  1000   ;  1000   ;  1000  ] ;
    
    % >>>> package
    Joint     = struct('Pos',  PosLimit,...
                       'Vel',  VelLimit,...
                       'Acc',  AccLimit,...
                       'Jerk', JerkLimit);
    
    Cartesian = struct('Pos',  PosCLimit,...
                       'Vel',  VelCLimit,...
                       'Acc',  AccCLimit,...
                       'Jerk', JerkLimit);
    
    ITRI_Limitation = struct('Joint', Joint, 'Cartesian', Cartesian);

end