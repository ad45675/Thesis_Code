function [ITRI_parameter, DH_table, SamplingTime] = ITRI_Parameter

    %-----自由度-----
    DOF  = 6; 

    %-----DH_table-----
               %(theta) (D)             (a)      (alpha) 
    DH_table = [ 0      34.5            7.5      pi/2    ;
                 pi/2   0               27.0     0       ;
                 0      0               9.0      pi/2    ; 
                 0      29.5            0       -pi/2    ;
                 0      0               0        pi/2    ;
                 0      10.2 + 18.0     0        0       ; ] ;

    %-----Sample Time-----

    SamplingTime = 0.001 ;

    %-----馬達額定轉矩-----

    RatedTorque = [  1.3  1.3  1.3  0.32  0.32  0.32  ] ;

    %-----馬達齒輪比-----

    GearRatio = [  120.0  120.0  120.0  102.0  80.0  51.0  ] ;
    
    ITRI_parameter = struct('DOF',          DOF,...
                            'DH_table',     DH_table,...
                            'SamplingTime', SamplingTime,...
                            'RatedTorque',  RatedTorque,...
                            'GearRatio',    GearRatio);

end