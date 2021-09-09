#### 這是ya0的code

exp1_exp2_picture                       --> 碩論實驗照片

ddpg 和 sac 都是二軸的(庭瑜學姊的碩論)

vrep_camera                             --> 有關vrep相機的code

vrep --> 是六軸的
    scene --> 是模擬環境各個場景
        robot_with_joint_initial        --> 最原始的六軸模型
        scene_final                     --> 碩論訓練場景
        scenewithobject                 --> 這個不用理
        scene_for_master                --> 用來放碩論和ppt圖的場景

    // vrep連python一定要有 sim, simConst, simpleTest三個程式 //

    SAC_version_tra                     --> 訓練三軸到隨機點(訓練軌跡)
    SAC_camera_version2                 --> 訓練拾取點位置
    ===================================
    | 跑此程式請開scene_final,main_sac |
    ===================================
        imgTemp                         --> 存你想要存的實驗圖
        model                           --> 訓練好的權重
         * 04212333                     --> Banana
         * 04240939                     --> Banana
         * 06121204                     --> 畫碩論圖6-15,cubic,apple,orange,cup
         * 06221503                     --> 畫碩論圖6-16(pre-train)
         * 06230850                     --> 畫碩論圖6-16(no-pre-train)
         * 07082329                     --> 畫碩論圖6-17(no-yolo)
         * PLOT                         --> 出強化學習訓練結果實驗圖
            >>>--取YOLO訓練結果圖
                * extract_log
                * train_iou_visualization
                * train_loss_visualization            
        >>>--跟強化學習有關
            * buffer    
            * sac4
            * network2
            * env_new2          
        >>>--跟yolo有關
            * darknet
            * darknetUtils
            * yolo
        >>>--跟robot有關  
            * IK_FindOptSol             --> 逆向8組找最佳解
            * inverseKinematics         --> 逆向
            * Kinematics                --> 順向
            * Rot2RPY
        >>>--其他
            * config                    --> 所有需要調或設定的參數
            * main_sac                  --> 主程式
            * robot_vrep                --> vrep與python的程式
            * utils                     --> 畫最後訓練結果圖


    SAC_camera_version_real_world       --> 訓練好的拿來上機的code
    ==================================================
    | 跑此程式請開,main_sac,kinect,scene_final(可不開) |
    ==================================================
        >>>--跟強化學習有關
            * buffer    
            * sac4
            * network2
            * Eval_env                  --> 特定物件拾取和分類環境   
            * hand_env                  --> 線上訓練環境
        >>>--跟yolo有關
            * darknet
            * darknetUtils
            * yolo
        >>>--跟robot有關  
            * IK_FindOptSol             --> 逆向8組找最佳解
            * inverseKinematics         --> 逆向
            * Kinematics                --> 順向
            * Rot2RPY
        >>>--跟kinect有關
            * Kinect_new                --> kinect處理影像
            * mapper                    --> python連接kinect
        >>>--其他
            * main_sac                  --> 主程式
            * cnn                       --> 拿來看cnn輸入輸出(測試用)
            * config                    --> 所有需要調或設定的參數
            * robot_vrep                --> vrep與python的程式
            * utils                     --> 畫最後訓練結果圖
            * vrep_camera               --> 有關vrep相機的code


