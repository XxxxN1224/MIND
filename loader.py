import numpy as np
from importlib import import_module
from common.data import padding_traj_nn
from common.geometry import project_point_on_polyline
from agent import AgentColor, MINDAgent, NonReactiveAgent
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import TrackCategory


class ArgoAgentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_agents(self, smp, cl_agt_cfg=None):
        """
        根据 语义地图信息 和 闭环智能体配置 加载智能体实体。

        参数:
            smp: 语义地图 SemanticMap。
            cl_agt_cfg: 闭环智能体的配置信息，指示如何初始化和配置闭环智能体。

        返回:
            初始化后的智能体实体列表，包括每个智能体的配置信息和行为预测模型。
        """
        cl_agts = self.get_closed_loop_agents(cl_agt_cfg)   # 根据cl_agt_cfg和smp里的AV信息配置cl_agts
        trajs_info = self.get_trajs_info(smp)   # 获取trajs信息
        agents = []

        for traj_pos, traj_ang, traj_vel, traj_type, traj_tid, traj_cat, has_flag in zip(*trajs_info):

            traj_info = [traj_pos, traj_ang, traj_vel, has_flag]

            if traj_tid in cl_agts: # 找到AV，并将cl_agts配置赋值到智能体实体中

                agent_file, agent_name = cl_agts[traj_tid]["agent"].split(':')  # 分割以获取智能体的文件路径和类名
                planner_cfg = cl_agts[traj_tid]["planner_config"]   # 获取智能体的规划器配置
                # get planner type 动态导入智能体类并实例化
                agent = getattr(import_module(agent_file), agent_name)()    # Python 内置函数，用于获取对象的属性值

                if isinstance(agent, MINDAgent):    # isinstance 是 Python 内置的一个函数，用于检查一个对象是否是指定类的实例
                    agt_clr = AgentColor().ego_disable()    # 设置MINDAgent类型的智能体颜色，禁用自车渲染

                agent.init(traj_tid, traj_type, traj_cat, traj_info, smp, agt_clr,
                        semantic_lane_id=cl_agts[traj_tid]["semantic_lane"],
                        target_velocity=cl_agts[traj_tid]["target_velocity"])

                agent.set_enable_timestep(cl_agts[traj_tid]["enable_timestep"]) # 设置智能体的启用时间步
                agent.init_planner(planner_cfg) # 使用配置初始化智能体的规划器

                if isinstance(agent, MINDAgent):    # 对于MINDAgent类型的智能体，更新其目标车道信息
                    agent.update_target_lane(smp, cl_agts[traj_tid]["semantic_lane"])

            else:
                agent = NonReactiveAgent()  # 对于未配置为闭环智能体的实体，初始化为非反应型智能体
                agt_clr = AgentColor().exo()    # 设置非反应型智能体的颜色
                agent.init(traj_tid, traj_type, traj_cat, traj_info, smp, agt_clr)
                
            agents.append(agent)    # 将初始化后的智能体添加到智能体列表中
        return agents


    def get_closed_loop_agents(self, cl_agt_cfg):
        """
        根据给定的闭环agent配置，构建并返回一个包含这些agent及其配置的字典。
        
        参数:
        cl_agt_cfg (list): 一个包含agent配置的列表，每个配置都是一个字典。
        
        返回:
        dict: 一个字典，键是agent ID，值是另一个字典，包含该agent的配置信息。
        """
        closed_loop_agents = dict() # 初始化一个空的字典来存储闭环代理及其配置

        if cl_agt_cfg is None:# 如果闭环代理配置列表为空，则直接返回空字典
            return closed_loop_agents
        
        for c in cl_agt_cfg:    # 遍历配置列表中的每个代理配置，但实际上cl_agt_cfg的len=1
            agt_id = c["id"]    # 获取当前代理的ID

            if agt_id in closed_loop_agents.keys(): # 如果当前代理ID已经在闭环代理字典中存在，则跳过此配置
                continue

            closed_loop_agents[agt_id] = dict() # 为新代理初始化一个字典来存储其配置信息
            closed_loop_agents[agt_id]["enable_timestep"] = c["enable_timestep"]    # 设置代理的启用时间步

            if c["target_velocity"] == -1:  # 根据配置设置代理的目标速度，-1表示无目标速度
                closed_loop_agents[agt_id]["target_velocity"] = None
            else:
                closed_loop_agents[agt_id]["target_velocity"] = c["target_velocity"]

            if c["semantic_lane"] == -1:    # 根据配置设置代理的目标车道，-1表示无特定车道
                closed_loop_agents[agt_id]["semantic_lane"] = None
            else:
                closed_loop_agents[agt_id]["semantic_lane"] = c["semantic_lane"]

            closed_loop_agents[agt_id]["agent"] = c["agent"]    # 设置代理的代理类型
            closed_loop_agents[agt_id]["planner_config"] = c["planner_config"]  # 设置代理的规划器配置
        return closed_loop_agents

    def get_trajs_info(self, smp):
        """
        从Argoverse数据集中提取轨迹信息。

        该函数从Parquet文件加载场景数据，处理每个轨迹的数据，并按照预定义的类别和顺序组织这些数据。
        它还执行数据过滤和填充，以确保后续处理所需的数据格式和完整性。

        参数:
        - smp: 样本对象，包含关于场景和语义地图的信息。

        返回:
        - trajs_pos: 轨迹位置数组。
        - trajs_ang: 轨迹角度数组。
        - trajs_vel: 轨迹速度数组。
        - trajs_type: 每个轨迹的对象类型列表。
        - trajs_tid: 每个轨迹的轨迹ID列表。
        - trajs_cat: 每个轨迹的轨迹类别列表。
        - has_flags: 数组，指示每个轨迹的每个时间步是否有数据。
        """
        scenario = scenario_serialization.load_argoverse_scenario_parquet(self.data_path)   # av2的api，从Parquet文件加载场景数据

        obs_len = 50    # 定义观察长度
        scored_idcs, unscored_idcs, fragment_idcs = list(), list(), list()  # exclude AV

        for idx, x in enumerate(scenario.tracks):   # 筛选，仅处理以下类型agent的轨迹
            # scenario.focal_track_id告诉你哪一个具体的轨迹是当前场景的主要关注点
            # TrackCategory.FOCAL_TRACK则是用于表示这条轨迹具有特殊的重要性
            if x.track_id == scenario.focal_track_id and x.category == TrackCategory.FOCAL_TRACK:
                focal_idx = idx
            elif x.track_id == 'AV':
                av_idx = idx
            elif x.category == TrackCategory.SCORED_TRACK:  # 被标记为可以评估或打分的实体，场景中的关键参与者
                scored_idcs.append(idx)
            elif x.category == TrackCategory.UNSCORED_TRACK:    # 这类轨迹是不参与评分的对象
                unscored_idcs.append(idx)
            elif x.category == TrackCategory.TRACK_FRAGMENT:    # 一个不完整的轨迹片段，物体只在场景的一部分时间内可见
                fragment_idcs.append(idx)

        assert av_idx is not None, '[ERROR] Wrong av_idx'
        assert focal_idx is not None, '[ERROR] Wrong focal_idx'
        assert av_idx not in unscored_idcs, '[ERROR] Duplicated av_idx'

        # 按照预定义顺序排序索引
        sorted_idcs = [focal_idx, av_idx] + scored_idcs + unscored_idcs + fragment_idcs
        sorted_cat = ["focal", "av"] + ["score"] * \
                     len(scored_idcs) + ["unscore"] * len(unscored_idcs) + ["frag"] * len(fragment_idcs)
        sorted_tid = [scenario.tracks[idx].track_id for idx in sorted_idcs]

        # * must follows the pre-defined order
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags = list(), list(), list(), list(), list()
        trajs_tid, trajs_cat = list(), list()  # track id and category

        # 处理每个轨迹
        for k, ind in enumerate(sorted_idcs):
            track = scenario.tracks[ind]
            
            # 提取轨迹的时间步、位置、角度和速度
            traj_ts = np.array([x.timestep for x in track.object_states], dtype=np.int16)  # [N_{frames}]
            traj_pos = np.array([list(x.position) for x in track.object_states])  # [N_{frames}, 2]
            traj_ang = np.array([x.heading for x in track.object_states])  # [N_{frames}]
            traj_vel = np.array([list(x.velocity) for x in track.object_states])  # [N_{frames}, 2]
            # cal scalar velocity：计算标量速度
            traj_vel = np.linalg.norm(traj_vel, axis=1)  # [N_{frames}]

            # 定义时间步范围和观察时间步
            ts = np.arange(0, 110)  # [0, 1,..., 109]
            ts_obs = ts[obs_len - 1]  # always 49

            # * only contains future part：过滤掉未来部分的轨迹
            if traj_ts[0] > ts_obs:
                continue
            # * not observed at ts_obs：过滤掉在ts_obs时间步没有观测到的轨迹
            if ts_obs not in traj_ts:
                continue

            # * far away from map (only for observed part)：过滤掉远离地图的轨迹
            traj_obs_pts = traj_pos[:obs_len]  # [N_{frames}, 2]
            on_lanes = []
            on_lane_thres = 5.0
            for traj_pt in traj_obs_pts:
                on_lane = False
                for semantic_lane in smp.semantic_lanes.values():
                    proj_pt, _, _ = project_point_on_polyline(traj_pt, semantic_lane)
                    if np.linalg.norm(proj_pt - traj_pt) < on_lane_thres:
                        on_lane = True
                        break
                on_lanes.append(on_lane)

            # if any of the observed points is not on the lane, then skip：如果任何一个观测点不在车道上，则跳过
            if not np.all(on_lanes):
                continue

            # 创建标记数组
            has_flag = np.zeros_like(ts)
            # print(has_flag.shape, traj_ts.shape, traj_ts)
            has_flag[traj_ts] = 1

            # object type：对象类型
            traj_type = [track.object_type for _ in range(len(ts))]

            # pad pos, nearest neighbor：填充位置数据，使用最近邻插值
            traj_pos_pad = np.full((len(ts), 2), None)
            traj_pos_pad[traj_ts] = traj_pos
            traj_pos_pad = padding_traj_nn(traj_pos_pad)
            # pad ang, nearest neighbor：填充角度数据，使用最近邻插值
            traj_ang_pad = np.full(len(ts), None)
            traj_ang_pad[traj_ts] = traj_ang
            traj_ang_pad = padding_traj_nn(traj_ang_pad)
            traj_vel_pad = np.full((len(ts),), 0.0)
            traj_vel_pad[traj_ts] = traj_vel

            # 将处理后的轨迹数据添加到结果列表中
            trajs_pos.append(traj_pos_pad)
            trajs_ang.append(traj_ang_pad)
            trajs_vel.append(traj_vel_pad)
            has_flags.append(has_flag)
            trajs_type.append(traj_type)
            trajs_tid.append(sorted_tid[k])
            trajs_cat.append(sorted_cat[k])

        # 重新采样轨迹信息
        res_traj_infos = self.resample_trajs_info(
            [trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags])

        trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags = res_traj_infos

        # 转换为NumPy数组并设置数据类型
        trajs_pos = np.array(trajs_pos).astype(np.float32)  # [N, 110(50), 2]
        trajs_ang = np.array(trajs_ang).astype(np.float32)  # [N, 110(50)]
        trajs_vel = np.array(trajs_vel).astype(np.float32)  # [N, 110(50), 2]
        has_flags = np.array(has_flags).astype(np.int16)  # [N, 110(50)]

        return trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags

    def resample_trajs_info(self, trajs_info):
        # traj_info = traj_pos, traj_ang, traj_vel, traj_type, traj_tid, traj_cat, has_flag
        ori_sim_step = 0.1
        sim_step = 0.02
        res_trajs_pos, res_trajs_ang, res_trajs_vel, res_trajs_type, res_trajs_tid, res_trajs_cat, res_has_flags = [], [], [], [], [], [], []
        interp_len = int(ori_sim_step / sim_step)

        trajs_pos, trajs_ang, trajs_vel, trajs_type, trajs_tid, trajs_cat, has_flags = trajs_info
        for a_idx in range(len(trajs_pos)):
            res_traj_pos, res_traj_ang, res_traj_vel, res_traj_type, res_traj_tid, res_traj_cat, res_has_flag = [], [], [], [], [], [], []
            for t_idx in range(len(trajs_pos[a_idx])):
                if t_idx == len(trajs_pos[a_idx]) - 1:
                    res_traj_pos.append(trajs_pos[a_idx][t_idx])
                    res_traj_ang.append(trajs_ang[a_idx][t_idx])
                    res_traj_vel.append(trajs_vel[a_idx][t_idx])
                    res_has_flag.append(has_flags[a_idx][t_idx])
                    res_traj_type.append(trajs_type[a_idx][t_idx])
                else:
                    for iidx in range(interp_len):
                        r = iidx / interp_len
                        res_traj_pos.append(trajs_pos[a_idx][t_idx] * (1 - r) + trajs_pos[a_idx][t_idx + 1] * r)
                        angle_diff = trajs_ang[a_idx][t_idx + 1] - trajs_ang[a_idx][t_idx]
                        # normalize to [-pi, pi]
                        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                        interp_ang = trajs_ang[a_idx][t_idx] + angle_diff * r
                        # normalize to [-pi, pi]
                        interp_ang = np.arctan2(np.sin(interp_ang), np.cos(interp_ang))
                        res_traj_ang.append(interp_ang)
                        res_traj_vel.append(trajs_vel[a_idx][t_idx] * (1 - r) + trajs_vel[a_idx][t_idx + 1] * r)
                        res_has_flag.append(has_flags[a_idx][t_idx] * (1 - r) + has_flags[a_idx][t_idx + 1] * r > 0.5)
                        res_traj_type.append(trajs_type[a_idx][t_idx])

            res_trajs_pos.append(np.array(res_traj_pos))
            res_trajs_ang.append(np.array(res_traj_ang))
            res_trajs_vel.append(np.array(res_traj_vel))
            res_trajs_type.append(res_traj_type)
            res_has_flags.append(np.array(res_has_flag))
            res_trajs_tid.append(trajs_tid[a_idx])
            res_trajs_cat.append(trajs_cat[a_idx])

        res_traj_info = [res_trajs_pos, res_trajs_ang, res_trajs_vel, res_trajs_type, res_trajs_tid, res_trajs_cat,
                         res_has_flags]
        return res_traj_info
