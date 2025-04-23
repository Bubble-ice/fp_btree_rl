#include "../include/api2py.h"
#include "../include/btree.h"
#include "../include/sa.h"
#include <cstring>
#include <iostream>
#include <stack>

const float rotate_rate = 0.3;
const float swap_rate = 0.5;

SAResult run_with_sa(B_Tree_Ext fp, int times, int local, float init_temp,
                     float term_temp, float alpha, const char *outfile, bool is_debug)
{
    SAResult result;
    try
    {
        srand(time(0));

        double time = seconds();
        // fp.show_modules();
        double last_time = SA_Floorplan(fp, times, local, term_temp, is_debug);
        // Random_Floorplan(fp,times);
        fp.list_information();

        // 填充结果结构体
        result.cpu_time = seconds() - time;
        result.last_cpu_time = last_time - time;
        result.cost = float(fp.getCost());
        result.area = float(fp.getArea());
        result.wire_length = float(fp.getWireLength());
        result.dead_space = float(fp.getDeadSpace());

        // if (false){ // log performance and quality
        //     if (strlen(outfile) == 0)
        //         strcpy(outfile, strcat(fp.get_filename(), ".res"));

        //     last_time = last_time - time;
        //     printf("CPU time       = %.2f\n", seconds() - time);
        //     printf("Last CPU time  = %.2f\n", last_time);
        //     FILE *fs = fopen(outfile, "a+");

        //     fprintf(fs,
        //             "CPU= %.2f, Cost= %.6f, Area= %.0f, Wire= %.0f, Dead=%.4f ",
        //             last_time, float(fp.getCost()), float(fp.getArea()),
        //             float(fp.getWireLength()), float(fp.getDeadSpace()));
        //     fprintf(fs, " :%d %d %.0f \n", times, local, avg_ratio);
        //     fclose(fs);
        // }
    }
    catch (...)
    {
    }
    return result;
}

B_Tree_Ext::B_Tree_Ext(string fn, float calpha) : B_Tree(calpha)
{
    read(fn);
    init();
    pin_nodes_init();
}

void B_Tree_Ext::pin_nodes_init()
{
    pin_nodes.clear();
    pin_list.clear();

    int p_id_cnt = 0; // 引脚id计数

    // 根模块
    for (Pin &p : root_module.pins)
    {
        PinNode pn;
        pn.id = p_id_cnt++, pn.mod_id = -1, pn.net_id = p.net;
        pn.pin_x = p.ax, pn.pin_y = p.ay;
        pn.mod_w = pn.mod_h = pn.mod_x = pn.mod_y = 0;

        pin_table[&p] = pn.id;
        pin_list.push_back(&p);
        pin_nodes.push_back(pn);
    }

    // 普通模块
    for (int i = 0; i < modules_N; i++)
    {
        Module &m = modules[i];
        Module_Info &mf = modules_info[i];
        for (Pin &p : m.pins)
        {
            PinNode pn;
            pn.id = p_id_cnt++, pn.mod_id = -1, pn.net_id = p.net;
            pn.pin_x = p.ax, pn.pin_y = p.ay;
            pn.mod_w = m.width, pn.mod_h = m.height;
            pn.mod_x = mf.x, pn.mod_y = mf.y;

            pin_table[&p] = pn.id;
            pin_list.push_back(&p);
            pin_nodes.push_back(pn);
        }
    }

    num_of_pins = p_id_cnt;
}

py::array_t<float> B_Tree_Ext::get_py_pin_nodes_info()
{
    // TODO 确保已经计算过布局中模块信息和引脚信息

    // 更新引脚节点信息
    for (size_t i = 0; i < pin_nodes.size(); i++)
    {
        PinNode &pn = pin_nodes[i];
        // 非根模块
        if (pn.mod_id != -1)
        {
            // 对应模块信息
            Module_Info &mf = modules_info[pn.mod_id];
            // 模块位置
            pn.mod_x = mf.x, pn.mod_y = mf.y;
        }
        Pin &p = *(pin_list[i]); // 原始引脚对象
        // 更新引脚坐标
        pn.pin_x = p.ax, pn.pin_y = p.ay;
    }

    // 构造返回数组
    py::array_t<float> node_array(
        (vector<size_t>){pin_nodes.size(), 8}); // N x 8 的二维数组
    auto buf = node_array.mutable_unchecked<2>();
    for (size_t i = 0; i < pin_nodes.size(); i++)
    {
        buf(i, 0) = pin_nodes[i].pin_x;                      // 引脚X (绝对坐标)
        buf(i, 1) = pin_nodes[i].pin_y;                      // 引脚Y (绝对坐标)
        buf(i, 2) = pin_nodes[i].mod_x;                      // 模块X (绝对坐标)
        buf(i, 3) = pin_nodes[i].mod_y;                      // 模块Y (绝对坐标)
        buf(i, 4) = pin_nodes[i].mod_w;                      // 模块宽度
        buf(i, 5) = pin_nodes[i].mod_h;                      // 模块高度
        buf(i, 6) = static_cast<float>(pin_nodes[i].mod_id); // 模块ID
        buf(i, 7) = static_cast<float>(pin_nodes[i].net_id); // 网络ID
    }

    return node_array;
}

// 获得邻接矩阵（稠密形式）
py::array_t<float> B_Tree_Ext::get_py_adj_matrix()
{
    vector<size_t> arr_size = {pin_nodes.size(), pin_nodes.size()};
    py::array_t<bool> adj_matrix(arr_size);
    auto adj_buf = adj_matrix.mutable_unchecked<2>();
    for (Net &n : network)
    {
        // 一个网络内的引脚
        vector<int> pin_id_list(n.size());
        for (size_t i = 0; i < n.size(); i++)
            pin_id_list[i] = pin_table[n[i]];

        // 对网络内引脚的组合对，一对对标记
        for (size_t i = 0; i < pin_id_list.size() - 1; i++)
        {
            for (size_t j = i + 1; j < pin_id_list.size(); j++)
            {
                adj_buf(pin_id_list[i], pin_id_list[j]) += 1;
                adj_buf(pin_id_list[j], pin_id_list[i]) += 1;
            }
        }
    }
    return adj_matrix;
}

// 获得邻接矩阵（稀疏形式）
py::object B_Tree_Ext::get_py_adj_matrix_zip()
{
    size_t n = num_of_pins;
    typedef Eigen::Triplet<bool> T;
    std::vector<T> tripletList;
    // 指定一个合适的初始化大小（n*n*0.05）
    tripletList.reserve(size_t(n * n * 0.05));

    for (Net &n : network)
    {
        // 一个网络内的引脚
        vector<int> pin_id_list(n.size());
        for (size_t i = 0; i < n.size(); i++)
            pin_id_list[i] = pin_table[n[i]];

        // 对网络内引脚的组合对，一对对标记
        for (size_t i = 0; i < pin_id_list.size() - 1; i++)
        {
            for (size_t j = i + 1; j < pin_id_list.size(); j++)
            {
                tripletList.emplace_back(pin_id_list[i], pin_id_list[j], true);
                tripletList.emplace_back(pin_id_list[j], pin_id_list[i], true);
            }
        }
    }

    // cout << "tripletList length:" << tripletList.size() << endl;

    Eigen::SparseMatrix<bool> adj_matrix(n, n);
    adj_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return py::cast(adj_matrix);
}

vector<vector<int>> B_Tree_Ext::get_network_info()
{
    vector<vector<int>> res;
    for (Net &n : network)
    {
        // 一个网络内的引脚
        vector<int> pin_id_list(n.size());
        for (size_t i = 0; i < n.size(); i++)
        {
            pin_id_list[i] = pin_table[n[i]];
        }
        res.push_back(pin_id_list);
    }
    return res;
}

void B_Tree_Ext::update()
{
    B_Tree::perturb();
    B_Tree::packing();
    if (cost_alpha == 1)
    {
        // alpha = 1 时原代码不执行线长计算
        calcWireLength();
    }
}

vector<Action> B_Tree_Ext::perturb_gen(u_int32_t num)
{
    vector<Action> acts(num);
    for (Action &act : acts)
    {
        int n = rand() % modules_N, p;
        act.node1 = n; // 随机节点一

        // 一定概率选择旋转
        if (rotate_rate > rand_01())
        {
            act.type = ROTATE_FLIP;
            act.flip = rand_bool();
            continue;
        }

        // 一定概率选择交换节点
        if (swap_rate > rand_01())
        {
            do
            {
                p = rand() % modules_N;
            } while (n == p || nodes[n].parent == p || nodes[p].parent == n);
            act.type = SWAP_NODES;
            act.node2 = p;
            continue;
        }

        // 一定概率选择删除和插入节点
        {
            do
            {
                p = rand() % modules_N;
            } while (n == p);
            act.type = DELETE_INSERT;
            act.node2 = p;
        }
    }
    return acts;
}
void B_Tree_Ext::perturb_run(Action act)
{
    switch (act.type)
    {
    case ROTATE_FLIP:
        nodes[act.node1].rotate = !nodes[act.node1].rotate;
        if (act.flip)
            nodes[act.node1].flip = !nodes[act.node1].flip;
        break;

    case SWAP_NODES:
        if (nodes[act.node1].parent == act.node2 ||
            nodes[act.node2].parent == act.node1)
        {
            error("pertub_run error, Cannot swap parent and child.");
            return;
        }
        swap_node(nodes[act.node1], nodes[act.node2]);
        break;

    case DELETE_INSERT:
        if (act.node1 == act.node2)
        {
            error("pertub_run error, Cannot delete and insert same node.");
        }
        delete_node(nodes[act.node1]);
        insert_node(nodes[act.node2], nodes[act.node1]);
        break;

    default:
        error("pertub_run error, unknow type %s.",
              to_string(int(act.type)).c_str());
        break;
    }
}

FplanEnv::FplanEnv(std::string fn, float calpha, int max_times)
    : filename(fn), cost_alpha(calpha), max_times(max_times),
      t(0), cost_list(), has_rolled_back(false)
{
    bt = new B_Tree_Ext(fn, calpha);
    bt->normalize_cost(50);

    norm_a = bt->get_norm_area();
    norm_w = bt->get_norm_wire();

    init_cost = bt->getCost();
    double optimal_cost = run_with_sa(*bt).cost;
    baseline = 1 - (optimal_cost / init_cost);

    cout << "SA result: " << "area" << bt->getArea() << ", wirelen" << bt->getWireLength() << endl;

    reset();
}

py::array_t<double> FplanEnv::reset(int seed)
{
    if (seed > 0)
        srand(seed);
    else
        srand(time(0));

    if (bt != nullptr)
    {
        delete bt;
        bt = nullptr;
    }

    bt = new B_Tree_Ext(filename, cost_alpha);
    if (bt == nullptr)
        cout << "BTree reset error." << endl;

    bt->keep_sol();
    bt->packing();
    bt->set_normalize(norm_a, norm_w); // 同步归一化的基准值

    this->init_area = bt->getArea();
    this->init_wirelen = bt->getWireLength();

    t = 0;                                // 清空计数
    this->cost_list.clear();              // 清空成本列表
    this->cost_list.push_back(init_cost); // 加入成本列表

    this->has_rolled_back = false;
    this->pre_statu = this->go1step(bt->perturb_gen(1)[0]);

    return this->pre_statu[0].cast<py::array_t<double>>();
}

vector<Action> FplanEnv::act_gen_batch(u_int32_t num)
{
    return bt->perturb_gen(num);
}

py::tuple FplanEnv::go1step(Action act)
{
    double cur_cost = 0.0, next_cost = 0.0;
    Modules_Info cur_ms_info, next_ms_info;
    double changed_rate = 0.0, changed_area = 0.0;

    cur_cost = bt->getCost();
    cur_ms_info = bt->get_mods_info();

    bt->keep_sol();
    bt->perturb_run(act);
    bt->packing();

    next_cost = bt->getCost();
    next_ms_info = bt->get_mods_info();

    cost_list.push_back(next_cost);

    calc_d_mods_info(cur_ms_info, next_ms_info, changed_rate, changed_area);

    py::array_t<double> arr({this->s_dim});
    auto buf = arr.mutable_unchecked<1>();

    buf(0) = changed_rate;
    buf(1) = changed_area;
    buf(2) = norm_cost(cur_cost);
    buf(3) = norm_cost(next_cost);
    buf(4) = norm_cost(*std::min_element(cost_list.begin(), cost_list.end()));
    buf(5) = norm_cost(std::accumulate(cost_list.begin(), cost_list.end(), 0.0) / cost_list.size());
    buf(6) = this->t / this->max_times;

    double reward = (cur_cost - next_cost) / baseline;
    bool done = (t >= max_times);
    return py::make_tuple(arr, reward, done);
}

py::tuple FplanEnv::step(bool act_bool)
{
    t++;
    py::array_t<double> arr;
    double reward;
    bool done = (this->t >= this->max_times);
    if (act_bool)
    {
        this->has_rolled_back = false;
        py::tuple tup_t = this->pre_statu;
        this->pre_statu = go1step(bt->perturb_gen(1)[0]);

        arr = this->pre_statu[0].cast<py::array_t<double>>();

        reward = tup_t[1].cast<double>();
    }
    else
    {
        if (!has_rolled_back)
        {
            this->recover();
            this->has_rolled_back = true;
        }

        py::tuple tup = go1step(bt->perturb_gen(1)[0]);

        arr = tup[0].cast<py::array_t<double>>();
        auto buf = arr.mutable_unchecked<1>();
        buf(6) = this->t / this->max_times;

        double reward = 0.0;
    }
    return py::make_tuple(arr, reward, done);
}

void FplanEnv::recover()
{
    if (cost_list.size() <= 1)
        return;

    bt->recover();
    bt->packing();
    cost_list.pop_back();
}

void FplanEnv::calc_d_mods_info(
    Modules_Info &mods_a, Modules_Info &mods_b,
    double &changed_rate, double &changed_area)
{
    assert(mods_a.size() > 0 && mods_a.size() == mods_b.size());

    int cnt = 0, area_cnt = 0;
    for (size_t i = 0; i < mods_a.size(); i++)
    {
        if (mods_a[i].x != mods_b[i].x || mods_a[i].y != mods_b[i].y)
        {
            cnt++;
            area_cnt += bt->get_mod(i).area;
        }
    }
    changed_rate = double(cnt) / mods_a.size();
    changed_area = double(area_cnt) / bt->getTotalArea();
}

double FplanEnv::norm_cost(double cost)
{
    return std::min(1.0, cost / init_cost - 1);
}

void FplanEnv::show_info()
{
    cout << "--filename: " << this->filename;
    cout << " cost_alpha: " << this->cost_alpha << endl;
    // cout << "--t: \t" << this->t << endl;
    // cout << "--max_t: \t" << this->max_times << endl;
    // cout << "--init_cost: \t" << this->init_cost << endl;
    // cout << "--baseline: \t" << this->baseline << endl;
    cout << "--area change: " << this->init_area;
    cout << " --> " << bt->getArea() << endl;
    cout << "--wirelen change: " << this->init_wirelen;
    cout << " --> " << bt->getWireLength() << endl;
}