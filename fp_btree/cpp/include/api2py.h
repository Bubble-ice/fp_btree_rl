#ifndef API2PY_H
#define API2PY_H
#include <map>
#include <vector>
#include <algorithm>

#include <Eigen/SparseCore>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "btree.h"

namespace py = pybind11;

struct PinNode
{
  int id;
  int mod_id;
  int net_id;
  int mod_w, mod_h;
  int mod_x, mod_y; // absolute position
  int pin_x, pin_y; // absolute position
};

enum ActionType
{
  ROTATE_FLIP,  // 旋转 + 可能翻转
  SWAP_NODES,   // 交换两个节点
  DELETE_INSERT // 删除并重新插入
};

struct Action
{
  ActionType type;
  int node1;         // 主节点
  int node2 = -1;    // 第二节点
  bool flip = false; // 是否翻转
};

class B_Tree_Ext : public B_Tree
{
public:
  B_Tree_Ext(string fn, float calpha = 1);

  void show() { show_tree(); };
  string get_filename() { return filename; };
  vector<vector<int>> get_network_info();
  map<string, int> get_net_table() { return net_table; };
  Modules_Info get_mods_info() { return modules_info; };
  Module &get_mod(int idx) { return modules[idx]; };

  double get_norm_area() { return norm_area; };
  double get_norm_wire() { return norm_wire; };
  void set_normalize(double norm_a, double norm_w)
  {
    norm_area = norm_a, norm_wire = norm_w;
  };

  py::array_t<float> get_py_pin_nodes_info();
  py::array_t<float> get_py_adj_matrix();
  py::object get_py_adj_matrix_zip();

  void update();

  vector<Action> perturb_gen(u_int32_t num = 1);
  void perturb_run(Action act);

  // void reset(const char *fn, float calpha);

protected:
  int num_of_pins;
  vector<PinNode> pin_nodes; // 引脚节点列表
  map<Pin_p, int> pin_table; // 原始引脚-引脚节点的索引
  vector<Pin_p> pin_list;    // 引脚节点-原始引脚的索引

  void pin_nodes_init();
};

class FplanEnv
{
public:
  FplanEnv(std::string fn, float calpha = 1, int max_times = 5000);
  py::array_t<double> reset(int seed = NIL);

  vector<Action> act_gen_batch(u_int32_t num = 1);

  py::tuple go1step(Action act);
  py::tuple step(bool act_bool);

  void recover();

  double get_cost() { return bt->getCost(); };
  double get_init_cost() { return init_cost; };
  double get_baseline() { return baseline; };
  vector<double> get_cost_list() { return cost_list; };

  void show_info();

  const uint32_t s_dim = 6;

protected:
  B_Tree_Ext *bt;

  std::string filename;
  float cost_alpha;
  double norm_a, norm_w;

  int max_times;
  int t;

  vector<double> cost_list;
  double init_cost;

  double baseline;

  bool has_rolled_back;

  double init_area, init_wirelen;

  double norm_cost(double cost);
  void calc_d_mods_info(
      Modules_Info &mods_a,
      Modules_Info &mods_b,
      double &changed_rate,
      double &changed_area);
};

// 返回结构体包含所有结果信息
struct SAResult
{
  double cpu_time;
  double last_cpu_time;
  float cost;
  float area;
  float wire_length;
  float dead_space;
};

SAResult run_with_sa(B_Tree_Ext fp, int times = 400, int local = 7,
                     float init_temp = 0.9f, float term_temp = 0.1f,
                     float alpha = 1.0f, const char *outfile = "", bool is_debug = false);

#endif