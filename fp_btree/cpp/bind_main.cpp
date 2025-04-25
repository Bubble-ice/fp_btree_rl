#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/SparseCore>

#include "include/api2py.h"

namespace py = pybind11;

PYBIND11_MODULE(fp_btree, m) {
    m.doc() = "B*-Tree source code";

    py::enum_<ActionType>(m, "ActionType")
        .value("ROTATE_FLIP", ActionType::ROTATE_FLIP)
        .value("SWAP_NODES", ActionType::SWAP_NODES)
        .value("DELETE_INSERT", ActionType::DELETE_INSERT)
        .export_values();  // 导出所有枚举值到 Python

    // 暴露 Action 结构体
    py::class_<Action>(m, "Action")
        .def(py::init<>())                     // 默认构造函数
        .def_readwrite("type", &Action::type)  // 读写字段
        .def_readwrite("node1", &Action::node1)
        .def_readwrite("node2", &Action::node2)
        .def_readwrite("flip", &Action::flip)
        // 可选：添加 __repr__ 方便调试
        .def("__repr__", [](const Action& a) {
            std::string type_str;
            switch (a.type) {
                case ROTATE_FLIP:
                    type_str = "ROTATE_FLIP";
                    break;
                case SWAP_NODES:
                    type_str = "SWAP_NODES";
                    break;
                case DELETE_INSERT:
                    type_str = "DELETE_INSERT";
                    break;
            }
            return "<Action type=" + type_str +
                   " node1=" + std::to_string(a.node1) +
                   " node2=" + std::to_string(a.node2) +
                   " flip=" + (a.flip ? "True" : "False") + ">";
        });

    py::class_<B_Tree_Ext>(m, "B_Tree_Ext")
        .def(py::init<char*, float>(), py::arg("filename"),
             py::arg("calpha") = 1.0f, "init from file")

        // 父类方法
        .def("get_area", &B_Tree_Ext::getArea, "")
        .def("get_total_area", &B_Tree_Ext::getTotalArea, "")
        .def("get_wire_length", &B_Tree_Ext::getWireLength, "")

        // 调试信息
        .def("show", &B_Tree_Ext::show, "")
        .def("get_filename", &B_Tree_Ext::get_filename, "")
        .def("get_network_info", &B_Tree_Ext::get_network_info, "")
        .def("get_net_table", &B_Tree_Ext::get_net_table, "")

        // 核心信息
        .def("get_pin_nodes_info", &B_Tree_Ext::get_py_pin_nodes_info, "")
        .def("get_adj_matrix", &B_Tree_Ext::get_py_adj_matrix, "")
        .def("get_adj_matrix_zip", &B_Tree_Ext::get_py_adj_matrix_zip, "")

        // 核心更新函数
        .def("update", &B_Tree_Ext::update, "");

    py::class_<FplanEnv>(m, "FplanEnv")
        .def(py::init<char*, float, int, bool>(), py::arg("fn"),
             py::arg("calpha") = 1.0f, py::arg("max_times") = 5000,
             py::arg("is_debug") = false, "init from bt")
        .def("reset", &FplanEnv::reset, py::arg("seed") = NIL, "")

        .def("act_gen_batch", &FplanEnv::act_gen_batch, "")

        .def("step", &FplanEnv::step, py::arg("act_bool"), "")
        .def("back1step", &FplanEnv::back1step, "")

        .def_readonly("state_dim", &FplanEnv::s_dim)
        .def("get_cost", &FplanEnv::get_cost, "")
        .def("get_init_cost", &FplanEnv::get_init_cost, "")
        .def("get_baseline", &FplanEnv::get_baseline, "")
        .def("get_cost_list", &FplanEnv::get_cost_list, "")
        .def("show_info", &FplanEnv::show_info, "");

    py::class_<SAResult>(m, "SAResult")
        .def_readonly("cpu_time", &SAResult::cpu_time)
        .def_readonly("last_cpu_time", &SAResult::last_cpu_time)
        .def_readonly("cost", &SAResult::cost)
        .def_readonly("area", &SAResult::area)
        .def_readonly("wire_length", &SAResult::wire_length)
        .def_readonly("dead_space", &SAResult::dead_space);

    m.def("run_with_sa", &run_with_sa, py::arg("fp"), py::arg("times") = 400,
          py::arg("local") = 7, py::arg("init_temp") = 0.9f,
          py::arg("term_temp") = 0.1f, py::arg("alpha") = 1.0f,
          py::arg("outfile") = "", py::arg("is_debug") = false,
          "Run simulated annealing floorplanning");
}