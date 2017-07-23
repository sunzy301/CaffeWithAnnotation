/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *   自定义的solver名称必须是Solver结尾
 *   不然就不能使用宏函数进行注册
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *    注册一个类
 *    去掉最后面的Solver
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 * 注册一个函数
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

// 工厂模式思想
// 所有的solver类型都需要在SolverRegistry类中注册
// 需要的时候使用注册过的名称和生成函数生成这个类

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
   //Creator为函数指针类型；CreatorRegistry为标准map容器，储存函数指针；
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  //创建map
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.添加一个solver
  // type是solver名称
  // creator是函数指针
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  // param.type是solver名称
  // 返回调用creator函数创建solver
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    // 调用函数，返回solver
	return registry[type](param);
  }

  // 返回solver类型名称列表
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  // 工厂类不能被实例化，所有操作都只能是静态操作
  // =delete
  SolverRegistry() {}

  // 将solver类型名称从vector转为string，仅作输出使用
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};

// solver注册类
template <typename Dtype>
class SolverRegisterer {
 public:
  // type是solver函数名称，creator是函数指针
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

// 两个宏函数
// 注册函数指针
// 用名称和函数指针生成SolverRegisterer类，从而再调用SolverRegistry::AddCreator函数
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

// 注册一个solver类
// ## 是宏函数参数中的连接符，可以连接两个token
// type是自定义solver类名称去掉最后Solver剩下的部分
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
